""" 
Training Script for DDPM (Denoising Diffusion Probabilistic Models)

This is a config-tolerant training script (won't crash on missing YAML keys).
It supports:
- Mixed precision (AMP)
- EMA (exponential moving average)
- Gradient clipping
- Periodic checkpoints + sampling
- Optional torchrun DistributedDataParallel

Usage:
  python train.py --method ddpm --config configs/ddpm_modal.yaml
  python train.py --method ddpm --config configs/ddpm_modal.yaml --overfit-single-batch
  python train.py --method ddpm --config configs/ddpm_modal.yaml --resume logs/.../checkpoints/ddpm_0001000.pt
"""

import os
import argparse
import math
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import create_dataloader_from_config, save_image, unnormalize
from src.methods.ddpm_x0 import DDPM_X0
from src.utils import EMA

# Optional deps
try:
    import wandb  # type: ignore
except Exception:
    wandb = None

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:
    PILImage = None


# -------------------------
# Config + logging
# -------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict, method_name: str) -> tuple[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging_cfg = config.get("logging", {}) or {}
    log_root = logging_cfg.get("dir", "./logs")
    log_dir = os.path.join(log_root, f"{method_name}_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Logging to: {log_dir}")

    wandb_run = None
    wandb_cfg = logging_cfg.get("wandb", {})
    # allow wandb: true/false
    if isinstance(wandb_cfg, bool):
        wandb_cfg = {"enabled": wandb_cfg}

    if wandb_cfg.get("enabled", False) and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "cmu-10799-diffusion"),
                entity=wandb_cfg.get("entity", None),
                name=wandb_cfg.get("name", f"{method_name}_{timestamp}"),
                config=config,
                dir=log_dir,
                tags=[method_name],
            )
            print(f"Weights & Biases: {wandb_run.url}")
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")
            wandb_run = None

    return log_dir, wandb_run


# -------------------------
# Distributed helpers
# -------------------------

def get_distributed_context() -> tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def reduce_metrics(metrics: Dict[str, Any], device: torch.device, world_size: int) -> Dict[str, float]:
    if world_size < 2 or not dist.is_initialized():
        return {k: (v.detach().item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        t = v.detach() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
        if t.device != device:
            t = t.to(device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = (t / world_size).item()
    return out


# -------------------------
# Optimizer + checkpointing
# -------------------------

def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {}) or {}
    lr = training_cfg.get("learning_rate", training_cfg.get("lr", 2e-4))
    betas = training_cfg.get("betas", (0.9, 0.999))
    weight_decay = training_cfg.get("weight_decay", 0.0)

    if not (isinstance(betas, (list, tuple)) and len(betas) == 2):
        betas = (0.9, 0.999)
    betas = (float(betas[0]), float(betas[1]))

    return torch.optim.AdamW(model.parameters(), lr=float(lr), betas=betas, weight_decay=float(weight_decay))


def save_checkpoint(path: str, model: nn.Module, optimizer, ema: Optional[EMA], scaler: GradScaler, step: int, config: dict):
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "config": config,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer, ema: Optional[EMA], scaler: GradScaler, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt.get("scaler", {}))
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    step = int(ckpt.get("step", 0))
    print(f"Loaded checkpoint from {path} at step {step}")
    return step


# -------------------------
# Sampling + saving
# -------------------------

def _method_eval(method):
    if hasattr(method, "eval_mode"):
        method.eval_mode()
    else:
        unwrap_model(method.model).eval()


def _method_train(method):
    if hasattr(method, "train_mode"):
        method.train_mode()
    else:
        unwrap_model(method.model).train()


@torch.no_grad()
def generate_samples(
    method,
    num_samples: int,
    image_shape: Tuple[int, int, int],
    config: dict,
    ema: Optional[EMA] = None,
    current_step: Optional[int] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    _method_eval(method)

    training_cfg = config.get("training", {}) or {}
    sampling_cfg = config.get("sampling", {}) or {}

    ema_start = int(training_cfg.get("ema_start", 0))
    use_ema = ema is not None and (current_step is None or current_step >= ema_start)
    if use_ema:
        ema.apply_shadow()

    num_steps = sampling_kwargs.get("num_steps", sampling_cfg.get("num_steps", getattr(method, "num_timesteps", 1000)))
    eta = sampling_kwargs.get("eta", sampling_cfg.get("eta", 0.0))

    # Preferred signature (what our HW code usually expects)
    try:
        samples = method.sample(
            batch_size=num_samples,
            image_shape=image_shape,
            num_steps=num_steps,
            eta=eta,
        )
    except TypeError:
        # Fallback for older positional signature
        samples = method.sample(num_samples, image_shape, num_steps)

    if samples is None:
        raise ValueError("generate_samples got samples=None. Implement DDPM.sample() to return a tensor.")

    if use_ema:
        ema.restore()

    _method_train(method)
    return samples


def save_samples(samples: torch.Tensor, save_path: str, num_samples: int) -> None:
    samples = samples.detach().cpu()
    # samples are typically in [-1, 1]; map to [0, 1]
    samples = unnormalize(samples).clamp(0.0, 1.0)
    nrow = max(1, int(math.sqrt(num_samples)))
    save_image(samples, save_path, nrow=nrow)


# -------------------------
# Training loop
# -------------------------

def train(method_name: str, config: dict, resume_path: Optional[str] = None, overfit_single_batch: bool = False):
    rank, world_size, local_rank = get_distributed_context()

    infra_cfg = config.get("infrastructure", {}) or {}
    training_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}

    config_device = infra_cfg.get("device", "cuda")
    config_num_gpus = infra_cfg.get("num_gpus", None)

    config_allows_distributed = (config_device != "cpu") and (config_num_gpus is None or int(config_num_gpus) > 1)
    is_distributed = world_size > 1 and config_allows_distributed
    is_main = rank == 0

    # Device
    if is_distributed:
        if not torch.cuda.is_available() or config_device == "cpu":
            raise RuntimeError("Distributed training requires CUDA and infrastructure.device='cuda'.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        use_cuda = torch.cuda.is_available() and config_device != "cpu"
        device = torch.device("cuda" if use_cuda else "cpu")

    mixed_precision = bool(infra_cfg.get("mixed_precision", False))

    if is_main:
        print("=" * 60)
        print("DEVICE CONFIGURATION")
        print("=" * 60)
        if is_distributed:
            print("✓ Distributed training")
            print(f"  - World size: {world_size}")
            print(f"  - Device: {device}")
        else:
            if device.type == "cuda":
                print("✓ Single GPU training")
                print(f"  - Device: cuda ({torch.cuda.get_device_name(device)})")
            else:
                print("✓ CPU training")
                print(f"  - Device: {device}")
        print(f"  - Config device setting: {config_device}")
        print(f"  - Mixed precision: {mixed_precision}")
        print("=" * 60)

    # Seed
    seed = int(infra_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Data
    if is_main:
        print("Creating data loader...")
    dataloader = create_dataloader_from_config(config, split="train")

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataloader.dataset,
            batch_size=int(training_cfg.get("batch_size", 64)),
            sampler=sampler,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=True,
        )

    if is_main:
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")

    # Model
    if is_main:
        print("Creating model...")
    base_model = create_model_from_config(config).to(device)
    model = base_model
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        nparams = sum(p.numel() for p in base_model.parameters())
        print(f"Model parameters: {nparams:,} ({nparams/1e6:.2f}M)")

    # Method
    if is_main:
        print(f"Creating {method_name}...")

    if method_name != "ddpm":
        raise ValueError(f"Unknown method: {method_name}. Only 'ddpm' is supported.")

    if False  # DDPM_X0 uses direct ctor:
        method = DDPM_X0.from_config(model, config, device)
    else:
        ddpm_cfg = config.get("ddpm", {}) or {}
        method = DDPM_X0(
            model=model,
            device=device,
            num_timesteps=int(ddpm_cfg.get("num_timesteps", 1000)),
            beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
            beta_end=float(ddpm_cfg.get("beta_end", 2e-2)),
        )

    # Ensure buffers on device if method provides .to
    if hasattr(method, "to"):
        method = method.to(device)

    # Optimizer, EMA, scaler
    optimizer = create_optimizer(model, config)
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    ema = EMA(unwrap_model(model), decay=ema_decay)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=mixed_precision)

    # Logging
    log_dir, wandb_run = (None, None)
    if is_main:
        log_dir, wandb_run = setup_logging(config, method_name)

    # Resume
    start_step = 0
    if resume_path is not None:
        if is_distributed:
            dist.barrier()
        start_step = load_checkpoint(resume_path, model, optimizer, ema, scaler, device)

    # Schedule
    num_iterations = int(training_cfg.get("num_iterations", training_cfg.get("max_steps", 2000)))
    log_every = int(training_cfg.get("log_every", 50))
    sample_every = training_cfg.get("sample_every", training_cfg.get("eval_every", 200))
    save_every = training_cfg.get("save_every", training_cfg.get("checkpoint_every", 1000))

    sample_every = None if sample_every is None or int(sample_every) <= 0 else int(sample_every)
    save_every = None if save_every is None or int(save_every) <= 0 else int(save_every)

    num_samples = int(training_cfg.get("num_samples", (config.get("sampling", {}) or {}).get("num_samples", 64)))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))

    channels = int(data_cfg.get("channels", 3))
    image_size = int(data_cfg.get("image_size", 64))
    image_shape = (channels, image_size, image_size)

    if is_main:
        print(f"\nStarting training from step {start_step}...")
        print(f"Total iterations: {num_iterations}")
        if overfit_single_batch:
            print("DEBUG MODE: Overfitting to a single batch")
        print("-" * 50)

    _method_train(method)

    data_iter = iter(dataloader)
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)

    # Overfit cache
    single_batch = None
    if overfit_single_batch:
        single_batch = next(data_iter)
        if isinstance(single_batch, (tuple, list)):
            single_batch = single_batch[0]
        single_batch = single_batch.to(device)

        desired = int(training_cfg.get("batch_size", single_batch.shape[0]))
        if desired > single_batch.shape[0]:
            reps = (desired + single_batch.shape[0] - 1) // single_batch.shape[0]
            single_batch = single_batch.repeat(reps, 1, 1, 1)[:desired]

        if is_main:
            print(f"Cached single batch with shape: {single_batch.shape}")

    # Metric accumulator
    metrics_sum: Dict[str, list] = {}
    metrics_count = 0
    t0 = time.time()

    pbar = tqdm(range(start_step, num_iterations), initial=start_step, total=num_iterations, disable=not is_main)
    for step in pbar:
        # Batch
        if overfit_single_batch:
            batch = single_batch
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type, enabled=mixed_precision):
            loss, metrics = method.compute_loss(batch)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        ema.update()

        for k, v in metrics.items():
            metrics_sum.setdefault(k, []).append(v.detach().item() if torch.is_tensor(v) else float(v))
        metrics_count += 1

        # Console + wandb log
        if log_every and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = metrics_count / max(1e-9, elapsed)
            local_avg = {k: sum(v) / len(v) for k, v in metrics_sum.items()}
            avg = reduce_metrics(local_avg, device, world_size)

            if is_main:
                pbar.set_postfix({
                    "loss": f"{avg.get('loss', float(loss.detach().item())):.4f}",
                    "steps/s": f"{steps_per_sec:.2f}",
                })

            if is_main and wandb_run is not None and wandb is not None:
                log_dict = {"train/step": step + 1, "train/steps_per_sec": steps_per_sec, "train/lr": optimizer.param_groups[0]["lr"]}
                for k, v in avg.items():
                    log_dict[f"train/{k}"] = v
                try:
                    wandb.log(log_dict, step=step + 1)
                except Exception as e:
                    print(f"Warning: wandb.log failed: {e}")

            metrics_sum = {}
            metrics_count = 0
            t0 = time.time()

        # Sampling
        if sample_every and (step + 1) % sample_every == 0:
            if is_main:
                print(f"\nGenerating samples at step {step + 1}...")
                samples = generate_samples(method, num_samples, image_shape, config, ema=ema, current_step=step + 1)
                sample_path = os.path.join(log_dir, "samples", f"samples_{step + 1:07d}.png")
                save_samples(samples, sample_path, num_samples)

                if wandb_run is not None and wandb is not None and PILImage is not None:
                    try:
                        img = PILImage.open(sample_path)
                        wandb.log({"samples": wandb.Image(img, caption=f"Step {step + 1}")}, step=step + 1)
                    except Exception as e:
                        print(f"Warning: wandb image log failed: {e}")

            if is_distributed:
                dist.barrier()

        # Checkpoint
        if save_every and (step + 1) % save_every == 0:
            if is_main:
                ckpt_path = os.path.join(log_dir, "checkpoints", f"{method_name}_{step + 1:07d}.pt")
                save_checkpoint(ckpt_path, model, optimizer, ema, scaler, step + 1, config)
            if is_distributed:
                dist.barrier()

    # Final checkpoint
    if is_main:
        final_path = os.path.join(log_dir, "checkpoints", f"{method_name}_final.pt")
        save_checkpoint(final_path, model, optimizer, ema, scaler, num_iterations, config)
        print("\nTraining complete!")
        print(f"Final checkpoint: {final_path}")
        print(f"Samples saved to: {os.path.join(log_dir, 'samples')}")

    if is_main and wandb_run is not None and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    if is_distributed:
        dist.barrier()
        cleanup_distributed(is_distributed)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion models")
    parser.add_argument("--method", type=str, required=True, choices=["ddpm"], help="Method to train")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--overfit-single-batch", action="store_true", help="Overfit to one batch for debugging")
    args = parser.parse_args()

    config = load_config(args.config)
    train(args.method, config, resume_path=args.resume, overfit_single_batch=args.overfit_single_batch)


if __name__ == "__main__":
    main()
