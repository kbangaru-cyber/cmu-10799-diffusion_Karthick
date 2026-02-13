"""
Training Script for CFG Flow Matching (Conditional)

Trains a conditional flow matching model with classifier-free guidance on
CelebA 64×64 with binary attribute labels.

Key differences from the base train.py:
  - Uses CondUNet instead of UNet (accepts attribute vector c)
  - Uses CelebAWithAttributes dataset (returns image + attributes)
  - Passes attributes to method.compute_loss(batch, c=attrs)
  - Condition dropout (p_uncond) is handled inside CFGFlowMatching

Usage:
  python train_cfg.py --config configs/cfg_flow_matching.yaml
  python train_cfg.py --config configs/cfg_flow_matching.yaml --overfit-single-batch
  python train_cfg.py --config configs/cfg_flow_matching.yaml --resume logs/.../checkpoints/cfg_fm_XXXX.pt
"""

import os
import argparse
import math
import time
import csv
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet_cond import create_cond_model_from_config
from src.data.celeba_cond import create_cond_dataloader_from_config
from src.methods.cfg_flow_matching import CFGFlowMatching
from src.utils import EMA

# Try to import image saving from your existing code
try:
    from src.data import save_image, unnormalize
except ImportError:
    from torchvision.utils import save_image as _tv_save_image

    def unnormalize(x):
        return x * 0.5 + 0.5

    def save_image(tensor, path, nrow=8):
        _tv_save_image(tensor, path, nrow=nrow)

# Optional
try:
    import wandb
except Exception:
    wandb = None

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict, method_name: str) -> tuple:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_cfg = config.get("logging", {}) or {}
    log_root = logging_cfg.get("dir", "./logs")
    log_dir = os.path.join(log_root, f"{method_name}_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Logging to: {log_dir}")
    return log_dir, None  # wandb_run = None for simplicity


def save_checkpoint(path, model, optimizer, ema, scaler, step, config):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "config": config,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer, ema, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt.get("scaler", {}))
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    step = int(ckpt.get("step", 0))
    print(f"Loaded checkpoint from {path} at step {step}")
    return step


@torch.no_grad()
def generate_samples(
    method: CFGFlowMatching,
    num_samples: int,
    image_shape: Tuple[int, int, int],
    config: dict,
    ema: Optional[EMA] = None,
    current_step: Optional[int] = None,
) -> torch.Tensor:
    """Generate samples with different attribute conditions for visualization."""
    method.eval_mode()

    training_cfg = config.get("training", {}) or {}
    sampling_cfg = config.get("sampling", {}) or {}
    cond_cfg = config.get("conditioning", {}) or {}

    ema_start = int(training_cfg.get("ema_start", 0))
    use_ema = ema is not None and (current_step is None or current_step >= ema_start)
    if use_ema:
        ema.apply_shadow()

    num_steps = int(sampling_cfg.get("num_steps", 200))
    num_attrs = int(cond_cfg.get("num_attributes", 3))
    attr_names = cond_cfg.get("attributes", ["Smiling", "Eyeglasses", "Male"])

    device = method.device

    # Generate samples with various attribute combinations
    # Row 1: unconditional (all zeros)
    # Row 2: Smiling only
    # Row 3: Eyeglasses only
    # Row 4: Male only
    # Row 5+: combinations
    samples_per_row = min(8, num_samples // 4)
    all_samples = []

    conditions = [
        torch.zeros(samples_per_row, num_attrs, device=device),  # unconditional
    ]
    # Single attributes on
    for i in range(min(num_attrs, 3)):
        c = torch.zeros(samples_per_row, num_attrs, device=device)
        c[:, i] = 1.0
        conditions.append(c)

    for c in conditions:
        try:
            s = method.sample(
                batch_size=c.shape[0],
                image_shape=image_shape,
                num_steps=num_steps,
                c=c,
            )
            all_samples.append(s)
        except Exception as e:
            print(f"Warning: sample generation failed: {e}")
            break

    if use_ema:
        ema.restore()

    method.train_mode()

    if all_samples:
        return torch.cat(all_samples, dim=0)[:num_samples]
    else:
        return torch.randn(num_samples, *image_shape)


def save_samples(samples, save_path, num_samples):
    samples = samples.detach().cpu()
    samples = unnormalize(samples).clamp(0.0, 1.0)
    nrow = max(1, int(math.sqrt(num_samples)))
    save_image(samples, save_path, nrow=nrow)


def train(config: dict, resume_path: Optional[str] = None, overfit_single_batch: bool = False):
    """Main training loop for CFG Flow Matching."""

    infra_cfg = config.get("infrastructure", {}) or {}
    training_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}
    cond_cfg = config.get("conditioning", {}) or {}

    # Device
    use_cuda = torch.cuda.is_available() and infra_cfg.get("device", "cuda") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    mixed_precision = bool(infra_cfg.get("mixed_precision", False))

    print("=" * 60)
    print("CFG FLOW MATCHING TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mixed precision: {mixed_precision}")

    # Seed
    seed = int(infra_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Data (conditional dataloader)
    print("Creating conditional data loader...")
    dataloader = create_cond_dataloader_from_config(config, split="train")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Attributes: {cond_cfg.get('attributes', ['Smiling', 'Eyeglasses', 'Male'])}")

    # Model (conditional UNet)
    print("Creating conditional UNet...")
    model = create_cond_model_from_config(config).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {nparams:,} ({nparams / 1e6:.2f}M)")

    # Method (CFG Flow Matching)
    print("Creating CFG Flow Matching method...")
    method = CFGFlowMatching.from_config(model, config, device)

    # Optimizer
    lr = float(training_cfg.get("learning_rate", 2e-4))
    betas = training_cfg.get("betas", [0.9, 0.999])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=weight_decay,
    )

    # EMA
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    ema = EMA(model, decay=ema_decay)

    # Scaler
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=mixed_precision)

    # Logging
    method_name = "cfg_flow_matching"
    log_dir, wandb_run = setup_logging(config, method_name)

    # CSV loggers
    loss_csv_f = open(os.path.join(log_dir, "train_loss.csv"), "w", newline="")
    loss_writer = csv.DictWriter(loss_csv_f, fieldnames=["step", "epoch", "loss"])
    loss_writer.writeheader()
    loss_csv_f.flush()

    # Resume
    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, model, optimizer, ema, scaler, device)

    # Training params
    num_iterations = int(training_cfg.get("num_iterations", 5000))
    log_every = int(training_cfg.get("log_every", 50))
    sample_every = training_cfg.get("sample_every", 2500)
    sample_every = int(sample_every) if sample_every else None
    save_every = training_cfg.get("save_every", 2500)
    save_every = int(save_every) if save_every else None
    num_samples = int(training_cfg.get("num_samples", 32))
    grad_clip = float(training_cfg.get("gradient_clip_norm", 1.0))

    channels = int(data_cfg.get("channels", 3))
    image_size = int(data_cfg.get("image_size", 64))
    image_shape = (channels, image_size, image_size)

    print(f"\nStarting training from step {start_step}...")
    print(f"Total iterations: {num_iterations}")
    print(f"p_uncond: {method.p_uncond}")
    print(f"Default guidance scale: {method.guidance_scale}")
    if overfit_single_batch:
        print("DEBUG MODE: Overfitting to a single batch")
    print("-" * 50)

    method.train_mode()
    data_iter = iter(dataloader)
    epoch = 0

    # Overfit cache
    single_batch_img = None
    single_batch_attr = None

    if overfit_single_batch:
        batch = next(data_iter)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            single_batch_img, single_batch_attr = batch[0].to(device), batch[1].to(device)
        else:
            single_batch_img = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            single_batch_attr = torch.zeros(single_batch_img.shape[0], method.num_attributes, device=device)
        print(f"Cached single batch: images={single_batch_img.shape}, attrs={single_batch_attr.shape}")

    metrics_sum = {}
    metrics_count = 0
    t0 = time.time()

    pbar = tqdm(range(start_step, num_iterations), initial=start_step, total=num_iterations)
    try:
        for step in pbar:
            # Get batch
            if overfit_single_batch:
                images, attrs = single_batch_img, single_batch_attr
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    images, attrs = batch[0].to(device), batch[1].to(device)
                else:
                    images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
                    attrs = torch.zeros(images.shape[0], method.num_attributes, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type, enabled=mixed_precision):
                loss, metrics = method.compute_loss(images, c=attrs)

            # CSV logging
            loss_val = float(loss.detach().item())
            loss_writer.writerow({"step": step + 1, "epoch": epoch, "loss": loss_val})

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            ema.update()

            for k, v in metrics.items():
                metrics_sum.setdefault(k, []).append(
                    v.detach().item() if torch.is_tensor(v) else float(v)
                )
            metrics_count += 1

            # Logging
            if log_every and (step + 1) % log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = metrics_count / max(1e-9, elapsed)
                avg = {k: sum(v) / len(v) for k, v in metrics_sum.items()}
                avg_loss = avg.get("loss", loss_val)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "s/s": f"{steps_per_sec:.1f}"})
                metrics_sum = {}
                metrics_count = 0
                t0 = time.time()

            # Flush CSV
            if (step + 1) % 200 == 0:
                loss_csv_f.flush()

            # Sampling
            if sample_every and (step + 1) % sample_every == 0:
                print(f"\nGenerating samples at step {step + 1}...")
                samples = generate_samples(
                    method, num_samples, image_shape, config,
                    ema=ema, current_step=step + 1,
                )
                sample_path = os.path.join(log_dir, "samples", f"samples_{step + 1:07d}.png")
                save_samples(samples, sample_path, num_samples)
                print(f"Saved to {sample_path}")

            # Checkpoint
            if save_every and (step + 1) % save_every == 0:
                ckpt_path = os.path.join(
                    log_dir, "checkpoints", f"{method_name}_{step + 1:07d}.pt"
                )
                save_checkpoint(ckpt_path, model, optimizer, ema, scaler, step + 1, config)

    finally:
        loss_csv_f.flush()
        loss_csv_f.close()

    # Final checkpoint
    final_path = os.path.join(log_dir, "checkpoints", f"{method_name}_final.pt")
    save_checkpoint(final_path, model, optimizer, ema, scaler, num_iterations, config)
    print("\nTraining complete!")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CFG Flow Matching")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--overfit-single-batch", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_path=args.resume, overfit_single_batch=args.overfit_single_batch)


if __name__ == "__main__":
    main()
