"""
Training Script for DDPM (Denoising Diffusion Probabilistic Models)

This script provides a training loop for DDPM.
It supports:
- Mixed precision training (AMP)
- Exponential Moving Average (EMA)
- Gradient clipping
- Periodic checkpointing
- MultiGPU training
- Periodic sampling to check the generation quality
- Logging to console and optionally wandb

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging

Usage:
    # Train DDPM
    python train.py --method ddpm --config configs/ddpm.yaml

    # Resume training
    python train.py --method ddpm --config configs/ddpm.yaml --resume checkpoints/ddpm_50000.pt
"""

import os
import sys
import argparse
import math
import time
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models import UNet, create_model_from_config
from src.data import create_dataloader_from_config, save_image, unnormalize
from src.methods import DDPM
from src.utils import EMA

import wandb
from PIL import Image as PILImage

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict, method_name: str) -> tuple[str, Any]:
    """Set up logging directories and wandb. Returns (log_dir, wandb_run)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = config.get('logging', {}).get('dir', './logs')
    log_dir = os.path.join(log_root, f"{method_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"Logging to: {log_dir}")

    # Initialize wandb if enabled
    wandb_run = None
    wandb_config = config.get('logging', {}).get('wandb', {})
    if isinstance(wandb_config, bool):
        wandb_config = {'enabled': wandb_config}
    if wandb_config.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_config.get('project', 'cmu-10799-diffusion'),
                entity=wandb_config.get('entity', None),
                name=f"{method_name}_{timestamp}",
                config=config,
                dir=log_dir,
                tags=[method_name],
            )
            print(f"Weights & Biases: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")

    return log_dir, wandb_run


def get_distributed_context() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from torchrun env vars if set."""
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed(is_distributed: bool) -> None:
    """Tear down the process group if initialized."""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by DDP."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def reduce_metrics(
    metrics: Dict[str, Any],
    device: torch.device,
    world_size: int,
) -> Dict[str, float]:
    """Average metrics across ranks for consistent logging."""
    if world_size < 2 or not dist.is_initialized():
        return {
            k: (v.detach().item() if torch.is_tensor(v) else float(v))
            for k, v in metrics.items()
        }

    reduced: Dict[str, float] = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            tensor = v.detach()
        else:
            tensor = torch.tensor(v, dtype=torch.float32)
        if tensor.device != device:
            tensor = tensor.to(device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
        reduced[k] = tensor.item()
    return reduced


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config (robust to schema variations)."""
    training_config = config.get('training', {}) or {}
    lr = training_config.get('learning_rate', training_config.get('lr', 2e-4))
    betas = training_config.get('betas', (0.9, 0.999))
    weight_decay = training_config.get('weight_decay', 0.0)

    # Normalize betas to a tuple[float,float]
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas = (float(betas[0]), float(betas[1]))
    else:
        betas = (0.9, 0.999)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        betas=betas,
        weight_decay=float(weight_decay),
    )
    return optimizer


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    scaler: GradScaler,
    step: int,
    config: dict,
):
    """Save training checkpoint."""
    model_to_save = unwrap_model(model)
    checkpoint = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'config': config,
    }
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    scaler: GradScaler,
    device: torch.device,
) -> int:
    """Load training checkpoint and return the step."""
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if ema is not None and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
    scaler.load_state_dict(checkpoint['scaler'])
    step = checkpoint['step']
    print(f"Loaded checkpoint from {path} at step {step}")
    return step


@torch.no_grad()
def generate_samples(
    method,
    num_samples: int,
    image_shape: tuple,
    device: torch.device,
    method_name: str,
    config: dict,
    ema: Optional[EMA] = None,
    current_step: Optional[int] = None,
    # TODO: add/delete your arguments here
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Generate samples using EMA parameters if available.

    TODO: incoporate your own sampling scheme here

    Args:
        method: The diffusion method object (e.g., DDPM) with sample() and eval/train mode methods.
        num_samples: Number of samples to generate.
        image_shape: Shape of each image as (channels, height, width).
        device: The torch device to generate samples on.
        method_name: Name of the method being used (e.g., 'ddpm').
        config: Configuration dictionary containing training and model settings.
        ema: Optional EMA wrapper for the model. If provided and conditions are met,
            EMA parameters will be used during sampling.
        current_step: Current training step. Used to determine if EMA should be applied
            based on ema_start config.
        **sampling_kwargs: Additional keyword arguments passed to method.sample().

    Returns:
        torch.Tensor: Generated samples with shape (num_samples, *image_shape).
    """
    method.eval_mode()

    ema_start = config.get('training', {}).get('ema_start', 0)
    use_ema = ema is not None and (current_step is None or current_step >= ema_start)
    if use_ema:
        ema.apply_shadow()

    samples = None
    # TODO: sample with your method.sample()

    if use_ema:
        ema.restore()

    method.train_mode()
    return samples


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
) -> None:
    """
    TODO: save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
    """

    raise NotImplementedError


def train(
    method_name: str,
    config: dict,
    resume_path: Optional[str] = None,
    overfit_single_batch: bool = False,
):
    """
    Main training loop.

    Args:
        method_name: 'ddpm' (you can add more later)
        config: Configuration dictionary
        resume_path: Path to checkpoint to resume from
        overfit_single_batch: If True, train on a single batch repeatedly for debugging
    """
    # Auto-detect distributed setup from environment
    rank, world_size, local_rank = get_distributed_context()

    # Check if config wants distributed training
    config_device = (config.get('infrastructure', {}) or {}).get('device', 'cuda')
    config_num_gpus = (config.get('infrastructure', {}) or {}).get('num_gpus', None)

    # Distributed only if: world_size > 1 AND config allows it
    # Config disables distributed if: device='cpu' OR num_gpus=1
    config_allows_distributed = (
        config_device != 'cpu' and
        (config_num_gpus is None or config_num_gpus > 1)
    )
    is_distributed = world_size > 1 and config_allows_distributed
    is_main_process = rank == 0

    # Distributed training requires CUDA
    if is_distributed and (not torch.cuda.is_available() or config_device == 'cpu'):
        raise RuntimeError(
            "Distributed training requires CUDA. Either:\n"
            "  1. Run without torchrun for CPU training, or\n"
            "  2. Set infrastructure.device='cuda' in config and ensure GPUs are available"
        )

    # Determine device
    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')
    else:
        use_cuda = torch.cuda.is_available() and config_device != 'cpu'
        device = torch.device('cuda' if use_cuda else 'cpu')

    if is_main_process:
        print("=" * 60)
        print("DEVICE CONFIGURATION")
        print("=" * 60)
        if is_distributed:
            print(f"✓ Distributed training enabled")
            print(f"  - World size: {world_size} GPU(s)")
            print(f"  - Backend: {backend}")
            print(f"  - Device: {device}")
        else:
            if device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(device)
                print(f"✓ Single GPU training")
                print(f"  - Device: {device} ({gpu_name})")
            else:
                print(f"✓ CPU training")
                print(f"  - Device: {device}")
        print(f"  - Config device setting: {config_device}")
        print(f"  - Mixed precision: {(config.get('infrastructure', {}) or {}).get('mixed_precision', False)}")
        print("=" * 60)

    # Set seed for reproducibility
    seed = (config.get('infrastructure', {}) or {}).get('seed', 42)
    torch.manual_seed(seed)  # Same seed for all ranks for model init
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    training_config = config.get('training', {}) or {}
    data_config = config.get('data', {}) or {}

    # Create data loader
    if is_main_process:
        print("Creating data loader...")
    dataloader = create_dataloader_from_config(config, split='train')
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataloader.dataset,
            batch_size=int(training_config.get('batch_size', 64)),
            sampler=sampler,
            num_workers=int(data_config.get('num_workers', 4)),
            pin_memory=bool(data_config.get('pin_memory', True)),
            drop_last=True,
        )

    if is_main_process:
        print(f"Dataset size: {len(dataloader.dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    if is_main_process:
        print("Creating model...")
    base_model = create_model_from_config(config).to(device)

    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

    model = base_model
    if is_distributed:
        # Ensure model is on correct device before DDP wrapping
        assert device.type == 'cuda', f"Rank {rank}: Distributed training requires CUDA, but got device.type={device.type}"
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    num_params = sum(p.numel() for p in base_model.parameters())
    if is_main_process:
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Create method
    if is_main_process:
        print(f"Creating {method_name}...")
    if method_name == 'ddpm':
        if hasattr(DDPM, "from_config"):
            method = DDPM.from_config(model, config, device)
        else:
            ddpm_cfg = config.get("ddpm", {}) or {}
            method = DDPM(
                model=model,
                device=device,
                num_timesteps=int(ddpm_cfg.get("num_timesteps", 1000)),
                beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
                beta_end=float(ddpm_cfg.get("beta_end", 2e-2)),
            )
        # Ensure both model + buffers live on the right device
        if hasattr(method, "to"):
            method = method.to(device)
    else:
        raise ValueError(f"Unknown method: {method_name}. Only 'ddpm' is currently supported.")
    
        # Create optimizer
        optimizer = create_optimizer(model, config) # default to AdamW optimizer
    
        # Create EMA
        ema_decay = float(training_config.get('ema_decay', 0.9999))
        ema = EMA(unwrap_model(model), decay=ema_decay)
    
        # Create gradient scaler for mixed precision
        # Determine device type for GradScaler (cuda or cpu)
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        scaler = GradScaler(device_type, enabled=(config.get('infrastructure', {}) or {}).get('mixed_precision', False))
    
        # Setup logging
        log_dir = None
        wandb_run = None
        if is_main_process:
            log_dir, wandb_run = setup_logging(config, method_name)
    
        # Log model info to wandb
        if is_main_process and wandb_run is not None:
            try:
                wandb.log({
                    'model/parameters': num_params,
                    'model/parameters_millions': num_params / 1e6,
                    'data/dataset_size': len(dataloader.dataset),
                    'data/batches_per_epoch': len(dataloader),
                }, step=0)
                # Watch model gradients and parameters
                # wandb.watch(model, log='all', log_freq=config['training']['log_every'])
            except Exception as e:
                print(f"Warning: Failed to log model info to wandb: {e}")
    
    
        # Resume from checkpoint if specified
        start_step = 0
        if resume_path is not None:
            # Barrier to ensure all processes wait before loading checkpoint
            if is_distributed:
                dist.barrier()
            start_step = load_checkpoint(resume_path, model, optimizer, ema, scaler, device)
        
        # Training config (robust defaults)
    num_iterations = int(training_config.get('num_iterations', training_config.get('max_steps', 2000)))
    log_every = int(training_config.get('log_every', 50))
    sample_every = training_config.get('sample_every', training_config.get('eval_every', 200))
    save_every = training_config.get('save_every', training_config.get('checkpoint_every', 1000))
    
    # Allow disabling by setting <=0 / None
    sample_every = None if sample_every is None or int(sample_every) <= 0 else int(sample_every)
    save_every = None if save_every is None or int(save_every) <= 0 else int(save_every)
    
    num_samples = int(training_config.get('num_samples', (config.get('sampling', {}) or {}).get('num_samples', 64)))
    gradient_clip_norm = float(training_config.get('gradient_clip_norm', 1.0))
    
    # Image shape for sampling
    channels = int(data_config.get('channels', 3))
    image_size = int(data_config.get('image_size', 64))
    image_shape = (channels, image_size, image_size)
    
        
        # Training loop
        if is_main_process:
            print(f"\nStarting training from step {start_step}...")
            print(f"Total iterations: {num_iterations}")
            if overfit_single_batch:
                print("DEBUG MODE: Overfitting to a single batch")
            print("-" * 50)
    
        method.train_mode()
        data_iter = iter(dataloader)
        epoch = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
    
        # Pro tips: before big training runs, it's usually a good idea to sanity check 
        # by overfitting to a single batch with a small number of training iterations
        # For single batch overfitting, grab one batch and reuse it
        single_batch = None
        single_batch_base = None  # Store the original small batch
        if overfit_single_batch:
            single_batch_base = next(data_iter)
            if isinstance(single_batch_base, (tuple, list)):
                single_batch_base = single_batch_base[0]  # Handle (image, label) tuples
            single_batch_base = single_batch_base.to(device)
    
            # Replicate to match desired batch size
            base_batch_size = single_batch_base.shape[0]
            desired_batch_size = int(training_config.get('batch_size', single_batch_base.shape[0]))
    
            if desired_batch_size > base_batch_size:
                # Replicate the batch to reach desired size
                num_repeats = (desired_batch_size + base_batch_size - 1) // base_batch_size
                single_batch = single_batch_base.repeat(num_repeats, 1, 1, 1)[:desired_batch_size]
                if is_main_process:
                    print(f"Cached single batch: {base_batch_size} samples replicated to {desired_batch_size}")
                    print(f"  Base batch shape: {single_batch_base.shape}")
                    print(f"  Training batch shape: {single_batch.shape}")
            else:
                single_batch = single_batch_base
                if is_main_process:
                    print(f"Cached single batch with shape: {single_batch.shape}")
    
        metrics_sum = {}
        metrics_count = 0
        start_time = time.time()
    
        pbar = tqdm(
            range(start_step, num_iterations),
            initial=start_step,
            total=num_iterations,
            disable=not is_main_process,
        )
        for step in pbar:
            # Get batch (cycle through dataset or use single batch)
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
                    batch = batch[0]  # Handle (image, label) tuples
    
                batch = batch.to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast(device_type, enabled=(config.get('infrastructure', {}) or {}).get('mixed_precision', False)):
                loss, metrics = method.compute_loss(batch)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
    
            # EMA update - DISABLED
            ema.update()
            
            # Accumulate metrics (store raw values, will reduce when logging)
            for k, v in metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = []
                metrics_sum[k].append(v.detach().item() if torch.is_tensor(v) else float(v))
            metrics_count += 1
            
            # Logging
            if log_every and (step + 1) % log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = metrics_count / elapsed
    
                # Average metrics locally first
                local_avg_metrics = {k: sum(v) / len(v) for k, v in metrics_sum.items()}
    
                # Reduce metrics across all ranks
                avg_metrics = reduce_metrics(local_avg_metrics, device, world_size)
    
                # Update progress bar
                if is_main_process:
                    pbar.set_postfix({
                        'loss': f"{avg_metrics['loss']:.4f}",
                        'steps/s': f"{steps_per_sec:.2f}",
                    })
    
                # Log to wandb
                if is_main_process and wandb_run is not None:
                    log_dict = {
                        'train/step': step + 1,
                        'train/steps_per_sec': steps_per_sec,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                    }
                    # Add all metrics
                    for k, v in avg_metrics.items():
                        log_dict[f'train/{k}'] = v
    
                    try:
                        wandb.log(log_dict, step=step + 1)
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
    
    
                # Reset metrics
                metrics_sum = {}
                metrics_count = 0
                start_time = time.time()
    
            # Generate samples
            if sample_every and (step + 1) % sample_every == 0:
                if is_main_process:
                    print(f"\nGenerating samples at step {step + 1}...")
                    samples = generate_samples(
                        method,
                        num_samples,
                        image_shape,
                        device,
                        method_name,
                        config,
                        ema,
                        current_step=step + 1,
                    )
                    sample_path = os.path.join(log_dir, 'samples', f'samples_{step + 1:07d}.png')
                    save_samples(samples, sample_path, num_samples)
    
                    # Log samples to wandb
                    if wandb_run is not None:
                        try:
                            # Load the saved image and log it
                            img = PILImage.open(sample_path)
                            wandb.log({
                                'samples': wandb.Image(img, caption=f'Step {step + 1}')
                            }, step=step + 1)
                        except Exception as e:
                            print(f"Warning: Failed to log samples to wandb: {e}")
    
                # Barrier to synchronize all processes after sampling
                if is_distributed:
                    dist.barrier()
    
            # Save checkpoint
            if save_every and (step + 1) % save_every == 0:
                if is_main_process:
                    checkpoint_path = os.path.join(log_dir, 'checkpoints', f'{method_name}_{step + 1:07d}.pt')
                    save_checkpoint(checkpoint_path, model, optimizer, ema, scaler, step + 1, config)
    
                # Barrier to synchronize all processes after checkpoint save
                if is_distributed:
                    dist.barrier()
        
        # Save final checkpoint
        if is_main_process:
            final_path = os.path.join(log_dir, 'checkpoints', f'{method_name}_final.pt')
            save_checkpoint(final_path, model, optimizer, ema, scaler, num_iterations, config)
    
            print("\nTraining complete!")
            print(f"Final checkpoint: {final_path}")
            print(f"Samples saved to: {os.path.join(log_dir, 'samples')}")
    
        # Finish wandb run
        if is_main_process and wandb_run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
    
        # Final barrier before cleanup
        if is_distributed:
            dist.barrier()
            cleanup_distributed(is_distributed)



def main():
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm'], # You can add more later
                       help='Method to train (currently only ddpm is supported)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/ddpm.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--overfit-single-batch', action='store_true',
                       help='DEBUG MODE: Train on a single batch repeatedly to verify model can overfit')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with resume path if specified
    if args.resume:
        config.setdefault('checkpoint', {})
        config['checkpoint']['resume'] = args.resume

    # Train
    train(
        method_name=args.method,
        config=config,
        resume_path=args.resume,
        overfit_single_batch=args.overfit_single_batch,
    )


if __name__ == '__main__':
    main()
