"""Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

    # Sample from Flow Matching (Euler steps)
    python sample.py --checkpoint checkpoints/flow_matching_final.pt --method flow_matching --num_steps 200 --grid
"""

import os
import argparse
from datetime import datetime

import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, unnormalize
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def _get_image_shape_from_config(config: dict) -> tuple[int, int, int]:
    """Robustly infer (C,H,W) from potentially different config schemas."""
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}

    C = (
        data_cfg.get("channels")
        or data_cfg.get("num_channels")
        or data_cfg.get("in_channels")
        or model_cfg.get("in_channels")
        or model_cfg.get("channels")
        or 3
    )
    H = (
        data_cfg.get("image_size")
        or data_cfg.get("resolution")
        or model_cfg.get("image_size")
        or model_cfg.get("resolution")
        or 64
    )
    C, H = int(C), int(H)
    return (C, H, H)


def _get_default_num_steps(config: dict) -> int:
    sampling_cfg = config.get("sampling", {}) or {}
    ddpm_cfg = config.get("ddpm", {}) or {}
    flow_matching_cfg = config.get("flow_matching", {}) or {}

    steps = sampling_cfg.get("num_steps") or sampling_cfg.get("steps")
    if steps is None:
        # Prefer ddpm timesteps if present; else flow_matching timesteps; else 1000
        steps = (
            ddpm_cfg.get("num_timesteps")
            or ddpm_cfg.get("timesteps")
            or flow_matching_cfg.get("num_timesteps")
            or flow_matching_cfg.get("timesteps")
            or 1000
        )
    return int(steps)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA (if present)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'config' field. "
            "Re-save your checkpoint using the provided training script (it stores config)."
        )
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    ema = None
    if "ema" in checkpoint:
        decay = (config.get("training", {}) or {}).get("ema_decay", 0.9999)
        ema = EMA(model, decay=decay)
        ema.load_state_dict(checkpoint["ema"])

    return model, config, ema


def save_samples(samples: torch.Tensor, save_path: str, nrow: int = 8) -> None:
    """Save a batch of samples as an image (grid if batch>1)."""
    samples = samples.detach().cpu()
    samples = unnormalize(samples).clamp(0.0, 1.0)  # [-1,1] -> [0,1]
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ddpm", "flow_matching"],
        help="Method used for training",
    )
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save individual images")
    parser.add_argument("--grid", action="store_true", help="Save as one grid image instead of individual images")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for grid (only used with --grid, default: samples_<timestamp>.png)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of reverse sampling steps")
    parser.add_argument("--eta", type=float, default=None, help="DDIM stochasticity (ddpm only; eta=0 is deterministic)")
    parser.add_argument("--no_ema", action="store_true", help="Use training weights instead of EMA weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)

    if args.method == "ddpm":
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
        if hasattr(method, "to"):
            method = method.to(device)

    elif args.method == "flow_matching":
        if hasattr(FlowMatching, "from_config"):
            method = FlowMatching.from_config(model, config, device)
        else:
            flow_matching_cfg = config.get("flow_matching", {}) or {}
            method = FlowMatching(
                model=model,
                device=device,
                num_timesteps=int(flow_matching_cfg.get("num_timesteps", 1000)),
            )
        if hasattr(method, "to"):
            method = method.to(device)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    if not args.no_ema and ema is not None:
        print("Using EMA weights")
        ema.apply_shadow()
    elif args.no_ema:
        print("Using training weights (no EMA)")
    else:
        print("No EMA found in checkpoint; using training weights")

    method.eval_mode()

    image_shape = _get_image_shape_from_config(config)
    default_steps = _get_default_num_steps(config)
    print(f"Image shape inferred as: {image_shape}")
    print(f"Default num_steps from config: {default_steps}")

    num_steps = int(args.num_steps) if args.num_steps is not None else default_steps
    eta = args.eta if args.eta is not None else (config.get("sampling", {}) or {}).get("eta", 0.0)
    print(f"Generating {args.num_samples} samples with num_steps={num_steps}, eta={eta}...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            bs = min(args.batch_size, remaining)

            try:
                samples = method.sample(batch_size=bs, image_shape=image_shape, num_steps=num_steps, eta=eta)
            except TypeError:
                samples = method.sample(batch_size=bs, image_shape=image_shape, num_steps=num_steps)

            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples[i:i+1], img_path, nrow=1)
                    sample_idx += 1

            remaining -= bs
            pbar.update(bs)
        pbar.close()

    if args.grid:
        all_samples = torch.cat(all_samples, dim=0)[: args.num_samples]
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"
        nrow = int((args.num_samples ** 0.5) + 0.999)
        save_samples(all_samples, args.output, nrow=nrow)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    if not args.no_ema and ema is not None:
        ema.restore()


if __name__ == "__main__":
    main()
