
"""
Sampling Script for DDPM (x0-parametrization)

Usage:
  PYTHONPATH=. python sample_x0.py --checkpoint <ckpt.pt> --config configs/ddpm_modal_x0.yaml --num_samples 16 --grid --output x0_grid.png
  PYTHONPATH=. python sample_x0.py --checkpoint <ckpt.pt> --config configs/ddpm_modal_x0.yaml --num_samples 1000 --output_dir kid_fake_x0
"""
import os
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.data import save_image, unnormalize
from src.utils import EMA
from src.methods.ddpm_x0 import DDPM_X0


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True, help="YAML config used to build the model/method")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--output", type=str, default=None, help="grid output path if --grid")
    p.add_argument("--output_dir", type=str, default="samples_x0", help="directory for individual images")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(args.seed)

    cfg = load_config(args.config)

    # Build model
    model = create_model_from_config(cfg).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # tolerate alternative checkpoint keys
        model.load_state_dict(ckpt)

    # EMA (optional)
    ema = None
    if "training" in ckpt and "ema" in ckpt:
        try:
            ema = EMA(model, decay=float(cfg.get("training", {}).get("ema_decay", 0.9999)))
            ema.load_state_dict(ckpt["ema"])
        except Exception:
            ema = None

    ddpm_cfg = cfg.get("ddpm", {}) or {}
    method = DDPM_X0(
        model=model,
        device=device,
        num_timesteps=int(ddpm_cfg.get("num_timesteps", 1000)),
        beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
        beta_end=float(ddpm_cfg.get("beta_end", 2e-2)),
    ).to(device)

    if (not args.no_ema) and (ema is not None):
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA or EMA unavailable)")

    method.eval_mode()

    C = int(cfg.get("data", {}).get("channels", cfg.get("model", {}).get("in_channels", 3)))
    H = int(cfg.get("data", {}).get("image_size", 64))
    W = H
    image_shape = (C, H, W)

    # Generate
    all_samples = []
    remaining = args.num_samples
    pbar = tqdm(total=args.num_samples, desc="Generating")
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        x = method.sample(bs, image_shape, num_steps=args.num_steps, eta=args.eta)
        all_samples.append(x.detach().cpu())
        remaining -= bs
        pbar.update(bs)
    pbar.close()

    samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    samples = unnormalize(samples).clamp(0, 1)

    if args.grid:
        if args.output is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"x0_samples_{ts}.png"
        save_image(samples, args.output, nrow=int(max(1, round(args.num_samples ** 0.5))))
        print(f"Saved grid to {args.output}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(samples.shape[0]):
            save_image(samples[i:i+1], os.path.join(args.output_dir, f"{i:06d}.png"), nrow=1)
        print(f"Saved {samples.shape[0]} images to {args.output_dir}")


if __name__ == "__main__":
    main()
