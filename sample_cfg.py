"""
Sampling & Editing Script for CFG Flow Matching

Supports:
  1. Conditional generation: Generate faces with specific attributes
  2. Image editing: Edit real images by changing attributes
  3. Guidance sweep: Compare different guidance scales

Usage:
  # Conditional generation with specific attributes
  python sample_cfg.py --checkpoint ckpt.pt --mode generate \
      --attributes "Smiling=1,Eyeglasses=0,Male=1" --guidance_scale 2.0 --grid

  # Generate comparison grid across guidance scales
  python sample_cfg.py --checkpoint ckpt.pt --mode sweep \
      --attributes "Smiling=1" --guidance_scales "0,1,2,3,4"

  # Edit real images
  python sample_cfg.py --checkpoint ckpt.pt --mode edit \
      --input_dir ./data/celeba-subset/train/images \
      --attributes "Smiling=1" --t_edit 0.5 --guidance_scale 2.0

  # Generate attribute comparison grid (each row = different attribute)
  python sample_cfg.py --checkpoint ckpt.pt --mode attr_grid
"""

import os
import argparse
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from src.models.unet_cond import create_cond_model_from_config
from src.methods.cfg_flow_matching import CFGFlowMatching
from src.utils import EMA

try:
    from src.data import save_image, unnormalize
except ImportError:
    from torchvision.utils import save_image as _tv_save_image

    def unnormalize(x):
        return x * 0.5 + 0.5

    def save_image(tensor, path, nrow=8):
        _tv_save_image(tensor, path, nrow=nrow)


def _get_image_shape_from_config(config):
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    C = int(data_cfg.get("channels", model_cfg.get("in_channels", 3)))
    H = int(data_cfg.get("image_size", model_cfg.get("image_size", 64)))
    return (C, H, H)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_cond_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    ema = None
    if "ema" in checkpoint:
        decay = (config.get("training", {}) or {}).get("ema_decay", 0.9999)
        ema = EMA(model, decay=decay)
        ema.load_state_dict(checkpoint["ema"])

    return model, config, ema


def parse_attributes(attr_string, attr_names):
    """
    Parse attribute string like 'Smiling=1,Eyeglasses=0,Male=1'
    into a tensor of shape (K,).
    """
    K = len(attr_names)
    vec = torch.zeros(K)

    if not attr_string:
        return vec

    for part in attr_string.split(","):
        part = part.strip()
        if "=" in part:
            name, val = part.split("=", 1)
            name = name.strip()
            val = float(val.strip())
        else:
            name = part
            val = 1.0

        # Find matching attribute
        found = False
        for i, an in enumerate(attr_names):
            if an.lower() == name.lower():
                vec[i] = val
                found = True
                break
        if not found:
            print(f"WARNING: Attribute '{name}' not found in {attr_names}")

    return vec


def save_grid(samples, path, nrow=8):
    samples = samples.detach().cpu()
    samples = unnormalize(samples).clamp(0.0, 1.0)
    save_image(samples, path, nrow=nrow)
    print(f"Saved to {path}")


def mode_generate(args, method, config, device):
    """Generate conditional samples with a specific attribute vector."""
    image_shape = _get_image_shape_from_config(config)
    cond_cfg = config.get("conditioning", {}) or {}
    attr_names = cond_cfg.get("attributes", ["Smiling", "Eyeglasses", "Male"])
    num_attrs = len(attr_names)

    c_vec = parse_attributes(args.attributes, attr_names)
    print(f"Generating with attributes: {dict(zip(attr_names, c_vec.tolist()))}")
    print(f"Guidance scale: {args.guidance_scale}")

    c = c_vec.unsqueeze(0).expand(args.num_samples, -1).to(device)

    all_samples = []
    remaining = args.num_samples
    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating")
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            samples = method.sample(
                batch_size=bs,
                image_shape=image_shape,
                num_steps=args.num_steps,
                c=c[:bs],
                guidance_scale=args.guidance_scale,
            )
            all_samples.append(samples)
            remaining -= bs
            pbar.update(bs)
        pbar.close()

    all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

    if args.grid:
        nrow = int(args.num_samples ** 0.5 + 0.999)
        out_path = args.output or f"cfg_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_grid(all_samples, out_path, nrow=nrow)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        for i in range(all_samples.shape[0]):
            img_path = os.path.join(args.output_dir, f"{i:06d}.png")
            save_grid(all_samples[i:i+1], img_path, nrow=1)
        print(f"Saved {args.num_samples} images to {args.output_dir}")


def mode_sweep(args, method, config, device):
    """Generate comparison across different guidance scales."""
    image_shape = _get_image_shape_from_config(config)
    cond_cfg = config.get("conditioning", {}) or {}
    attr_names = cond_cfg.get("attributes", ["Smiling", "Eyeglasses", "Male"])

    c_vec = parse_attributes(args.attributes, attr_names)
    print(f"Attributes: {dict(zip(attr_names, c_vec.tolist()))}")

    scales = [float(s.strip()) for s in args.guidance_scales.split(",")]
    print(f"Guidance scales: {scales}")

    samples_per_scale = min(8, args.num_samples)

    # Fix the noise seed so all scales start from the same noise
    torch.manual_seed(args.seed or 42)
    fixed_noise = torch.randn(samples_per_scale, *image_shape, device=device)

    all_rows = []
    for w in scales:
        print(f"  Generating with w={w}...")
        c = c_vec.unsqueeze(0).expand(samples_per_scale, -1).to(device)

        # Reset noise
        x = fixed_noise.clone()
        S = args.num_steps
        dt = 1.0 / float(S)
        c_null = torch.zeros_like(c)

        with torch.no_grad():
            for k in range(S):
                t_cont = torch.full((samples_per_scale,), k / float(S), device=device)
                t_idx = method._t_to_index(t_cont)
                if w > 0:
                    v_uncond = method.predict_v(x, t_idx, c_null)
                    v_cond = method.predict_v(x, t_idx, c)
                    v = (1.0 + w) * v_cond - w * v_uncond
                else:
                    v = method.predict_v(x, t_idx, c)
                x = x + dt * v

        all_rows.append(x.clamp(-1.0, 1.0))

    # Stack: each row is one guidance scale
    grid = torch.cat(all_rows, dim=0)
    out_path = args.output or f"cfg_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_grid(grid, out_path, nrow=samples_per_scale)


def mode_attr_grid(args, method, config, device):
    """Generate grid: each row has different attribute combination."""
    image_shape = _get_image_shape_from_config(config)
    cond_cfg = config.get("conditioning", {}) or {}
    attr_names = cond_cfg.get("attributes", ["Smiling", "Eyeglasses", "Male"])
    num_attrs = len(attr_names)

    samples_per_row = 8
    w = args.guidance_scale

    # Row 0: unconditional
    # Row 1..K: each single attribute ON
    # Row K+1: all attributes ON
    conditions = []
    labels = []

    # Unconditional
    conditions.append(torch.zeros(samples_per_row, num_attrs, device=device))
    labels.append("None (uncond)")

    # Single attributes
    for i, name in enumerate(attr_names):
        c = torch.zeros(samples_per_row, num_attrs, device=device)
        c[:, i] = 1.0
        conditions.append(c)
        labels.append(name)

    # All on
    c_all = torch.ones(samples_per_row, num_attrs, device=device)
    conditions.append(c_all)
    labels.append("All")

    all_rows = []
    with torch.no_grad():
        for c, label in zip(conditions, labels):
            print(f"  Generating: {label}...")
            samples = method.sample(
                batch_size=samples_per_row,
                image_shape=image_shape,
                num_steps=args.num_steps,
                c=c,
                guidance_scale=w,
            )
            all_rows.append(samples)

    grid = torch.cat(all_rows, dim=0)
    out_path = args.output or f"attr_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_grid(grid, out_path, nrow=samples_per_row)
    print(f"Rows: {labels}")


def mode_edit(args, method, config, device):
    """Edit real images by noise-and-denoise with target attributes."""
    image_shape = _get_image_shape_from_config(config)
    cond_cfg = config.get("conditioning", {}) or {}
    attr_names = cond_cfg.get("attributes", ["Smiling", "Eyeglasses", "Male"])

    c_vec = parse_attributes(args.attributes, attr_names)
    print(f"Edit target: {dict(zip(attr_names, c_vec.tolist()))}")
    print(f"Edit strength (t_edit): {args.t_edit}")
    print(f"Guidance scale: {args.guidance_scale}")

    # Load input images
    transform = transforms.Compose([
        transforms.Resize(image_shape[1]),
        transforms.CenterCrop(image_shape[1]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    input_dir = args.input_dir
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:args.num_samples]

    images = []
    for fname in image_files:
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        images.append(transform(img))

    if not images:
        print(f"No images found in {input_dir}")
        return

    x_orig = torch.stack(images).to(device)
    B = x_orig.shape[0]
    c_target = c_vec.unsqueeze(0).expand(B, -1).to(device)

    print(f"Editing {B} images...")

    with torch.no_grad():
        x_edited = method.edit(
            x_orig=x_orig,
            c_target=c_target,
            t_edit=args.t_edit,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

    # Create before/after comparison
    # Interleave: orig1, edit1, orig2, edit2, ...
    comparison = torch.stack([x_orig, x_edited], dim=1).view(-1, *image_shape)

    out_path = args.output or f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_grid(comparison, out_path, nrow=min(B, 8) * 2)  # pairs side by side

    # Also save just edits
    edit_path = out_path.replace(".png", "_edits_only.png")
    save_grid(x_edited, edit_path, nrow=min(B, 8))


def main():
    parser = argparse.ArgumentParser(description="CFG Sampling & Editing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="generate",
                        choices=["generate", "sweep", "attr_grid", "edit"])
    parser.add_argument("--attributes", type=str, default="Smiling=1",
                        help="Comma-separated attr=val pairs")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--guidance_scales", type=str, default="0,1,2,3,4",
                        help="Comma-separated scales for sweep mode")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="cfg_samples")
    parser.add_argument("--input_dir", type=str, default="./data/celeba-subset/train/images",
                        help="Input images directory (for edit mode)")
    parser.add_argument("--t_edit", type=float, default=0.5,
                        help="Edit strength: 0=max edit, 1=no edit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)

    method = CFGFlowMatching.from_config(model, config, device)

    if not args.no_ema and ema is not None:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights")

    method.eval_mode()

    # Dispatch
    if args.mode == "generate":
        mode_generate(args, method, config, device)
    elif args.mode == "sweep":
        mode_sweep(args, method, config, device)
    elif args.mode == "attr_grid":
        mode_attr_grid(args, method, config, device)
    elif args.mode == "edit":
        mode_edit(args, method, config, device)

    # Restore EMA
    if not args.no_ema and ema is not None:
        ema.restore()


if __name__ == "__main__":
    main()
