import os, math
import torch
from torchvision.utils import save_image

from src.data import create_dataloader_from_config, unnormalize
from src.models import create_model_from_config
from src.methods.ddpm_x0 import DDPM_X0
import yaml

def dump_real(config, out_dir, n=1000):
    os.makedirs(out_dir, exist_ok=True)
    dl = create_dataloader_from_config(config, split="train")
    i = 0
    for batch in dl:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        # batch is typically in [-1, 1] for this homework codebase
        batch = unnormalize(batch).clamp(0, 1)
        for b in range(batch.size(0)):
            save_image(batch[b], os.path.join(out_dir, f"real_{i:05d}.png"))
            i += 1
            if i >= n:
                return

@torch.no_grad()
def dump_fake_x0(config, ckpt_path, out_dir, n=1000, bs=64, num_steps=1000):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model_from_config(config).to(device)
    ddpm_cfg = config.get("ddpm", {}) or {}
    method = DDPM_X0(
        model=model,
        device=device,
        num_timesteps=int(ddpm_cfg.get("num_timesteps", 1000)),
        beta_start=float(ddpm_cfg.get("beta_start", 1e-4)),
        beta_end=float(ddpm_cfg.get("beta_end", 2e-2)),
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    data_cfg = config.get("data", {}) or {}
    C = int(data_cfg.get("channels", 3))
    H = int(data_cfg.get("image_size", 64))
    image_shape = (C, H, H)

    i = 0
    while i < n:
        cur_bs = min(bs, n - i)
        x = method.sample(batch_size=cur_bs, image_shape=image_shape, num_steps=num_steps, eta=0.0)
        x = unnormalize(x).clamp(0, 1).cpu()
        for b in range(x.size(0)):
            save_image(x[b], os.path.join(out_dir, f"fake_{i:05d}.png"))
            i += 1

if __name__ == "__main__":
    # EDIT THESE:
    CONFIG_PATH = "configs/ddpm_modal_x0.yaml"
    CKPT_PATH   = "./logs/ddpm_x0_YYYYMMDD_HHMMSS/checkpoints/ddpm_x0_00010000.pt"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    dump_real(config, "kid_real_1k", n=1000)
    dump_fake_x0(config, CKPT_PATH, "kid_fake_x0_1k", n=1000, bs=64, num_steps=1000)
    print("Done: kid_real_1k and kid_fake_x0_1k")
