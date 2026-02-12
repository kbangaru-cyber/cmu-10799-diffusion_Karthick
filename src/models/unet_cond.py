"""
Conditional U-Net for Classifier-Free Guidance (CFG)

Extends the base UNet with attribute conditioning:
  - Embeds a binary attribute vector c ∈ {0,1}^K via a small MLP
  - Fuses attribute embedding with the timestep embedding (addition)
  - Supports a "null" condition (all-zeros) for the unconditional path

The rest of the architecture is identical to the HW2 UNet.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class CondUNet(nn.Module):
    """
    U-Net with attribute conditioning for classifier-free guidance.

    New parameter vs base UNet:
        num_attributes (int): Number of binary attributes (e.g., 3 for
            Smiling, Eyeglasses, Male). The attribute vector is embedded
            into the same hidden dimension as the timestep embedding and
            added to it, so every ResBlock sees the fused signal.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        image_size: int = 64,
        # --- NEW for conditioning ---
        num_attributes: int = 3,
    ):
        super().__init__()

        self.image_size = int(image_size)
        self.attention_resolutions = list(attention_resolutions)
        self.num_attributes = num_attributes

        # Time embedding: sinusoidal(base_channels) -> MLP -> (base_channels*4)
        time_hidden = base_channels * 4
        self.time_embed = TimestepEmbedding(
            time_embed_dim=base_channels, hidden_dim=time_hidden
        )

        # ---- Attribute conditioning embedding ----
        # Maps binary vector c ∈ R^K -> R^time_hidden via a small MLP
        # This is fused with the timestep embedding by addition.
        self.attr_embed = nn.Sequential(
            nn.Linear(num_attributes, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, time_hidden),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- Down path ----
        self.down_blocks = nn.ModuleList()
        self.down_save = []
        ch = base_channels
        ds = 1
        res = self.image_size
        skip_channels = [ch]

        for level, mult in enumerate(tuple(channel_mult)):
            out_ch = base_channels * int(mult)
            for _ in range(int(num_res_blocks)):
                self.down_blocks.append(
                    ResBlock(
                        ch, out_ch, time_hidden,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                self.down_save.append(True)
                ch = out_ch
                skip_channels.append(ch)

                if res in self.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch, num_heads=num_heads))
                    self.down_save.append(False)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                self.down_save.append(True)
                ds *= 2
                res = self.image_size // ds
                skip_channels.append(ch)

        # ---- Middle ----
        self.mid1 = ResBlock(
            ch, ch, time_hidden,
            dropout=dropout, use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid2 = ResBlock(
            ch, ch, time_hidden,
            dropout=dropout, use_scale_shift_norm=use_scale_shift_norm,
        )

        # ---- Up path ----
        self.up_blocks = nn.ModuleList()
        self.up_concat = []
        for level, mult in reversed(list(enumerate(tuple(channel_mult)))):
            out_ch = base_channels * int(mult)
            n_blocks = int(num_res_blocks) + 1
            for _ in range(n_blocks):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock(
                        ch + skip_ch, out_ch, time_hidden,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                self.up_concat.append(True)
                ch = out_ch

                if res in self.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch, num_heads=num_heads))
                    self.up_concat.append(False)

            if level != 0:
                self.up_blocks.append(Upsample(ch))
                self.up_concat.append(False)
                ds //= 2
                res = self.image_size // ds

        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) integer timestep indices or scalar
            c: (B, K) binary attribute vector, or None for unconditional
               Pass all-zeros for the null condition (unconditional mode).
        """
        if t.dim() == 0:
            t = t[None]
        if t.dim() > 1:
            t = t.view(-1)

        # Timestep embedding
        temb = self.time_embed(t)

        # Attribute conditioning — fuse by addition
        if c is not None:
            c = c.float()
            attr_emb = self.attr_embed(c)
            temb = temb + attr_emb
        # If c is None, we treat it as unconditional (no attribute signal added)

        h = self.in_conv(x)

        skips = [h]
        for block, save_skip in zip(self.down_blocks, self.down_save):
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)
            if save_skip:
                skips.append(h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        for block, do_concat in zip(self.up_blocks, self.up_concat):
            if do_concat:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, temb)
            else:
                h = block(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


def create_cond_model_from_config(config: dict) -> CondUNet:
    """Create a CondUNet from a config dictionary."""
    model_config = config["model"]
    image_size = 64
    if "data" in config and isinstance(config["data"], dict):
        image_size = int(config["data"].get("image_size", image_size))

    # Get number of attributes from config
    cond_cfg = config.get("conditioning", {}) or {}
    num_attributes = int(cond_cfg.get("num_attributes", 3))

    return CondUNet(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 3)),
        base_channels=int(model_config["base_channels"]),
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=int(model_config["num_res_blocks"]),
        attention_resolutions=list(model_config.get("attention_resolutions", [])),
        num_heads=int(model_config.get("num_heads", 4)),
        dropout=float(model_config.get("dropout", 0.0)),
        use_scale_shift_norm=bool(model_config.get("use_scale_shift_norm", True)),
        image_size=image_size,
        num_attributes=num_attributes,
    )
