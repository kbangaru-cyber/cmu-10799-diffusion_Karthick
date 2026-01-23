"""
U-Net Architecture for Diffusion Models (DDPM)

Uses building blocks from src/models/blocks.py:
- TimestepEmbedding
- ResBlock
- AttentionBlock
- Downsample / Upsample
"""

from __future__ import annotations

from typing import List, Tuple

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


class UNet(nn.Module):
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
    ):
        super().__init__()

        self.image_size = int(image_size)
        self.attention_resolutions = list(attention_resolutions)

        # Time embedding: sinusoidal(base_channels) -> MLP -> (base_channels*4)
        time_hidden = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim=base_channels, hidden_dim=time_hidden)

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down
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
                    ResBlock(ch, out_ch, time_hidden, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm)
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

        # Middle
        self.mid1 = ResBlock(ch, ch, time_hidden, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm)
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid2 = ResBlock(ch, ch, time_hidden, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm)

        # Up
        self.up_blocks = nn.ModuleList()
        self.up_concat = []
        for level, mult in reversed(list(enumerate(tuple(channel_mult)))):
            out_ch = base_channels * int(mult)
            n_blocks = int(num_res_blocks) + 1
            for _ in range(n_blocks):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_hidden, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.dim() > 1:
            t = t.view(-1)

        temb = self.time_embed(t)

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


def create_model_from_config(config: dict) -> UNet:
    model_config = config["model"]
    image_size = 64
    if "data" in config and isinstance(config["data"], dict):
        image_size = int(config["data"].get("image_size", image_size))

    return UNet(
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
    )
