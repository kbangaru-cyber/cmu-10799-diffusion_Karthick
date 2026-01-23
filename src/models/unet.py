"""
U-Net Architecture for Diffusion Models

Implements a U-Net suitable for DDPM-style diffusion models.
The model predicts noise epsilon given x_t and timestep t.

Key ideas:
- Sinusoidal time embedding -> MLP (TimestepEmbedding)
- Residual blocks (ResBlock) with FiLM conditioning
- Optional self-attention (AttentionBlock) at chosen resolutions
- Strided conv downsample and nearest-neighbor upsample
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


class TimestepSequential(nn.Sequential):
    """A sequential container that passes time embeddings to ResBlocks."""

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    U-Net backbone for diffusion.

    Args:
        in_channels: Input image channels (e.g., 3 for RGB)
        out_channels: Output channels (usually match in_channels; predicts noise)
        base_channels: Base channel width
        channel_mult: Multipliers per resolution level
        num_res_blocks: ResBlocks per level
        attention_resolutions: Spatial resolutions (e.g. [16, 8]) where attention is applied
        num_heads: Attention heads
        dropout: Dropout prob inside ResBlocks
        use_scale_shift_norm: Use FiLM conditioning in ResBlocks
        image_size: Input spatial size (used to decide where attention goes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        image_size: int = 64,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = list(attention_resolutions)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        self.image_size = int(image_size)

        # Time embedding
        time_embed_dim = 4 * self.base_channels
        self.time_embed = TimestepEmbedding(time_embed_dim)

        # Input blocks (down path). We follow the "guided-diffusion" style:
        # number of output blocks equals number of input blocks.
        self.input_blocks = nn.ModuleList()
        self._input_block_chans: List[int] = []

        ch = self.base_channels
        self.input_blocks.append(TimestepSequential(nn.Conv2d(self.in_channels, ch, kernel_size=3, padding=1)))
        self._input_block_chans.append(ch)

        # Track current resolution
        res = self.image_size

        for level, mult in enumerate(self.channel_mult):
            out_ch = self.base_channels * mult
            for _ in range(self.num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_embed_dim, dropout=self.dropout, use_scale_shift_norm=self.use_scale_shift_norm)]
                ch = out_ch
                if res in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                self.input_blocks.append(TimestepSequential(*layers))
                self._input_block_chans.append(ch)

            # Downsample between levels (except last)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(TimestepSequential(Downsample(ch)))
                self._input_block_chans.append(ch)
                res //= 2

        # Middle block
        middle_layers = [
            ResBlock(ch, ch, time_embed_dim, dropout=self.dropout, use_scale_shift_norm=self.use_scale_shift_norm),
            AttentionBlock(ch, num_heads=self.num_heads),
            ResBlock(ch, ch, time_embed_dim, dropout=self.dropout, use_scale_shift_norm=self.use_scale_shift_norm),
        ]
        self.middle_block = TimestepSequential(*middle_layers)

        # Output blocks (up path)
        self.output_blocks = nn.ModuleList()

        # Start from the bottom resolution
        # res is currently image_size // 2**(num_levels-1)
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            out_ch = self.base_channels * mult
            for i in range(self.num_res_blocks + 1):
                skip_ch = self._input_block_chans.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_embed_dim, dropout=self.dropout, use_scale_shift_norm=self.use_scale_shift_norm)]
                ch = out_ch
                if res in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                # Upsample at end of each level except the top
                if level != 0 and i == self.num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.output_blocks.append(TimestepSequential(*layers))

        assert len(self._input_block_chans) == 0, "Internal error: skip-channel stack not fully consumed."

        # Output head
        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy image x_t
            t: (B,) timesteps (int64 preferred, but float also works)

        Returns:
            (B, out_channels, H, W) predicted noise epsilon
        """
        assert x.ndim == 4, f"Expected x to be 4D (B,C,H,W), got {x.shape}"
        # Build time embedding
        time_emb = self.time_embed(t)

        hs: List[torch.Tensor] = []
        h = x

        # Down path
        for block in self.input_blocks:
            h = block(h, time_emb)
            hs.append(h)

        # Middle
        h = self.middle_block(h, time_emb)

        # Up path
        for block in self.output_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)

        assert len(hs) == 0, "Internal error: not all skip activations were consumed."

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


def create_model_from_config(config: dict) -> UNet:
    """Factory function to create a UNet from a configuration dictionary."""
    model_config = config["model"]
    data_config = config["data"]

    return UNet(
        in_channels=data_config["channels"],
        out_channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=model_config["attention_resolutions"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"],
        image_size=data_config.get("image_size", 64),
    )


if __name__ == "__main__":
    # Minimal self-test
    print("Testing UNet...")

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.0,
        image_size=64,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4,))
    with torch.no_grad():
        y = model(x, t)

    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print("✓ Forward pass successful!")
