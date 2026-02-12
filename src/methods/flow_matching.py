"""
Flow Matching (straight-line / rectified flow style) — FIXED

Changes from original:
  1. REMOVED intermediate clamping during sampling (was distorting ODE trajectories)
  2. FIXED sampling to reach t=1.0 endpoint (was stopping at (S-1)/S)
  3. Removed unused beta_start/beta_end from config parsing

Trains a velocity field v_theta(x_t, t) that transports noise to data along
straight paths:

    x_t = (1 - t) * x_1 + t * x_0,   where x_1 ~ N(0, I),  t ~ Uniform[0, 1]
    v*(x_t, t) = d x_t / dt = x_0 - x_1

We discretize continuous time t in [0, 1] into integer indices in [0, T-1]
so we can reuse the exact same time-embedding interface your DDPM model uses.

Sampling: Euler integrate from t=0 to t=1:
    x <- x + dt * v_theta(x, t)
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
    ):
        super().__init__(model=model, device=device)
        self.num_timesteps = int(num_timesteps)

    @classmethod
    def from_config(cls, model, config, device):
        """
        Config-tolerant constructor.

        Looks for:
          config["flow_matching"]["num_timesteps"] (preferred)
        Falls back to:
          config["flow"]["num_timesteps"], or config["num_timesteps"], or 1000
        """
        fm_cfg = config.get("flow_matching", None)
        if fm_cfg is None:
            fm_cfg = config.get("flow", None)
        if fm_cfg is None:
            fm_cfg = config

        num_timesteps = int(fm_cfg.get("num_timesteps", fm_cfg.get("timesteps", 1000)))
        method = cls(model=model, device=device, num_timesteps=num_timesteps)
        return method.to(device)

    def _t_to_index(self, t_cont: torch.Tensor) -> torch.Tensor:
        """
        Map continuous t in [0,1] -> integer timestep index in [0, T-1].
        """
        t_cont = t_cont.clamp(0.0, 1.0)
        if self.num_timesteps <= 1:
            return torch.zeros_like(t_cont, dtype=torch.long)
        idx = torch.round(t_cont * (self.num_timesteps - 1)).long()
        return idx

    def predict_v(self, x_t: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field v_theta(x_t, t).
        Matches your DDPM convention: model(x, t_idx).
        """
        if t_idx.dtype != torch.long:
            t_idx = t_idx.long()
        return self.model(x_t, t_idx)

    def compute_loss(self, x0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Flow Matching loss:
          - sample x1 ~ N(0, I)
          - sample t ~ U[0,1]
          - x_t = (1-t)x1 + t x0
          - target velocity v* = x0 - x1
          - loss = MSE(v_pred, v*)
        """
        B = x0.shape[0]
        device = x0.device

        # Endpoint noise (source distribution)
        x1 = torch.randn_like(x0)

        # Continuous time in [0, 1]
        t_cont = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t_cont.view(B, 1, 1, 1)

        # Point on straight path
        x_t = (1.0 - t_view) * x1 + t_view * x0

        # Target velocity along the path
        v_target = x0 - x1

        # Discretize time for your existing model interface
        t_idx = self._t_to_index(t_cont).to(device)

        v_pred = self.predict_v(x_t, t_idx)

        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": float(loss.detach().item()),
            "mse": float(loss.detach().item()),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        eta: float = 0.0,  # unused, kept for interface compatibility
        **kwargs,
    ) -> torch.Tensor:
        """
        Euler integration from noise (t=0) to data (t=1).

        FIX 1: No intermediate clamping — the ODE trajectory legitimately
               passes through values outside [-1,1] at intermediate steps.
        FIX 2: Evaluate at t = (k + 0.5) / S (midpoint) so the full [0,1]
               interval is covered symmetrically, or use the standard
               grid t_k = k/S for k=0..S-1 with dt = 1/S which integrates
               exactly to t=1 after S steps.

        Returns:
            samples in [-1,1] range (final clamp only)
        """
        C, H, W = image_shape
        device = self.device

        # Start from noise
        x = torch.randn(batch_size, C, H, W, device=device)

        S = int(num_steps) if num_steps is not None else 200
        S = max(1, S)
        dt = 1.0 / float(S)

        for k in range(S):
            # t in [0, 1) — evaluation points
            t_cont = torch.full((batch_size,), k / float(S), device=device, dtype=torch.float32)
            t_idx = self._t_to_index(t_cont)

            v = self.predict_v(x, t_idx)
            x = x + dt * v
            # FIX: NO intermediate clamping — let the ODE evolve freely

        # Only clamp the final output to [-1, 1]
        x = x.clamp(-1.0, 1.0)

        return x

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "model": self.model.state_dict(),
            "num_timesteps": torch.tensor(self.num_timesteps),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict["model"])
        if "num_timesteps" in state_dict:
            try:
                self.num_timesteps = int(state_dict["num_timesteps"].item())
            except Exception:
                pass
