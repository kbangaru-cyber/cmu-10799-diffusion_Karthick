"""
Classifier-Free Guidance (CFG) with Flow Matching

Combines flow matching with classifier-free guidance for attribute-conditional
image generation and editing.

Training:
  - Same as flow matching, but the model receives an attribute condition c
  - With probability p_uncond, the condition is dropped (replaced by zeros)
  - The model learns both conditional v_θ(x_t, t, c) and unconditional v_θ(x_t, t, ∅)

Sampling (CFG):
  - At each step, run two forward passes:
      v_uncond = v_θ(x_t, t, ∅)
      v_cond   = v_θ(x_t, t, c)
  - Combine: v_guided = (1 + w) * v_cond - w * v_uncond
  - Euler step: x_{t+dt} = x_t + dt * v_guided

Image Editing (noise-and-denoise):
  - Given a real image x, add noise to timestep t_edit
  - Denoise from t_edit to t=1 using CFG with target attributes
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class CFGFlowMatching(BaseMethod):
    """
    Flow Matching with Classifier-Free Guidance.

    Args:
        model: Conditional U-Net (CondUNet) that accepts (x, t, c)
        device: torch device
        num_timesteps: number of discrete timestep indices (for time embedding)
        p_uncond: probability of dropping condition during training
        guidance_scale: default guidance scale w for sampling
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
        p_uncond: float = 0.1,
        guidance_scale: float = 2.0,
        num_attributes: int = 3,
    ):
        super().__init__(model=model, device=device)
        self.num_timesteps = int(num_timesteps)
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.num_attributes = num_attributes

    @classmethod
    def from_config(cls, model, config, device):
        """Create from config dict."""
        fm_cfg = config.get("flow_matching", {}) or {}
        cfg_cfg = config.get("cfg", {}) or {}
        cond_cfg = config.get("conditioning", {}) or {}

        num_timesteps = int(fm_cfg.get("num_timesteps", 1000))
        p_uncond = float(cfg_cfg.get("p_uncond", 0.1))
        guidance_scale = float(cfg_cfg.get("guidance_scale", 2.0))
        num_attributes = int(cond_cfg.get("num_attributes", 3))

        method = cls(
            model=model,
            device=device,
            num_timesteps=num_timesteps,
            p_uncond=p_uncond,
            guidance_scale=guidance_scale,
            num_attributes=num_attributes,
        )
        return method.to(device)

    def _t_to_index(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Map continuous t in [0,1] -> integer timestep index in [0, T-1]."""
        t_cont = t_cont.clamp(0.0, 1.0)
        if self.num_timesteps <= 1:
            return torch.zeros_like(t_cont, dtype=torch.long)
        idx = torch.round(t_cont * (self.num_timesteps - 1)).long()
        return idx

    def predict_v(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t, c)."""
        if t_idx.dtype != torch.long:
            t_idx = t_idx.long()
        return self.model(x_t, t_idx, c)

    def compute_loss(
        self,
        x0: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        CFG Flow Matching training loss.

        Same as standard flow matching, but:
          - Receives attribute condition c alongside images
          - Drops c -> zeros with probability p_uncond

        Args:
            x0: (B, C, H, W) clean images in [-1, 1]
            c:  (B, K) binary attribute vectors
        """
        B = x0.shape[0]
        device = x0.device

        # Null condition if not provided
        if c is None:
            c = torch.zeros(B, self.num_attributes, device=device)

        # --- CFG condition dropout ---
        # With probability p_uncond, replace c with zeros (null condition)
        drop_mask = torch.rand(B, device=device) < self.p_uncond
        c_train = c.clone()
        c_train[drop_mask] = 0.0

        # Sample noise (source distribution)
        x1 = torch.randn_like(x0)

        # Continuous time in [0, 1]
        t_cont = torch.rand(B, device=device, dtype=torch.float32)
        t_view = t_cont.view(B, 1, 1, 1)

        # Interpolation along straight path
        x_t = (1.0 - t_view) * x1 + t_view * x0

        # Target velocity
        v_target = x0 - x1

        # Discretize time for model
        t_idx = self._t_to_index(t_cont).to(device)

        # Predict velocity (with possibly-dropped condition)
        v_pred = self.predict_v(x_t, t_idx, c_train)

        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": float(loss.detach().item()),
            "mse": float(loss.detach().item()),
            "drop_rate": float(drop_mask.float().mean().item()),
        }
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        eta: float = 0.0,  # unused, interface compat
        **kwargs,
    ) -> torch.Tensor:
        """
        CFG-guided Euler sampling from noise (t=0) to data (t=1).

        At each step:
          v_uncond = model(x, t, null)
          v_cond   = model(x, t, c)
          v_guided = (1 + w) * v_cond - w * v_uncond
          x += dt * v_guided

        Args:
            batch_size: number of samples
            image_shape: (C, H, W)
            num_steps: Euler steps (default 200)
            c: (B, K) target attribute vector. If None, samples unconditionally.
            guidance_scale: override default guidance scale w
        """
        C, H, W = image_shape
        device = self.device
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Start from noise
        x = torch.randn(batch_size, C, H, W, device=device)

        S = int(num_steps) if num_steps is not None else 200
        S = max(1, S)
        dt = 1.0 / float(S)

        # Null condition
        c_null = torch.zeros(batch_size, self.num_attributes, device=device)

        # If no condition provided, sample unconditionally (w=0 effectively)
        if c is None:
            c = c_null
            w = 0.0

        for k in range(S):
            t_cont = torch.full(
                (batch_size,), k / float(S), device=device, dtype=torch.float32
            )
            t_idx = self._t_to_index(t_cont)

            if w > 0.0:
                # Two forward passes for CFG
                v_uncond = self.predict_v(x, t_idx, c_null)
                v_cond = self.predict_v(x, t_idx, c)
                # CFG combination: ṽ = (1 + w) * v_cond - w * v_uncond
                v = (1.0 + w) * v_cond - w * v_uncond
            else:
                # No guidance, just conditional (or unconditional) prediction
                v = self.predict_v(x, t_idx, c)

            x = x + dt * v
            # No intermediate clamping — let ODE evolve freely

        # Final clamp only
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def edit(
        self,
        x_orig: torch.Tensor,
        c_target: torch.Tensor,
        t_edit: float = 0.5,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Edit a real image via noise-and-denoise with CFG.

        Steps:
          1. Add noise to x_orig up to time t_edit:
             x_{t_edit} = (1 - t_edit) * noise + t_edit * x_orig
          2. Denoise from t_edit to t=1 using CFG with target attributes

        Args:
            x_orig: (B, C, H, W) real images in [-1, 1]
            c_target: (B, K) target attribute vector for the edit
            t_edit: noise level / edit strength (0 = pure noise, 1 = no edit)
                    Lower values = stronger edit, more identity change
                    Typical range: 0.3 - 0.7
            num_steps: total Euler steps for the full [0,1] interval
            guidance_scale: CFG guidance scale w
        """
        B, C, H, W = x_orig.shape
        device = x_orig.device
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        S = int(num_steps) if num_steps is not None else 200
        S = max(1, S)
        dt = 1.0 / float(S)

        # Step 1: Add noise to get x at time t_edit
        # x_t = (1 - t) * noise + t * data
        noise = torch.randn_like(x_orig)
        x = (1.0 - t_edit) * noise + t_edit * x_orig

        # Step 2: Continue Euler integration from t_edit to t=1
        # Find the starting step index
        k_start = int(t_edit * S)

        c_null = torch.zeros(B, self.num_attributes, device=device)

        for k in range(k_start, S):
            t_cont = torch.full(
                (B,), k / float(S), device=device, dtype=torch.float32
            )
            t_idx = self._t_to_index(t_cont)

            if w > 0.0:
                v_uncond = self.predict_v(x, t_idx, c_null)
                v_cond = self.predict_v(x, t_idx, c_target)
                v = (1.0 + w) * v_cond - w * v_uncond
            else:
                v = self.predict_v(x, t_idx, c_target)

            x = x + dt * v

        return x.clamp(-1.0, 1.0)

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
