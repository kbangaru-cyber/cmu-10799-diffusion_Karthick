"""
Flow Matching (straight-line / rectified flow style)

Trains a velocity field v_theta(x_t, t) that transports noise to data along
straight paths:

    x_t = (1 - t) * x_1 + t * x_0,   where x_1 ~ N(0, I),  t ~ Uniform[0, 1]
    v*(x_t, t) = d x_t / dt = x_0 - x_1

We discretize continuous time t in [0, 1] into integer indices in [0, T-1]
so we can reuse the exact same time-embedding interface your DDPM model uses:
    v_pred = model(x_t, t_idx)

Sampling: Euler integrate from t=0 to t=1:
    x <- x + dt * v_theta(x, t)

FIXES:
1. Removed aggressive clamping during intermediate steps (major issue!)
2. Added Heun's method (2nd order) as default solver for better accuracy
3. Fixed time discretization to properly cover [0, 1]
4. Added logit-normal time sampling for better training coverage
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
        time_sampling: str = "uniform",  # "uniform" or "logit_normal"
    ):
        super().__init__(model=model, device=device)
        self.num_timesteps = int(num_timesteps)
        self.time_sampling = time_sampling

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
        time_sampling = fm_cfg.get("time_sampling", "uniform")
        method = cls(model=model, device=device, num_timesteps=num_timesteps, time_sampling=time_sampling)
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

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample time values for training.
        
        Options:
        - "uniform": t ~ U[0, 1] (standard)
        - "logit_normal": t = sigmoid(N(0, 1)), concentrates samples near t=0.5
                          which can help with middle-of-path learning
        """
        if self.time_sampling == "logit_normal":
            # Logit-normal: sample z ~ N(0, 1), then t = sigmoid(z)
            z = torch.randn(batch_size, device=device)
            t_cont = torch.sigmoid(z)
        else:
            # Uniform [0, 1]
            t_cont = torch.rand(batch_size, device=device, dtype=torch.float32)
        
        # Clamp to avoid exact endpoints which can cause numerical issues
        t_cont = t_cont.clamp(1e-5, 1.0 - 1e-5)
        return t_cont

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
          - sample t ~ U[0,1] (or logit_normal)
          - x_t = (1-t)x1 + t x0
          - target velocity v* = x0 - x1
          - loss = MSE(v_pred, v*)
        """
        B = x0.shape[0]
        device = x0.device

        # Endpoint noise (source distribution)
        x1 = torch.randn_like(x0)

        # Sample time
        t_cont = self._sample_time(B, device)
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
        solver: str = "heun",  # "euler" or "heun" (2nd order, recommended)
        clamp: bool = False,  # CHANGED: Default to False! Only clamp at the end
        **kwargs,
    ) -> torch.Tensor:
        """
        ODE integration from noise (t=0) to data (t=1).

        Args:
            num_steps: number of integration steps S (default: 200 if None)
            solver: "euler" (1st order) or "heun" (2nd order, more accurate)
            clamp: if True, clamp ONLY the final output to [-1,1]
                   DO NOT clamp intermediate steps!

        Returns:
            samples in [-1,1] range (assuming your training images are in [-1,1])
        """
        C, H, W = image_shape
        device = self.device

        # Start from noise
        x = torch.randn(batch_size, C, H, W, device=device)

        S = int(num_steps) if num_steps is not None else 200
        S = max(1, S)
        dt = 1.0 / float(S)

        if solver == "heun":
            x = self._sample_heun(x, S, dt, batch_size, device)
        else:
            x = self._sample_euler(x, S, dt, batch_size, device)

        # Clamp ONLY the final output (not intermediate steps!)
        if clamp:
            x = x.clamp(-1.0, 1.0)

        return x

    def _sample_euler(
        self,
        x: torch.Tensor,
        S: int,
        dt: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Euler method (1st order): x_{k+1} = x_k + dt * v(x_k, t_k)
        """
        for k in range(S):
            t_cont = torch.full((batch_size,), k / float(S), device=device, dtype=torch.float32)
            t_idx = self._t_to_index(t_cont)

            v = self.predict_v(x, t_idx)
            x = x + dt * v
            # NO clamping here! This was the bug.

        return x

    def _sample_heun(
        self,
        x: torch.Tensor,
        S: int,
        dt: float,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Heun's method (2nd order Runge-Kutta):
        
        k1 = v(x_k, t_k)
        x_tilde = x_k + dt * k1
        k2 = v(x_tilde, t_{k+1})
        x_{k+1} = x_k + dt/2 * (k1 + k2)
        
        This is more accurate than Euler and recommended for flow matching.
        """
        for k in range(S):
            t_k = k / float(S)
            t_k1 = (k + 1) / float(S)
            
            t_cont_k = torch.full((batch_size,), t_k, device=device, dtype=torch.float32)
            t_cont_k1 = torch.full((batch_size,), min(t_k1, 1.0), device=device, dtype=torch.float32)
            
            t_idx_k = self._t_to_index(t_cont_k)
            t_idx_k1 = self._t_to_index(t_cont_k1)

            # Heun's method
            k1 = self.predict_v(x, t_idx_k)
            x_tilde = x + dt * k1
            k2 = self.predict_v(x_tilde, t_idx_k1)
            x = x + (dt / 2.0) * (k1 + k2)
            # NO clamping here!

        return x

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        solver: str = "heun",
        save_every: int = 10,
        **kwargs,
    ) -> Tuple[torch.Tensor, list]:
        """
        Sample and return intermediate states for visualization/debugging.
        Useful for checking if the ODE integration is working correctly.
        """
        C, H, W = image_shape
        device = self.device

        x = torch.randn(batch_size, C, H, W, device=device)
        trajectory = [x.clone().cpu()]

        S = int(num_steps) if num_steps is not None else 200
        S = max(1, S)
        dt = 1.0 / float(S)

        for k in range(S):
            t_k = k / float(S)
            t_cont_k = torch.full((batch_size,), t_k, device=device, dtype=torch.float32)
            t_idx_k = self._t_to_index(t_cont_k)

            if solver == "heun" and k < S - 1:
                t_k1 = (k + 1) / float(S)
                t_cont_k1 = torch.full((batch_size,), min(t_k1, 1.0), device=device, dtype=torch.float32)
                t_idx_k1 = self._t_to_index(t_cont_k1)
                
                k1 = self.predict_v(x, t_idx_k)
                x_tilde = x + dt * k1
                k2 = self.predict_v(x_tilde, t_idx_k1)
                x = x + (dt / 2.0) * (k1 + k2)
            else:
                v = self.predict_v(x, t_idx_k)
                x = x + dt * v

            if (k + 1) % save_every == 0:
                trajectory.append(x.clone().cpu())

        trajectory.append(x.clone().cpu())
        return x.clamp(-1.0, 1.0), trajectory

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
