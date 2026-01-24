
"""
DDPM variant: x0-parameterization (network predicts x0 instead of eps).

This keeps the same DDPM reverse-process formulation by converting between x0 and eps:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
so:
    eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
    x0  = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
"""

import math
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM_X0(BaseMethod):
    """DDPM where the model predicts x0."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.model = model
        self.device = device

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        # Linear beta schedule (matches the eps version)
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bar[:-1]], dim=0)

        # Buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)

        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_var", torch.clamp(posterior_var, min=1e-20))
        self.register_buffer("posterior_log_var_clipped", torch.log(torch.clamp(posterior_var, min=1e-20)))

        # Posterior mean coeffs (for q(x_{t-1} | x_t, x0))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract a[t] for each batch element and reshape for broadcasting."""
        if t.dtype != torch.long:
            t = t.long()
        t = t.to(a.device)
        out = a.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    # -----------------------
    # Parametrization helpers
    # -----------------------
    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Network predicts x0 directly."""
        return self.model(x_t, t)

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        return (x_t - sqrt_ab * x0) / torch.clamp(sqrt_1mab, min=1e-12)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / torch.clamp(sqrt_ab, min=1e-12)

    # -------------
    # Forward process
    # -------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    # -------------
    # Training loss
    # -------------
    def compute_loss(self, x0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        x0_pred = self.predict_x0(x_t, t)
        loss = F.mse_loss(x0_pred, x0)

        metrics = {"loss": loss.detach().item(), "mse_x0": loss.detach().item()}
        return loss, metrics

    # -------------
    # Reverse process
    # -------------
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Predict x0 directly
        x0_pred = self.predict_x0(x_t, t).clamp(-1.0, 1.0)

        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        var = self._extract(self.posterior_var, t, x_t.shape)
        return mean, var

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, var = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """DDIM-style sampler (eta=0) with optional noise (eta>0)."""
        C, H, W = image_shape
        device = self.device
        x = torch.randn(batch_size, C, H, W, device=device)

        T = self.num_timesteps
        S = int(num_steps) if num_steps is not None else T
        S = max(1, min(S, T))

        # Spaced timesteps
        t_seq = torch.linspace(0, T - 1, S, device=device).round().long()
        t_seq = torch.unique_consecutive(t_seq)
        if t_seq[-1].item() != T - 1:
            t_seq = torch.cat([t_seq, torch.tensor([T - 1], device=device, dtype=torch.long)], dim=0)

        for idx in reversed(range(1, len(t_seq))):
            t = t_seq[idx].expand(batch_size)
            t_prev = t_seq[idx - 1].expand(batch_size)

            x0_pred = self.predict_x0(x, t).clamp(-1.0, 1.0)
            eps = self.predict_eps_from_x0(x, t, x0_pred)

            ab_t = self._extract(self.alpha_bar, t, x.shape)
            ab_prev = self._extract(self.alpha_bar, t_prev, x.shape)

            if eta == 0.0:
                x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps
            else:
                sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / ab_prev)
                noise = torch.randn_like(x)
                x = (
                    torch.sqrt(ab_prev) * x0_pred
                    + torch.sqrt(torch.clamp(1.0 - ab_prev - sigma**2, min=0.0)) * eps
                    + sigma * noise
                )

        return x

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"model": self.model.state_dict(), "num_timesteps": torch.tensor(self.num_timesteps)}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict["model"])
