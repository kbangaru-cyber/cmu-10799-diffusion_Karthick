"""
Denoising Diffusion Probabilistic Models (DDPM)

This implementation matches the "epsilon prediction" (noise prediction) parameterization
from the original DDPM paper.

Key correctness points (the usual bug magnets):
- Training timesteps are sampled uniformly from [0, num_timesteps-1] (FULL horizon).
  This must NOT depend on sampling.num_steps.
- The reverse sampling loop runs from T-1 down to 0.
- Use posterior variance (beta_tilde), and do NOT add noise when t == 0.
- Timesteps passed to the model are integer indices (torch.long). The time embedding
  will cast to float internally.

Also supports fast sampling with DDIM-style striding when num_steps < num_timesteps.
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_denoised: bool = True,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.clip_denoised = bool(clip_denoised)

        # ---- diffusion schedule (linear betas) ----
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bar[:-1]], dim=0)

        # posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1})/(1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))

        # coefficients for posterior mean: mu = c1 * x0 + c2 * xt
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        # register as buffers so they move with .to(device) and are saved in state_dict
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        # convenience precomputes
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a 1-D tensor 'a' at indices t and reshape to [B,1,1,1,...].
        """
        out = a.gather(0, t)
        return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))

    # ---------------------------------------------------------------------
    # Forward process q(x_t | x_0)
    # ---------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    # ---------------------------------------------------------------------
    # Training objective
    # ---------------------------------------------------------------------
    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Standard DDPM loss: E_{t,eps} || eps - eps_theta(x_t, t) ||^2
        """
        B = x0.shape[0]
        device = x0.device

        # CRITICAL: sample timesteps from the FULL horizon
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)

        eps_pred = self.model(xt, t)

        loss = F.mse_loss(eps_pred, noise)
        metrics = {"loss": loss, "mse": loss}
        return loss, metrics

    # ---------------------------------------------------------------------
    # Reverse process p(x_{t-1} | x_t)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def p_mean_variance(self, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean/variance for p(x_{t-1} | x_t) using epsilon prediction.
        Returns: mean, variance, log_variance, x0_pred
        """
        eps_pred = self.model(xt, t)

        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, xt.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, xt.shape)

        # x0_pred = (xt - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        x0_pred = (xt - sqrt_1mab * eps_pred) / torch.clamp(sqrt_ab, min=1e-12)
        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # mu = c1*x0 + c2*xt
        c1 = self._extract(self.posterior_mean_coef1, t, xt.shape)
        c2 = self._extract(self.posterior_mean_coef2, t, xt.shape)
        mean = c1 * x0_pred + c2 * xt

        var = self._extract(self.posterior_variance, t, xt.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, xt.shape)
        return mean, var, log_var, x0_pred

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).
        """
        mean, var, log_var, _ = self.p_mean_variance(xt, t)

        # no noise when t == 0
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view((t.shape[0],) + (1,) * (xt.ndim - 1))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate samples.

        Args:
            batch_size: number of images
            image_shape: (C, H, W)
            num_steps: number of reverse steps. If None or == num_timesteps -> full DDPM.
                       If < num_timesteps -> DDIM-style striding.
            eta: DDIM noise parameter. 0.0 = deterministic DDIM. 1.0 ~ stochastic.
        """
        C, H, W = image_shape
        device = self.device

        T = self.num_timesteps
        if num_steps is None:
            num_steps = T
        num_steps = int(num_steps)

        x = torch.randn(batch_size, C, H, W, device=device)

        if num_steps >= T:
            # Full DDPM chain
            for t_val in reversed(range(T)):
                t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                x = self.p_sample(x, t)
            return x

        # DDIM-style striding
        # Build a decreasing list of timesteps in [T-1 ... 0] of length num_steps
        # We ensure they are unique and strictly decreasing.
        ts = torch.linspace(T - 1, 0, steps=num_steps, device=device)
        ts = torch.round(ts).long()
        # make strictly decreasing unique
        ts = torch.unique_consecutive(ts)
        if ts[0] != T - 1:
            ts = torch.cat([torch.tensor([T - 1], device=device, dtype=torch.long), ts])
        if ts[-1] != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)])
        # ensure decreasing
        ts = torch.sort(ts, descending=True).values

        for i in range(len(ts) - 1):
            t = ts[i].item()
            t_prev = ts[i + 1].item()
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps_pred = self.model(x, t_batch)

            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t_prev]

            # x0_pred from eps
            x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(torch.clamp(ab_t, min=1e-12))
            if self.clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM sigma
            # sigma^2 = eta^2 * (1-ab_prev)/(1-ab_t) * (1 - ab_t/ab_prev)
            sigma2 = (eta ** 2) * (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / torch.clamp(ab_prev, min=1e-12))
            sigma2 = torch.clamp(sigma2, min=0.0)
            sigma = torch.sqrt(sigma2)

            # direction pointing to x_t
            pred_dir = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma2, min=0.0)) * eps_pred

            x = torch.sqrt(ab_prev) * x0_pred + pred_dir
            if t_prev != 0:
                x = x + sigma * torch.randn_like(x)

        return x

    # ---------------------------------------------------------------------
    # Checkpoint helpers (optional)
    # ---------------------------------------------------------------------
    def save_state(self) -> Dict:
        """
        Return method-specific state to save in checkpoints (optional).
        """
        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "clip_denoised": self.clip_denoised,
        }

    def load_state(self, state: Dict) -> None:
        """
        Load method-specific state (optional).
        """
        # Schedule buffers are derived from (num_timesteps, beta_start, beta_end).
        # If these differ, you should re-instantiate the method from config instead.
        pass

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            clip_denoised=ddpm_config.get("clip_denoised", True),
        )
