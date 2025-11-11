import torch
import numpy as np

class DiffusionScheduler:
    """
    Simple DDPM scheduler in mel-spectrogram space.
    Predicts noise eps; training uses eps loss.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = torch.device(device)
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1.0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def q_sample(self, x0, t, noise=None):
        """
        x0: (B,1,M,T)
        t: (B,) int32
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_cum = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_cum * x0 + sqrt_one_minus * noise, noise

    def p_sample(self, model, x_t, t, txt_emb):
        """
        Single reverse step using epsilon-prediction model.
        model: predicts eps(x_t, t, txt_emb)
        """
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)

        # epsilon_theta
        eps_theta = model(x_t, t, txt_emb)

        # predicted x0
        x0_pred = (x_t - sqrt_one_minus * eps_theta) / sqrt_recip_alpha

        # posterior mean of q(x_{t-1}|x_t, x0_pred) in DDPM
        if (t == 0).all():
            return x0_pred
        else:
            # standard DDPM step
            beta_t = self.betas[t].view(-1, 1, 1, 1)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_bar_prev = self.alphas_cumprod[(t - 1).clamp(min=0)].view(-1, 1, 1, 1)

            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x_t

            noise = torch.randn_like(x_t)
            var = beta_t  # simple variance
            return mean + torch.sqrt(var) * noise

    def p_sample_loop(self, model, shape, txt_emb, device=None, steps=None):
        """
        Multi-step generation from pure noise.
        shape: (B,1,M,T)
        txt_emb: (B,E)
        steps: optionally use fewer steps than full T (simple thinning)
        """
        if device is None:
            device = self.device
        model.eval()
        x = torch.randn(shape, device=device)

        if steps is None or steps >= self.timesteps:
            times = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            # evenly spaced subset of timesteps
            idx = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
            times = idx

        with torch.no_grad():
            for ti in times:
                t_batch = torch.full((shape[0],), int(ti), device=device, dtype=torch.long)
                x = self.p_sample(model, x, t_batch, txt_emb)
        return x
