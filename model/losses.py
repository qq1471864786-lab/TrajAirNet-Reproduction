import torch
from torch import nn


def rmse_loss(prediction, target):
    """
    Per-agent RMSE, averaged over agents.
    Preserves the same recon/KL ratio as original ACTrajNet (per-agent loss),
    but normalised by N so gradient magnitude is independent of batch size.
    prediction/target: (pred_steps, N, 3)
    """
    mse_per_agent = (prediction - target).pow(2).mean(dim=(0, 2))  # (N,)
    return torch.sqrt(mse_per_agent + 1e-8).mean()


class HAINetLoss(nn.Module):
    """
    Combined loss aligned with ACTrajNet baseline:
    sum(agent_RMSE) + kl_weight * sum(agent_KL)
    """

    def __init__(self, kl_weight=1.0, free_bits=0.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_bits = free_bits

    def kl_divergence(self, mu, logvar):
        """
        Per-agent KL, averaged over agents.
        Each agent: sum over latent dims.  Then mean over agents.
        mu/logvar: (N, latent_dim)
        """
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_element = torch.clamp(kl_per_element, min=self.free_bits)
        kl_per_agent = kl_per_element.sum(dim=-1)  # (N,)
        return kl_per_agent.mean()

    def forward(self, prediction, target, mu, logvar):
        recon = rmse_loss(prediction, target)
        kl = self.kl_divergence(mu, logvar)
        return recon + self.kl_weight * kl, recon, kl
