import torch
from torch import nn


def rmse_loss(prediction, target):
    """
    Per-agent RMSE sum — matches original ACTrajNet loss_func loop:
      for agent in range(N): loss += sqrt(MSE(pred_agent, target_agent))
    prediction/target: (pred_steps, N, 3)
    """
    # MSE per agent: average over (time, channels), keep agent dim
    mse_per_agent = (prediction - target).pow(2).mean(dim=(0, 2))  # (N,)
    return torch.sqrt(mse_per_agent + 1e-8).sum()


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
        Per-agent KL sum — matches original: each agent's KL summed over latent,
        then summed across agents.  mu/logvar: (N, latent_dim)
        """
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_element = torch.clamp(kl_per_element, min=self.free_bits)
        return kl_per_element.sum()

    def forward(self, prediction, target, mu, logvar):
        recon = rmse_loss(prediction, target)
        kl = self.kl_divergence(mu, logvar)
        return recon + self.kl_weight * kl, recon, kl
