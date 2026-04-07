import torch
from torch import nn


def rmse_loss(prediction, target):
    """ACTrajNet-style RMSE: sqrt(MSE over all elements)."""
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(prediction, target))


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
        """KL(N(mu, sigma) || N(0, I)) summed over agents and latent dims."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, prediction, target, mu, logvar):
        recon = rmse_loss(prediction, target)
        kl = self.kl_divergence(mu, logvar)
        return recon + self.kl_weight * kl, recon, kl
