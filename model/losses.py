import torch
from torch import nn


def rmse_loss(prediction, target):
    """RMSE loss — matches ACTrajNet original."""
    return torch.sqrt(nn.functional.mse_loss(prediction, target))


class HAINetLoss(nn.Module):
    """
    Combined loss: RMSE + KL divergence with anti-collapse measures.

    Anti-collapse strategy:
    1. KL uses mean (not sum) to balance with RMSE regardless of latent dim
    2. Free bits: minimum KL per dimension prevents full collapse
    3. Cyclical annealing: controlled externally via kl_weight
    """

    def __init__(self, kl_weight=1.0, free_bits=0.05):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_bits = free_bits

    def kl_divergence(self, mu, logvar):
        """KL(N(mu, sigma) || N(0, I)) — sum over latent dims, mean over batch.
        Matches ACTrajNet original per-sample KL behavior."""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        return kl_per_dim.sum(dim=-1).mean()

    def forward(self, prediction, target, mu, logvar):
        recon = rmse_loss(prediction, target)
        kl = self.kl_divergence(mu, logvar)
        return recon + self.kl_weight * kl, recon, kl
