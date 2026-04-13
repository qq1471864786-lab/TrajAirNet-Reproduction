import torch
from torch import nn


def rmse_loss(prediction, target):
    mse_per_agent = (prediction - target).pow(2).mean(dim=(0, 2))
    return torch.sqrt(mse_per_agent + 1e-8).mean()


def vertical_rmse_loss(prediction, target):
    mse_per_agent = (prediction[..., 2] - target[..., 2]).pow(2).mean(dim=0)
    return torch.sqrt(mse_per_agent + 1e-8).mean()


def endpoint_rmse_loss(prediction, target):
    final_prediction = prediction[-1]
    final_target = target[-1]
    mse_per_agent = (final_prediction - final_target).pow(2).mean(dim=-1)
    return torch.sqrt(mse_per_agent + 1e-8).mean()


class TrajectoryForecastLoss(nn.Module):
    def __init__(self, kl_weight=1.0, free_bits=0.0, vertical_loss_weight=0.0, endpoint_loss_weight=0.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.free_bits = free_bits
        self.vertical_loss_weight = vertical_loss_weight
        self.endpoint_loss_weight = endpoint_loss_weight

    def kl_divergence(self, mu, logvar):
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_element = torch.clamp(kl_per_element, min=self.free_bits)
        kl_per_agent = kl_per_element.sum(dim=-1)
        return kl_per_agent.mean()

    def forward(self, prediction, target, mu, logvar):
        recon = rmse_loss(prediction, target)
        kl = self.kl_divergence(mu, logvar)
        vertical = vertical_rmse_loss(prediction, target)
        endpoint = endpoint_rmse_loss(prediction, target)
        total = (
            recon
            + self.kl_weight * kl
            + self.vertical_loss_weight * vertical
            + self.endpoint_loss_weight * endpoint
        )
        return total, recon, kl, vertical, endpoint
