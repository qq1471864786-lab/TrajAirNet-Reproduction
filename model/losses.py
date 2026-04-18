import torch
import torch.nn.functional as F
from torch import nn


def _candidate_l2(pred_xyz, gt_xyz):
    return torch.linalg.norm(pred_xyz - gt_xyz[:, None], dim=-1)


def _winner_indices(pred_xyz, gt_xyz):
    l2 = _candidate_l2(pred_xyz, gt_xyz)
    ade = l2.mean(dim=-1)
    fde = l2[..., -1]
    return ade.argmin(dim=-1), ade, fde


def _gather_candidates(tensor, indices):
    view_shape = [indices.size(0), 1] + [1] * (tensor.dim() - 2)
    expand_shape = [indices.size(0), 1] + list(tensor.shape[2:])
    gather_index = indices.view(*view_shape).expand(*expand_shape)
    return torch.gather(tensor, 1, gather_index).squeeze(1)


def _pairwise_margin(scores, best_idx, margin=0.1):
    best_score = scores.gather(1, best_idx.unsqueeze(1))
    penalties = F.relu(margin - (best_score - scores))
    mask = torch.ones_like(penalties)
    mask.scatter_(1, best_idx.unsqueeze(1), 0.0)
    return (penalties * mask).mean()


def _diversity_repulsion(pred_xyz, tau=1.0):
    endpoints = pred_xyz[:, :, -1]
    pairwise = torch.cdist(endpoints, endpoints)
    mask = ~torch.eye(pairwise.size(-1), dtype=torch.bool, device=pairwise.device)
    if mask.sum() == 0:
        return pairwise.new_tensor(0.0)
    repulsion = torch.exp(-pairwise / tau)
    return repulsion.masked_select(mask.unsqueeze(0)).mean()


def _trajectory_smoothness(xyz):
    if xyz.size(1) < 3:
        return xyz.new_tensor(0.0)
    second_diff = xyz[:, 2:] - 2.0 * xyz[:, 1:-1] + xyz[:, :-2]
    return second_diff.abs().mean()


class ProtoBasisLoss(nn.Module):
    def __init__(
        self,
        lambda_xyz=1.0,
        lambda_fde=0.8,
        lambda_proto=0.4,
        lambda_res=0.2,
        lambda_score=0.5,
        lambda_rank=0.1,
        lambda_div=0.05,
        lambda_coeff=0.02,
        lambda_smooth=0.10,
    ):
        super().__init__()
        self.lambda_xyz = lambda_xyz
        self.lambda_fde = lambda_fde
        self.lambda_proto = lambda_proto
        self.lambda_res = lambda_res
        self.lambda_score = lambda_score
        self.lambda_rank = lambda_rank
        self.lambda_div = lambda_div
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth

    def forward(self, outputs, batch, stage_cfg):
        pred_xyz = outputs["pred_xyz"]
        pred_score = outputs["pred_score"]
        proto_logits = outputs["proto_logits"]
        top_proto_idx = outputs["top_proto_idx"]
        aux = outputs["aux"]
        gt_xyz = batch["fut_xyz"]
        gt_proto_id = batch["gt_proto_id"]
        gt_proto_residual = batch["gt_proto_residual"]

        best_idx, ade, _ = _winner_indices(pred_xyz, gt_xyz)
        winner_xyz = _gather_candidates(pred_xyz, best_idx)
        winner_coeff = _gather_candidates(aux["coeff"], best_idx)

        xyz_loss = F.smooth_l1_loss(winner_xyz, gt_xyz)
        fde_loss = F.smooth_l1_loss(winner_xyz[:, -1], gt_xyz[:, -1])
        proto_loss = F.cross_entropy(proto_logits, gt_proto_id)

        match_mask = top_proto_idx.eq(gt_proto_id.unsqueeze(1))
        gt_slot = match_mask.float().argmax(dim=1)
        pred_residual = aux["endpoint_residual"][torch.arange(gt_slot.size(0), device=gt_slot.device), gt_slot]
        res_loss = F.smooth_l1_loss(pred_residual, gt_proto_residual)

        score_loss = F.cross_entropy(pred_score, best_idx)
        rank_loss = _pairwise_margin(pred_score, best_idx)
        div_loss = _diversity_repulsion(pred_xyz)
        coeff_loss = winner_coeff.pow(2).mean()
        smooth_loss = _trajectory_smoothness(winner_xyz)

        total = self.lambda_xyz * xyz_loss
        total = total + self.lambda_fde * fde_loss
        total = total + self.lambda_proto * proto_loss
        total = total + self.lambda_res * res_loss
        total = total + self.lambda_score * score_loss
        total = total + stage_cfg["rank_weight"] * self.lambda_rank * rank_loss
        total = total + stage_cfg["div_weight"] * self.lambda_div * div_loss
        total = total + self.lambda_coeff * coeff_loss
        total = total + self.lambda_smooth * smooth_loss

        stats = {
            "xyz": float(xyz_loss.detach().item()),
            "fde": float(fde_loss.detach().item()),
            "proto": float(proto_loss.detach().item()),
            "res": float(res_loss.detach().item()),
            "score": float(score_loss.detach().item()),
            "rank": float(rank_loss.detach().item()),
            "div": float(div_loss.detach().item()),
            "coeff": float(coeff_loss.detach().item()),
            "smooth": float(smooth_loss.detach().item()),
            "winner_ade": float(ade.min(dim=1).values.mean().detach().item()),
        }
        return total, stats
