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


def _soft_label_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _score_quality_targets(ade, fde, fde_weight=0.75, temperature=0.35):
    ade_scale = ade.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
    fde_scale = fde.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
    cost = ade.detach() / ade_scale + fde_weight * (fde.detach() / fde_scale)
    return torch.softmax(-cost / max(temperature, 1e-3), dim=1)


def _prototype_soft_targets(gt_summary_5d, proto_summary_5d, topm=5, temperature=1.0):
    safe_topm = min(max(int(topm), 1), proto_summary_5d.size(0))
    distances = ((gt_summary_5d[:, None] - proto_summary_5d[None]) ** 2).sum(dim=-1)
    positive_idx = distances.topk(safe_topm, dim=-1, largest=False).indices
    positive_dist = distances.gather(1, positive_idx)
    scale = positive_dist.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
    positive_weight = torch.softmax(-positive_dist / (scale * max(float(temperature), 1e-3)), dim=1)
    targets = torch.zeros_like(distances)
    targets.scatter_(1, positive_idx, positive_weight)
    return targets


def _stage_allows_proto_aux(stage_cfg, mode):
    if mode == "all":
        return True
    if mode == "pre_refiner":
        return not bool(stage_cfg.get("enable_refiner", False))
    if mode == "refiner":
        return bool(stage_cfg.get("enable_refiner", False))
    if mode == "basis_warmup":
        return stage_cfg.get("name") == "basis_warmup"
    raise ValueError(f"Unsupported proto_aux_stage: {mode}")


class ProtoBasisLoss(nn.Module):
    def __init__(
        self,
        lambda_xyz=1.0,
        lambda_fde=0.8,
        lambda_proto=0.4,
        lambda_proto_soft=0.0,
        lambda_res=0.2,
        lambda_score=0.5,
        lambda_rank=0.1,
        lambda_div=0.05,
        lambda_coeff=0.02,
        lambda_smooth=0.10,
        proto_soft_topm=5,
        proto_soft_temperature=1.0,
        proto_aux_stage="all",
        score_hard_mix=0.25,
        score_fde_weight=0.75,
        score_soft_temperature=0.35,
    ):
        super().__init__()
        self.lambda_xyz = lambda_xyz
        self.lambda_fde = lambda_fde
        self.lambda_proto = lambda_proto
        self.lambda_proto_soft = lambda_proto_soft
        self.lambda_res = lambda_res
        self.lambda_score = lambda_score
        self.lambda_rank = lambda_rank
        self.lambda_div = lambda_div
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.proto_soft_topm = proto_soft_topm
        self.proto_soft_temperature = proto_soft_temperature
        self.proto_aux_stage = proto_aux_stage
        self.score_hard_mix = min(max(score_hard_mix, 0.0), 1.0)
        self.score_fde_weight = score_fde_weight
        self.score_soft_temperature = score_soft_temperature

    def forward(self, outputs, batch, stage_cfg):
        pred_xyz = outputs["pred_xyz"]
        pred_score = outputs["pred_score"]
        proto_logits = outputs["proto_logits"]
        top_proto_idx = outputs["top_proto_idx"]
        aux = outputs["aux"]
        gt_xyz = batch["fut_xyz"]
        gt_proto_id = batch["gt_proto_id"]
        gt_proto_residual = batch["gt_proto_residual"]
        gt_summary_5d = batch["proto_summary_5d"]

        best_idx, ade, fde = _winner_indices(pred_xyz, gt_xyz)
        winner_xyz = _gather_candidates(pred_xyz, best_idx)
        coeff_for_regularization = aux.get("coeff_delta", aux["coeff"])
        winner_coeff = _gather_candidates(coeff_for_regularization, best_idx)

        xyz_loss = F.smooth_l1_loss(winner_xyz, gt_xyz)
        fde_loss = F.smooth_l1_loss(winner_xyz[:, -1], gt_xyz[:, -1])
        proto_loss = F.cross_entropy(proto_logits, gt_proto_id)
        proto_soft_targets = _prototype_soft_targets(
            gt_summary_5d,
            aux["proto_summary_5d"],
            topm=self.proto_soft_topm,
            temperature=self.proto_soft_temperature,
        )
        proto_soft_loss = _soft_label_cross_entropy(proto_logits, proto_soft_targets)

        match_mask = top_proto_idx.eq(gt_proto_id.unsqueeze(1))
        gt_slot = match_mask.float().argmax(dim=1)
        pred_residual = aux["endpoint_residual"][torch.arange(gt_slot.size(0), device=gt_slot.device), gt_slot]
        res_loss = F.smooth_l1_loss(pred_residual, gt_proto_residual)

        hard_score_loss = F.cross_entropy(pred_score, best_idx)
        soft_score_targets = _score_quality_targets(
            ade,
            fde,
            fde_weight=self.score_fde_weight,
            temperature=self.score_soft_temperature,
        )
        soft_score_loss = _soft_label_cross_entropy(pred_score, soft_score_targets)
        score_loss = self.score_hard_mix * hard_score_loss + (1.0 - self.score_hard_mix) * soft_score_loss
        rank_loss = _pairwise_margin(pred_score, best_idx)
        div_loss = _diversity_repulsion(pred_xyz)
        coeff_loss = winner_coeff.pow(2).mean()
        smooth_loss = _trajectory_smoothness(winner_xyz)

        total = self.lambda_xyz * xyz_loss
        total = total + self.lambda_fde * fde_loss
        total = total + self.lambda_proto * proto_loss
        if _stage_allows_proto_aux(stage_cfg, self.proto_aux_stage):
            total = total + self.lambda_proto_soft * proto_soft_loss
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
            "proto_soft": float(proto_soft_loss.detach().item()),
            "res": float(res_loss.detach().item()),
            "score": float(score_loss.detach().item()),
            "rank": float(rank_loss.detach().item()),
            "div": float(div_loss.detach().item()),
            "coeff": float(coeff_loss.detach().item()),
            "smooth": float(smooth_loss.detach().item()),
            "winner_ade": float(ade.min(dim=1).values.mean().detach().item()),
        }
        return total, stats
