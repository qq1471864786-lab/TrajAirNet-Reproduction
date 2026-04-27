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


def _prototype_loss(logits, targets, proto_frequency=None, focal_gamma=0.0, freq_weight_power=0.0, freq_weight_max=5.0):
    focal_gamma = max(float(focal_gamma), 0.0)
    freq_weight_power = max(float(freq_weight_power), 0.0)
    if focal_gamma <= 0.0 and freq_weight_power <= 0.0:
        return F.cross_entropy(logits, targets)

    log_prob = F.log_softmax(logits, dim=-1)
    target_log_prob = log_prob.gather(1, targets[:, None]).squeeze(1)
    loss = -target_log_prob
    if focal_gamma > 0.0:
        target_prob = target_log_prob.detach().exp()
        loss = (1.0 - target_prob).pow(focal_gamma) * loss

    if freq_weight_power > 0.0 and proto_frequency is not None:
        frequency = proto_frequency.detach().to(logits.device).float().clamp_min(1e-6)
        class_weight = (frequency.mean() / frequency).pow(freq_weight_power)
        class_weight = class_weight.clamp(max=max(float(freq_weight_max), 1.0))
        sample_weight = class_weight.gather(0, targets)
        return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    return loss.mean()


def _gt_proto_aligned_losses(pred_xyz, gt_xyz, top_proto_idx, gt_proto_id, ade):
    if top_proto_idx is None or top_proto_idx.size(1) != pred_xyz.size(1):
        zero = pred_xyz.sum() * 0.0
        return zero, zero, pred_xyz.new_tensor(0.0)

    match_mode = top_proto_idx.eq(gt_proto_id[:, None])
    hit_mask = match_mode.any(dim=1)
    if not hit_mask.any():
        zero = pred_xyz.sum() * 0.0
        return zero, zero, pred_xyz.new_tensor(0.0)

    masked_ade = ade.masked_fill(~match_mode, float("inf"))
    aligned_idx = masked_ade.argmin(dim=1)
    aligned_xyz = _gather_candidates(pred_xyz, aligned_idx)
    shape_loss = F.smooth_l1_loss(aligned_xyz[hit_mask], gt_xyz[hit_mask])
    fde_loss = F.smooth_l1_loss(aligned_xyz[hit_mask, -1], gt_xyz[hit_mask, -1])
    return shape_loss, fde_loss, hit_mask.float().mean()


def _gt_proto_coeff_loss(coeff, gt_basis_coeff, top_proto_idx, gt_proto_id):
    if gt_basis_coeff is None or coeff.size(-1) != gt_basis_coeff.size(-1) or coeff.size(-1) == 0:
        return coeff.sum() * 0.0

    if top_proto_idx is None or top_proto_idx.size(1) != coeff.size(1):
        return coeff.sum() * 0.0

    match_mode = top_proto_idx.eq(gt_proto_id[:, None])
    hit_mask = match_mode.any(dim=1)
    if not hit_mask.any():
        return coeff.sum() * 0.0

    coeff_distance = (coeff.detach() - gt_basis_coeff[:, None, :]).pow(2).mean(dim=-1)
    aligned_idx = coeff_distance.masked_fill(~match_mode, float("inf")).argmin(dim=1)
    aligned_coeff = _gather_candidates(coeff, aligned_idx)
    return F.smooth_l1_loss(aligned_coeff[hit_mask], gt_basis_coeff[hit_mask])


def _anchor_reconstruction_loss(
    coarse_local,
    endpoint_mode_local,
    gt_local,
    basis_pinv,
    basis_matrix,
    top_proto_idx,
    gt_proto_id,
    mode,
):
    if (
        mode == "none"
        or coarse_local is None
        or endpoint_mode_local is None
        or basis_pinv is None
        or basis_matrix is None
    ):
        return gt_local.sum() * 0.0

    if top_proto_idx is None or top_proto_idx.size(1) != coarse_local.size(1) or endpoint_mode_local.shape[:2] != coarse_local.shape[:2]:
        return coarse_local.sum() * 0.0

    alpha = torch.linspace(0.0, 1.0, steps=gt_local.size(1), device=gt_local.device, dtype=gt_local.dtype)
    anchor = alpha[None, None, :, None] * endpoint_mode_local[:, :, None, :]
    residual = (gt_local[:, None, :, :] - anchor).reshape(-1, gt_local.size(1) * gt_local.size(2))
    target_coeff = residual @ basis_pinv.to(device=gt_local.device, dtype=gt_local.dtype)
    target_recon = target_coeff @ basis_matrix.to(device=gt_local.device, dtype=gt_local.dtype)
    target_path = anchor + target_recon.reshape_as(coarse_local)

    if mode == "all":
        return F.smooth_l1_loss(coarse_local, target_path.detach())
    if mode == "gt_proto":
        match_mode = top_proto_idx.eq(gt_proto_id[:, None])
        if not match_mode.any():
            return coarse_local.sum() * 0.0
        return F.smooth_l1_loss(coarse_local[match_mode], target_path.detach()[match_mode])
    raise ValueError(f"Unsupported anchor_recon_supervision: {mode}")


def _projection_guided_losses(
    coeff,
    coarse_local,
    endpoint_mode_local,
    gt_local,
    basis_pinv,
    basis_matrix,
    candidate_proto_idx,
    gt_proto_id,
    best_idx,
    mode,
):
    if (
        mode == "none"
        or coeff is None
        or coarse_local is None
        or endpoint_mode_local is None
        or basis_pinv is None
        or basis_matrix is None
    ):
        zero = gt_local.sum() * 0.0
        return zero, zero, gt_local.new_tensor(0.0)

    if endpoint_mode_local.shape[:2] != coeff.shape[:2] or coarse_local.shape[:2] != coeff.shape[:2]:
        zero = coeff.sum() * 0.0
        return zero, zero, coeff.new_tensor(0.0)

    batch_size, num_modes, basis_dim = coeff.shape
    select_mask = torch.zeros(batch_size, num_modes, dtype=torch.bool, device=coeff.device)
    if mode in {"winner", "winner_gt_proto"}:
        select_mask.scatter_(1, best_idx[:, None], True)
    if mode in {"gt_proto", "winner_gt_proto"}:
        if candidate_proto_idx is None or candidate_proto_idx.shape[:2] != coeff.shape[:2]:
            if mode == "gt_proto":
                zero = coeff.sum() * 0.0
                return zero, zero, coeff.new_tensor(0.0)
        else:
            select_mask = select_mask | candidate_proto_idx.eq(gt_proto_id[:, None])
    if mode == "all":
        select_mask.fill_(True)
    if mode not in {"winner", "gt_proto", "winner_gt_proto", "all"}:
        raise ValueError(f"Unsupported projection_supervision: {mode}")
    if not select_mask.any():
        zero = coeff.sum() * 0.0
        return zero, zero, coeff.new_tensor(0.0)

    alpha = torch.linspace(0.0, 1.0, steps=gt_local.size(1), device=gt_local.device, dtype=gt_local.dtype)
    anchor = alpha[None, None, :, None] * endpoint_mode_local[:, :, None, :]
    residual = (gt_local[:, None, :, :] - anchor).reshape(-1, gt_local.size(1) * gt_local.size(2))
    target_coeff = residual @ basis_pinv.to(device=gt_local.device, dtype=gt_local.dtype)
    target_coeff = target_coeff.reshape(batch_size, num_modes, basis_dim)
    target_recon = target_coeff.reshape(-1, basis_dim) @ basis_matrix.to(device=gt_local.device, dtype=gt_local.dtype)
    target_path = anchor + target_recon.reshape_as(coarse_local)

    coeff_loss = F.smooth_l1_loss(coeff[select_mask], target_coeff.detach()[select_mask])
    path_loss = F.smooth_l1_loss(coarse_local[select_mask], target_path.detach()[select_mask])
    return coeff_loss, path_loss, select_mask.float().mean()


def _basis_bridge_guided_losses(
    bridge_coeff,
    bridge_local,
    endpoint_mode_local,
    gt_local,
    basis_pinv,
    basis_matrix,
    candidate_proto_idx,
    gt_proto_id,
    best_idx,
    mode,
):
    if (
        mode == "none"
        or bridge_coeff is None
        or bridge_local is None
        or endpoint_mode_local is None
        or basis_pinv is None
        or basis_matrix is None
    ):
        zero = gt_local.sum() * 0.0
        return zero, zero, gt_local.new_tensor(0.0)

    if endpoint_mode_local.shape[:2] != bridge_coeff.shape[:2] or bridge_local.shape[:2] != bridge_coeff.shape[:2]:
        zero = bridge_coeff.sum() * 0.0
        return zero, zero, bridge_coeff.new_tensor(0.0)

    batch_size, num_modes, basis_dim = bridge_coeff.shape
    select_mask = torch.zeros(batch_size, num_modes, dtype=torch.bool, device=bridge_coeff.device)
    if mode in {"winner", "winner_gt_proto"}:
        select_mask.scatter_(1, best_idx[:, None], True)
    if mode in {"gt_proto", "winner_gt_proto"}:
        if candidate_proto_idx is None or candidate_proto_idx.shape[:2] != bridge_coeff.shape[:2]:
            if mode == "gt_proto":
                zero = bridge_coeff.sum() * 0.0
                return zero, zero, bridge_coeff.new_tensor(0.0)
        else:
            select_mask = select_mask | candidate_proto_idx.eq(gt_proto_id[:, None])
    if mode == "all":
        select_mask.fill_(True)
    if mode not in {"winner", "gt_proto", "winner_gt_proto", "all"}:
        raise ValueError(f"Unsupported basis_bridge_supervision: {mode}")
    if not select_mask.any():
        zero = bridge_coeff.sum() * 0.0
        return zero, zero, bridge_coeff.new_tensor(0.0)

    alpha = torch.linspace(0.0, 1.0, steps=gt_local.size(1), device=gt_local.device, dtype=gt_local.dtype)
    anchor = alpha[None, None, :, None] * endpoint_mode_local[:, :, None, :]
    residual = (gt_local[:, None, :, :] - anchor).reshape(-1, gt_local.size(1) * gt_local.size(2))
    target_coeff = residual @ basis_pinv.to(device=gt_local.device, dtype=gt_local.dtype)
    target_coeff = target_coeff.reshape(batch_size, num_modes, basis_dim)
    target_recon = target_coeff.reshape(-1, basis_dim) @ basis_matrix.to(device=gt_local.device, dtype=gt_local.dtype)
    target_path = anchor + target_recon.reshape_as(bridge_local)

    coeff_loss = F.smooth_l1_loss(bridge_coeff[select_mask], target_coeff.detach()[select_mask])
    path_loss = F.smooth_l1_loss(bridge_local[select_mask], target_path.detach()[select_mask])
    return coeff_loss, path_loss, select_mask.float().mean()


class ProtoBasisLoss(nn.Module):
    def __init__(
        self,
        lambda_xyz=1.0,
        lambda_fde=0.8,
        lambda_proto=0.4,
        lambda_res=0.2,
        lambda_score=0.5,
        lambda_div=0.05,
        lambda_coeff=0.02,
        lambda_smooth=0.10,
        lambda_gt_proto_shape=0.0,
        lambda_gt_proto_fde=0.0,
        lambda_gt_proto_coeff=0.0,
        score_hard_mix=0.25,
        score_fde_weight=0.75,
        score_soft_temperature=0.35,
        proto_focal_gamma=0.0,
        proto_freq_weight_power=0.0,
        proto_freq_weight_max=5.0,
        endpoint_residual_supervision="all",
        lambda_anchor_recon=0.0,
        anchor_recon_supervision="none",
        lambda_projection_coeff=0.0,
        lambda_projection_path=0.0,
        projection_supervision="none",
        lambda_bridge_coeff=0.0,
        lambda_bridge_path=0.0,
        basis_bridge_supervision="none",
    ):
        super().__init__()
        if endpoint_residual_supervision not in {"all", "hit_only"}:
            raise ValueError(f"Unsupported endpoint_residual_supervision: {endpoint_residual_supervision}")
        if anchor_recon_supervision not in {"none", "gt_proto", "all"}:
            raise ValueError(f"Unsupported anchor_recon_supervision: {anchor_recon_supervision}")
        if projection_supervision not in {"none", "winner", "gt_proto", "winner_gt_proto", "all"}:
            raise ValueError(f"Unsupported projection_supervision: {projection_supervision}")
        if basis_bridge_supervision not in {"none", "winner", "gt_proto", "winner_gt_proto", "all"}:
            raise ValueError(f"Unsupported basis_bridge_supervision: {basis_bridge_supervision}")
        self.lambda_xyz = lambda_xyz
        self.lambda_fde = lambda_fde
        self.lambda_proto = lambda_proto
        self.lambda_res = lambda_res
        self.lambda_score = lambda_score
        self.lambda_div = lambda_div
        self.lambda_coeff = lambda_coeff
        self.lambda_smooth = lambda_smooth
        self.lambda_gt_proto_shape = lambda_gt_proto_shape
        self.lambda_gt_proto_fde = lambda_gt_proto_fde
        self.lambda_gt_proto_coeff = lambda_gt_proto_coeff
        self.score_hard_mix = min(max(score_hard_mix, 0.0), 1.0)
        self.score_fde_weight = score_fde_weight
        self.score_soft_temperature = score_soft_temperature
        self.proto_focal_gamma = proto_focal_gamma
        self.proto_freq_weight_power = proto_freq_weight_power
        self.proto_freq_weight_max = proto_freq_weight_max
        self.endpoint_residual_supervision = endpoint_residual_supervision
        self.lambda_anchor_recon = lambda_anchor_recon
        self.anchor_recon_supervision = anchor_recon_supervision
        self.lambda_projection_coeff = lambda_projection_coeff
        self.lambda_projection_path = lambda_projection_path
        self.projection_supervision = projection_supervision
        self.lambda_bridge_coeff = lambda_bridge_coeff
        self.lambda_bridge_path = lambda_bridge_path
        self.basis_bridge_supervision = basis_bridge_supervision

    def forward(self, outputs, batch, stage_cfg):
        pred_xyz = outputs["pred_xyz"]
        pred_score = outputs["pred_score"]
        proto_logits = outputs["proto_logits"]
        top_proto_idx = outputs["top_proto_idx"]
        aux = outputs["aux"]
        candidate_proto_idx = aux.get("candidate_proto_idx")
        if candidate_proto_idx is None:
            topk_proto = top_proto_idx.size(1)
            if topk_proto > 0 and pred_xyz.size(1) % topk_proto == 0:
                candidate_proto_idx = top_proto_idx.repeat_interleave(pred_xyz.size(1) // topk_proto, dim=1)
            else:
                candidate_proto_idx = None
        gt_xyz = batch["fut_xyz"]
        gt_proto_id = batch["gt_proto_id"]
        gt_proto_residual = batch["gt_proto_residual"]
        gt_basis_coeff = batch.get("gt_basis_coeff")

        best_idx, ade, fde = _winner_indices(pred_xyz, gt_xyz)
        winner_xyz = _gather_candidates(pred_xyz, best_idx)
        coeff_for_regularization = aux.get("coeff_delta", aux["coeff"])
        winner_coeff = _gather_candidates(coeff_for_regularization, best_idx)

        xyz_loss = F.smooth_l1_loss(winner_xyz, gt_xyz)
        fde_loss = F.smooth_l1_loss(winner_xyz[:, -1], gt_xyz[:, -1])
        proto_loss = _prototype_loss(
            proto_logits,
            gt_proto_id,
            proto_frequency=aux.get("proto_frequency"),
            focal_gamma=self.proto_focal_gamma,
            freq_weight_power=self.proto_freq_weight_power,
            freq_weight_max=self.proto_freq_weight_max,
        )

        match_mask = top_proto_idx.eq(gt_proto_id.unsqueeze(1))
        hit_mask = match_mask.any(dim=1)
        gt_slot = match_mask.float().argmax(dim=1)
        pred_residual = aux["endpoint_residual"][torch.arange(gt_slot.size(0), device=gt_slot.device), gt_slot]
        if self.endpoint_residual_supervision == "hit_only" and hit_mask.any():
            res_loss = F.smooth_l1_loss(pred_residual[hit_mask], gt_proto_residual[hit_mask])
        elif self.endpoint_residual_supervision == "hit_only":
            res_loss = aux["endpoint_residual"].sum() * 0.0
        else:
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
        div_loss = _diversity_repulsion(pred_xyz)
        coeff_loss = winner_coeff.pow(2).mean()
        smooth_loss = _trajectory_smoothness(winner_xyz)
        gt_proto_shape_loss, gt_proto_fde_loss, gt_proto_hit_rate = _gt_proto_aligned_losses(
            pred_xyz,
            gt_xyz,
            candidate_proto_idx,
            gt_proto_id,
            ade,
        )
        gt_proto_coeff_loss = _gt_proto_coeff_loss(
            aux["coeff"],
            gt_basis_coeff,
            candidate_proto_idx,
            gt_proto_id,
        )
        anchor_recon_loss = _anchor_reconstruction_loss(
            aux.get("coarse_local"),
            aux.get("endpoint_mode_local"),
            batch["fut_local"],
            aux.get("basis_pinv"),
            aux.get("basis_matrix"),
            candidate_proto_idx,
            gt_proto_id,
            self.anchor_recon_supervision,
        )
        projection_coeff_loss, projection_path_loss, projection_rate = _projection_guided_losses(
            aux["coeff"],
            aux.get("coarse_local"),
            aux.get("endpoint_mode_local"),
            batch["fut_local"],
            aux.get("basis_pinv"),
            aux.get("basis_matrix"),
            candidate_proto_idx,
            gt_proto_id,
            best_idx,
            self.projection_supervision,
        )
        bridge_coeff_loss, bridge_path_loss, bridge_rate = _basis_bridge_guided_losses(
            aux.get("bridge_coeff"),
            aux.get("bridge_local"),
            aux.get("endpoint_mode_local"),
            batch["fut_local"],
            aux.get("basis_pinv"),
            aux.get("basis_matrix"),
            candidate_proto_idx,
            gt_proto_id,
            best_idx,
            self.basis_bridge_supervision,
        )

        total = self.lambda_xyz * xyz_loss
        total = total + self.lambda_fde * fde_loss
        total = total + self.lambda_proto * proto_loss
        total = total + self.lambda_res * res_loss
        total = total + self.lambda_score * score_loss
        total = total + stage_cfg["div_weight"] * self.lambda_div * div_loss
        total = total + self.lambda_coeff * coeff_loss
        total = total + self.lambda_smooth * smooth_loss
        total = total + self.lambda_gt_proto_shape * gt_proto_shape_loss
        total = total + self.lambda_gt_proto_fde * gt_proto_fde_loss
        total = total + self.lambda_gt_proto_coeff * gt_proto_coeff_loss
        total = total + self.lambda_anchor_recon * anchor_recon_loss
        total = total + self.lambda_projection_coeff * projection_coeff_loss
        total = total + self.lambda_projection_path * projection_path_loss
        total = total + self.lambda_bridge_coeff * bridge_coeff_loss
        total = total + self.lambda_bridge_path * bridge_path_loss

        stats = {
            "xyz": float(xyz_loss.detach().item()),
            "fde": float(fde_loss.detach().item()),
            "proto": float(proto_loss.detach().item()),
            "res": float(res_loss.detach().item()),
            "score": float(score_loss.detach().item()),
            "div": float(div_loss.detach().item()),
            "coeff": float(coeff_loss.detach().item()),
            "smooth": float(smooth_loss.detach().item()),
            "gt_proto_shape": float(gt_proto_shape_loss.detach().item()),
            "gt_proto_fde": float(gt_proto_fde_loss.detach().item()),
            "gt_proto_coeff": float(gt_proto_coeff_loss.detach().item()),
            "anchor_recon": float(anchor_recon_loss.detach().item()),
            "projection_coeff": float(projection_coeff_loss.detach().item()),
            "projection_path": float(projection_path_loss.detach().item()),
            "projection_rate": float(projection_rate.detach().item()),
            "bridge_coeff": float(bridge_coeff_loss.detach().item()),
            "bridge_path": float(bridge_path_loss.detach().item()),
            "bridge_rate": float(bridge_rate.detach().item()),
            "gt_proto_hit_rate": float(gt_proto_hit_rate.detach().item()),
            "winner_ade": float(ade.min(dim=1).values.mean().detach().item()),
            "res_hit_rate": float(hit_mask.float().mean().detach().item()),
            "res_miss_rate": float((~hit_mask).float().mean().detach().item()),
        }
        return total, stats
