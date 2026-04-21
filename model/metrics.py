import math

import torch


def metric_names_for_protocol(primary_k=5, secondary_k=20):
    names = [
        f"ADE@{primary_k}",
        f"FDE@{primary_k}",
        f"GLeV@{primary_k}",
    ]
    if secondary_k != primary_k:
        names.extend(
            [
                f"ADE@{secondary_k}",
                f"FDE@{secondary_k}",
                f"GLeV@{secondary_k}",
            ]
        )
    rare_k = secondary_k if secondary_k != primary_k else primary_k
    names.extend(
        [
            f"rare_FDE@{rare_k}",
            "Top1_ADE",
            "Top1_FDE",
            "proto_top1_acc",
            "proto_rare_recall",
            "score_entropy",
            "endpoint_var",
        ]
    )
    return tuple(names)


METRIC_NAMES = metric_names_for_protocol()


def _topk_predictions(pred_xyz, pred_score, k):
    safe_k = min(k, pred_score.size(1))
    top_idx = pred_score.argsort(dim=-1, descending=True)[:, :safe_k]
    gather_idx = top_idx[:, :, None, None].expand(-1, -1, pred_xyz.size(2), pred_xyz.size(3))
    return torch.gather(pred_xyz, 1, gather_idx)


def minade_minfde(pred_xyz, gt_xyz, pred_score, k):
    topk = _topk_predictions(pred_xyz, pred_score, k)
    l2 = torch.linalg.norm(topk - gt_xyz[:, None], dim=-1)
    ade = l2.mean(dim=-1)
    fde = l2[..., -1]
    return ade.min(dim=1).values, fde.min(dim=1).values


def _glev_raw(pred_xyz, gt_xyz, pred_score, k, topn):
    topk = _topk_predictions(pred_xyz, pred_score, k)
    endpoints = topk[:, :, -1]
    gt_end = gt_xyz[:, -1][:, None]
    distance = torch.linalg.norm(endpoints - gt_end, dim=-1)
    safe_topn = min(topn, distance.size(1))
    near_idx = distance.topk(safe_topn, largest=False).indices
    near = torch.gather(endpoints, 1, near_idx[..., None].expand(-1, -1, 3))
    global_var = endpoints.var(dim=1, unbiased=False).sum(dim=-1) + 1e-6
    local_var = near.var(dim=1, unbiased=False).sum(dim=-1) + 1e-6
    # Keep a single GooDFlight-style lower-is-better GLeV value in the public API.
    return local_var / global_var
def rare_subset_fde(pred_xyz, gt_xyz, pred_score, is_rare, k):
    if is_rare.sum() == 0:
        return pred_xyz.new_full((pred_xyz.size(0),), float("nan"))
    _, fde = minade_minfde(pred_xyz, gt_xyz, pred_score, k)
    return fde.masked_fill(~is_rare, float("nan"))


def top1_metrics(pred_xyz, gt_xyz, pred_score):
    top1_idx = pred_score.argmax(dim=-1)
    gather_idx = top1_idx[:, None, None, None].expand(-1, 1, pred_xyz.size(2), pred_xyz.size(3))
    top1 = torch.gather(pred_xyz, 1, gather_idx).squeeze(1)
    l2 = torch.linalg.norm(top1 - gt_xyz, dim=-1)
    return l2.mean(dim=-1), l2[:, -1]


def init_metric_sums(metric_names=METRIC_NAMES):
    return {name: 0.0 for name in metric_names}


def update_metric_sums(metric_sums, batch_metrics, count):
    for name, value in batch_metrics.items():
        if math.isnan(value):
            continue
        metric_sums[name] += value * count


def average_metric_sums(metric_sums, count, rare_count):
    averaged = {}
    for name, value in metric_sums.items():
        divisor = rare_count if name.startswith("rare_FDE@") else count
        averaged[name] = value / max(divisor, 1)
    return averaged


def summarize_batch_metrics(
    outputs,
    batch,
    primary_k=5,
    secondary_k=20,
    glev_topn_primary=2,
    glev_topn_secondary=5,
):
    pred_xyz = outputs["pred_xyz"]
    pred_score = outputs["pred_score"]
    gt_xyz = batch["fut_xyz"]

    ade_primary, fde_primary = minade_minfde(pred_xyz, gt_xyz, pred_score, k=primary_k)
    glev_primary = _glev_raw(pred_xyz, gt_xyz, pred_score, k=primary_k, topn=glev_topn_primary)

    metrics = {
        f"ADE@{primary_k}": float(ade_primary.mean().item()),
        f"FDE@{primary_k}": float(fde_primary.mean().item()),
        f"GLeV@{primary_k}": float(glev_primary.mean().item()),
    }

    rare_k = secondary_k if secondary_k != primary_k else primary_k
    if secondary_k != primary_k:
        ade_secondary, fde_secondary = minade_minfde(pred_xyz, gt_xyz, pred_score, k=secondary_k)
        glev_secondary = _glev_raw(
            pred_xyz,
            gt_xyz,
            pred_score,
            k=secondary_k,
            topn=glev_topn_secondary,
        )
        metrics.update(
            {
                f"ADE@{secondary_k}": float(ade_secondary.mean().item()),
                f"FDE@{secondary_k}": float(fde_secondary.mean().item()),
                f"GLeV@{secondary_k}": float(glev_secondary.mean().item()),
            }
        )

    top1_ade, top1_fde = top1_metrics(pred_xyz, gt_xyz, pred_score)
    proto_top1 = outputs["proto_logits"].argmax(dim=-1).eq(batch["gt_proto_id"]).float()
    rare_mask = batch["is_rare"]
    rare_hits = proto_top1[rare_mask]

    score_prob = pred_score.softmax(dim=-1)
    score_entropy = -(score_prob * score_prob.clamp_min(1e-8).log()).sum(dim=-1)
    endpoint_var = pred_xyz[:, :, -1].var(dim=1, unbiased=False).sum(dim=-1)
    rare_fde = rare_subset_fde(pred_xyz, gt_xyz, pred_score, rare_mask, k=rare_k)

    metrics.update(
        {
            f"rare_FDE@{rare_k}": float(torch.nanmean(rare_fde).item()) if rare_mask.any() else float("nan"),
            "Top1_ADE": float(top1_ade.mean().item()),
            "Top1_FDE": float(top1_fde.mean().item()),
            "proto_top1_acc": float(proto_top1.mean().item()),
            "proto_rare_recall": float(rare_hits.mean().item()) if rare_hits.numel() > 0 else float("nan"),
            "score_entropy": float(score_entropy.mean().item()),
            "endpoint_var": float(endpoint_var.mean().item()),
        }
    )
    return metrics, int(gt_xyz.size(0)), int(rare_mask.sum().item())
