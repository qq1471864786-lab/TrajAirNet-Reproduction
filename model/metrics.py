import math

import torch


METRIC_NAMES = (
    "ADE@5",
    "FDE@5",
    "ADE@20",
    "FDE@20",
    "GLeV_report@5",
    "GLeV_report@20",
    "GLeV_raw@5",
    "GLeV_raw@20",
    "rare_FDE@20",
    "Top1_ADE",
    "Top1_FDE",
    "proto_top1_acc",
    "proto_rare_recall",
    "score_entropy",
    "endpoint_var",
)



def _topk_predictions(pred_xyz, pred_score, k):
    top_idx = pred_score.argsort(dim=-1, descending=True)[:, :k]
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
    near_idx = distance.topk(topn, largest=False).indices
    near = torch.gather(endpoints, 1, near_idx[..., None].expand(-1, -1, 3))
    global_var = endpoints.var(dim=1, unbiased=False).sum(dim=-1) + 1e-6
    local_var = near.var(dim=1, unbiased=False).sum(dim=-1) + 1e-6
    return global_var / local_var



def _glev_report(pred_xyz, gt_xyz, pred_score, k, topn):
    raw = _glev_raw(pred_xyz, gt_xyz, pred_score, k, topn)
    return 1.0 / raw



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



def init_metric_sums():
    return {name: 0.0 for name in METRIC_NAMES}



def update_metric_sums(metric_sums, batch_metrics, count):
    for name, value in batch_metrics.items():
        if math.isnan(value):
            continue
        metric_sums[name] += value * count



def average_metric_sums(metric_sums, count, rare_count):
    averaged = {}
    for name, value in metric_sums.items():
        divisor = rare_count if name == "rare_FDE@20" else count
        averaged[name] = value / max(divisor, 1)
    return averaged



def summarize_batch_metrics(outputs, batch, topn5=2, topn20=5):
    pred_xyz = outputs["pred_xyz"]
    pred_score = outputs["pred_score"]
    gt_xyz = batch["fut_xyz"]

    ade5, fde5 = minade_minfde(pred_xyz, gt_xyz, pred_score, k=5)
    ade20, fde20 = minade_minfde(pred_xyz, gt_xyz, pred_score, k=20)
    glev_report5 = _glev_report(pred_xyz, gt_xyz, pred_score, k=5, topn=topn5)
    glev_report20 = _glev_report(pred_xyz, gt_xyz, pred_score, k=20, topn=topn20)
    glev_raw5 = _glev_raw(pred_xyz, gt_xyz, pred_score, k=5, topn=topn5)
    glev_raw20 = _glev_raw(pred_xyz, gt_xyz, pred_score, k=20, topn=topn20)
    top1_ade, top1_fde = top1_metrics(pred_xyz, gt_xyz, pred_score)

    proto_top1 = outputs["proto_logits"].argmax(dim=-1).eq(batch["gt_proto_id"]).float()
    rare_mask = batch["is_rare"]
    rare_hits = proto_top1[rare_mask]

    score_prob = pred_score.softmax(dim=-1)
    score_entropy = -(score_prob * score_prob.clamp_min(1e-8).log()).sum(dim=-1)
    endpoint_var = outputs["pred_xyz"][:, :, -1].var(dim=1, unbiased=False).sum(dim=-1)
    rare_fde20 = rare_subset_fde(pred_xyz, gt_xyz, pred_score, rare_mask, k=20)

    metrics = {
        "ADE@5": float(ade5.mean().item()),
        "FDE@5": float(fde5.mean().item()),
        "ADE@20": float(ade20.mean().item()),
        "FDE@20": float(fde20.mean().item()),
        "GLeV_report@5": float(glev_report5.mean().item()),
        "GLeV_report@20": float(glev_report20.mean().item()),
        "GLeV_raw@5": float(glev_raw5.mean().item()),
        "GLeV_raw@20": float(glev_raw20.mean().item()),
        "rare_FDE@20": float(torch.nanmean(rare_fde20).item()) if rare_mask.any() else float("nan"),
        "Top1_ADE": float(top1_ade.mean().item()),
        "Top1_FDE": float(top1_fde.mean().item()),
        "proto_top1_acc": float(proto_top1.mean().item()),
        "proto_rare_recall": float(rare_hits.mean().item()) if rare_hits.numel() > 0 else float("nan"),
        "score_entropy": float(score_entropy.mean().item()),
        "endpoint_var": float(endpoint_var.mean().item()),
    }
    return metrics, int(gt_xyz.size(0)), int(rare_mask.sum().item())
