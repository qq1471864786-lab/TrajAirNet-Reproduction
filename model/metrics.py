import torch


METRIC_NAMES = ("ADE", "FDE", "MDE", "AADE", "AFDE", "AMDE")


def _pairwise_distance(prediction, target):
    return torch.linalg.norm(prediction - target, dim=-1)


def _per_agent_metrics(distance, altitude_error):
    return {
        "ADE": distance.mean(dim=0),
        "FDE": distance[-1],
        "MDE": distance.max(dim=0).values,
        "AADE": altitude_error.mean(dim=0),
        "AFDE": altitude_error[-1],
        "AMDE": altitude_error.max(dim=0).values,
    }


def metric_totals(prediction, target, scene_slices=None):
    distance = _pairwise_distance(prediction, target)
    altitude_error = torch.abs(prediction[..., 2] - target[..., 2])
    per_agent_metrics = _per_agent_metrics(distance, altitude_error)
    agent_count = int(distance.size(1))
    totals = {name: values.sum().item() for name, values in per_agent_metrics.items()}
    return totals, agent_count


def evaluate_trajectory_batch(prediction, target, scene_slices=None):
    """Compute average metrics across agents in one batch."""
    totals, agent_count = metric_totals(prediction, target, scene_slices)
    return average_metric_sums(totals, agent_count)


def select_best_of_n_prediction(model, obs, target, context, scene_ids, scene_slices, n_samples):
    """
    Draw N samples and keep the best sample for each agent independently.
    This avoids batch-level best-of-N bias when a batch contains many scenes.
    """
    best_prediction = None
    best_agent_ade = None
    sample_mean_ades = []

    for _ in range(n_samples):
        prediction = model(
            obs,
            context,
            target=None,
            scene_ids=scene_ids,
            scene_slices=scene_slices,
        )
        per_agent_ade = _pairwise_distance(prediction, target).mean(dim=0)
        sample_mean_ades.append(per_agent_ade.mean().item())

        if best_prediction is None:
            best_prediction = prediction.clone()
            best_agent_ade = per_agent_ade
            continue

        better_mask = per_agent_ade < best_agent_ade
        if better_mask.any():
            best_prediction[:, better_mask, :] = prediction[:, better_mask, :]
            best_agent_ade = torch.minimum(best_agent_ade, per_agent_ade)

    diversity = 0.0
    if len(sample_mean_ades) > 1:
        diversity = float(torch.tensor(sample_mean_ades, dtype=torch.float32).std(unbiased=False).item())
    return best_prediction, diversity


def init_metric_sums():
    return {name: 0.0 for name in METRIC_NAMES}


def update_metric_sums(metric_sums, batch_metrics, weight=1.0):
    for name, value in batch_metrics.items():
        metric_sums[name] += value * weight


def average_metric_sums(metric_sums, count):
    return {name: value / max(count, 1) for name, value in metric_sums.items()}
