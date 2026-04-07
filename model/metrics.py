import torch


METRIC_NAMES = ("ADE", "FDE", "MDE", "AADE", "AFDE", "AMDE")


def _pairwise_distance(prediction, target):
    return torch.linalg.norm(prediction - target, dim=-1)


def _scene_slices_or_full(scene_slices, agent_count, device):
    if scene_slices is None:
        return torch.tensor([[0, agent_count]], dtype=torch.long, device=device)
    return scene_slices


def _scene_metric_means(distance, altitude_error, scene_slices):
    metrics = {name: [] for name in METRIC_NAMES}
    for start, end in scene_slices.tolist():
        scene_distance = distance[:, start:end]
        scene_altitude = altitude_error[:, start:end]
        metrics["ADE"].append(scene_distance.mean())
        metrics["FDE"].append(scene_distance[-1].mean())
        metrics["MDE"].append(scene_distance.max(dim=0).values.mean())
        metrics["AADE"].append(scene_altitude.mean())
        metrics["AFDE"].append(scene_altitude[-1].mean())
        metrics["AMDE"].append(scene_altitude.max(dim=0).values.mean())
    return {name: torch.stack(values) for name, values in metrics.items()}


def _scene_metric_totals(distance, altitude_error, scene_slices):
    scene_metrics = _scene_metric_means(distance, altitude_error, scene_slices)
    return {
        name: values.sum().item() for name, values in scene_metrics.items()
    }


def metric_totals(prediction, target, scene_slices=None):
    distance = _pairwise_distance(prediction, target)
    altitude_error = torch.abs(prediction[..., 2] - target[..., 2])
    resolved_scene_slices = _scene_slices_or_full(scene_slices, distance.size(1), distance.device)
    totals = _scene_metric_totals(distance, altitude_error, resolved_scene_slices)
    return totals, int(resolved_scene_slices.size(0))


def evaluate_trajectory_batch(prediction, target, scene_slices=None):
    """Compute average metrics across agents in one batch."""
    totals, agent_count = metric_totals(prediction, target, scene_slices)
    return average_metric_sums(totals, agent_count)


def select_best_of_n_prediction(model, obs, target, context, scene_ids, scene_slices, n_samples):
    """
    Draw N samples and keep the best sample for each scene independently.
    This matches the baseline best-of-5 protocol at scene level.
    """
    best_prediction = None
    best_scene_ade = None
    sample_mean_ades = []
    resolved_scene_slices = _scene_slices_or_full(scene_slices, target.size(1), target.device)

    for _ in range(n_samples):
        prediction = model(
            obs,
            context,
            target=None,
            scene_ids=scene_ids,
            scene_slices=scene_slices,
        )
        distance = _pairwise_distance(prediction, target)
        altitude_error = torch.abs(prediction[..., 2] - target[..., 2])
        scene_ade = _scene_metric_means(distance, altitude_error, resolved_scene_slices)["ADE"]
        sample_mean_ades.append(scene_ade.mean().item())

        if best_prediction is None:
            best_prediction = prediction.clone()
            best_scene_ade = scene_ade
            continue

        better_mask = scene_ade < best_scene_ade
        if better_mask.any():
            for scene_index, is_better in enumerate(better_mask.tolist()):
                if not is_better:
                    continue
                start, end = resolved_scene_slices[scene_index].tolist()
                best_prediction[:, start:end, :] = prediction[:, start:end, :]
            best_scene_ade = torch.minimum(best_scene_ade, scene_ade)

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
