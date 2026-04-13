import torch


METRIC_NAMES = ("ADE", "FDE", "MDE", "z-ADE", "z-FDE", "z-MDE")


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
        metrics["z-ADE"].append(scene_altitude.mean())
        metrics["z-FDE"].append(scene_altitude[-1].mean())
        metrics["z-MDE"].append(scene_altitude.max(dim=0).values.mean())
    return {name: torch.stack(values) for name, values in metrics.items()}


def metric_totals(prediction, target, scene_slices=None, scene_mask=None):
    distance = _pairwise_distance(prediction, target)
    altitude_error = torch.abs(prediction[..., 2] - target[..., 2])
    resolved_scene_slices = _scene_slices_or_full(scene_slices, distance.size(1), distance.device)
    scene_metrics = _scene_metric_means(distance, altitude_error, resolved_scene_slices)

    if scene_mask is not None:
        if scene_mask.dtype != torch.bool:
            scene_mask = scene_mask.bool()
        if scene_mask.numel() != resolved_scene_slices.size(0):
            raise ValueError("scene_mask must match the number of scenes in scene_slices")
        scene_metrics = {name: values[scene_mask] for name, values in scene_metrics.items()}

    scene_count = int(next(iter(scene_metrics.values())).numel()) if scene_metrics else 0
    if scene_count == 0:
        return {name: 0.0 for name in METRIC_NAMES}, 0

    totals = {name: values.sum().item() for name, values in scene_metrics.items()}
    return totals, scene_count


def init_metric_sums():
    return {name: 0.0 for name in METRIC_NAMES}


def update_metric_sums(metric_sums, batch_metrics, weight=1.0):
    for name, value in batch_metrics.items():
        metric_sums[name] += value * weight


def average_metric_sums(metric_sums, count):
    divisor = max(count, 1)
    return {name: value / divisor for name, value in metric_sums.items()}


def select_best_of_n_prediction(model, obs, target, context, scene_ids, scene_slices, n_samples):
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
            latent_mode="sample",
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
