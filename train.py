import argparse
import math
import os
import random
import sys
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
)
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import (
    ProtoBasisLoss,
    ProtoBasisNet,
    ProtoBasisSceneDataset,
    average_metric_sums,
    basis_hash,
    git_commit,
    init_metric_sums,
    metric_names_for_protocol,
    model_artifact_hash,
    proto_basis_collate,
    protocol_hash,
    resolve_protocol,
    resolve_split_dir,
    summarize_batch_metrics,
    update_metric_sums,
)
from model.run_logging import RunRecorder

LOSS_STAT_KEYS = (
    "xyz",
    "fde",
    "proto",
    "res",
    "score",
    "div",
    "coeff",
    "smooth",
    "gt_proto_shape",
    "gt_proto_fde",
    "gt_proto_coeff",
    "anchor_recon",
    "projection_coeff",
    "projection_path",
    "projection_rate",
    "gt_proto_hit_rate",
    "winner_ade",
    "res_hit_rate",
    "res_miss_rate",
)

ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"


def build_parser():
    parser = argparse.ArgumentParser(description="Train ProtoBasis-Net")
    parser.add_argument("dataset", nargs="?", default="", help="Dataset name, e.g. 111_days or 7days1.")
    parser.add_argument("--dataset_variant", type=str, default="social", choices=["social"])
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--protocol_name", type=str, default="trajair_40to120_best20")
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--preds", type=int, default=0)
    parser.add_argument("--obs_stride", type=int, default=1)
    parser.add_argument("--pred_stride", type=int, default=1)
    parser.add_argument("--max_agents", type=int, default=7)
    parser.add_argument("--n_proto", type=int, default=None)
    parser.add_argument("--basis_dim", type=int, default=16)
    parser.add_argument("--local_basis_dim", type=int, default=None)
    parser.add_argument("--support_aware_local_basis", dest="support_aware_local_basis", action="store_true")
    parser.add_argument(
        "--no_support_aware_local_basis",
        dest="support_aware_local_basis",
        action="store_false",
        help="Disable support-aware shrinking for proto-local basis.",
    )
    parser.set_defaults(support_aware_local_basis=None)
    parser.add_argument("--two_stage_decoder", dest="two_stage_decoder", action="store_true")
    parser.add_argument("--no_two_stage_decoder", dest="two_stage_decoder", action="store_false")
    parser.add_argument("--no_two_stage_update_endpoint", action="store_true")
    parser.add_argument("--no_two_stage_update_coeff", action="store_true")
    parser.set_defaults(two_stage_decoder=None)
    parser.add_argument("--coupled_decoder", dest="coupled_decoder", action="store_true")
    parser.add_argument("--no_coupled_decoder", dest="coupled_decoder", action="store_false")
    parser.set_defaults(coupled_decoder=None)
    parser.add_argument("--coupled_decoder_iters", type=int, default=None)
    parser.add_argument("--endpoint_shape_refiner", dest="endpoint_shape_refiner", action="store_true")
    parser.add_argument("--no_endpoint_shape_refiner", dest="endpoint_shape_refiner", action="store_false")
    parser.set_defaults(endpoint_shape_refiner=None)
    parser.add_argument("--control_shape_refiner", dest="control_shape_refiner", action="store_true")
    parser.add_argument("--no_control_shape_refiner", dest="control_shape_refiner", action="store_false")
    parser.set_defaults(control_shape_refiner=None)
    parser.add_argument("--control_shape_points", type=int, default=None)
    parser.add_argument("--trajectory_control_refiner", dest="trajectory_control_refiner", action="store_true")
    parser.add_argument("--no_trajectory_control_refiner", dest="trajectory_control_refiner", action="store_false")
    parser.set_defaults(trajectory_control_refiner=None)
    parser.add_argument("--trajectory_control_points", type=int, default=None)
    parser.add_argument("--topk_proto", type=int, default=None)
    parser.add_argument("--micro_per_proto", type=int, default=None)
    parser.add_argument("--candidate_dense_topk", type=int, default=None)
    parser.add_argument("--micro_coeff_anchors", dest="micro_coeff_anchors", action="store_true")
    parser.add_argument("--no_micro_coeff_anchors", dest="micro_coeff_anchors", action="store_false")
    parser.set_defaults(micro_coeff_anchors=None)
    parser.add_argument(
        "--endpoint_conditioning",
        type=str,
        default="rank",
        choices=["rank", "proto"],
        help="Condition endpoint residuals by candidate rank or by the selected prototype token.",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--social_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=0)
    parser.add_argument("--phase_a_epochs", type=int, default=None)
    parser.add_argument("--phase_b_epochs", type=int, default=None)
    parser.add_argument("--phase_c_epochs", type=int, default=None)
    parser.add_argument("--extra_epochs", type=int, default=None, help="Extra refiner-stage epochs appended after phase C.")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Stop after this many epochs without a best20 update. Use 0 to disable.")
    parser.add_argument("--early_stop_min_epoch", type=int, default=None, help="Earliest epoch where early stopping may trigger.")
    parser.add_argument("--stage_a_lr", type=float, default=3e-4)
    parser.add_argument("--stage_b_lr", type=float, default=2e-4)
    parser.add_argument("--stage_c_lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=1.5e-5)
    parser.add_argument("--extra_lr", type=float, default=4e-5, help="Constant LR used for appended extra refiner epochs.")
    parser.add_argument("--stage_b_div_weight", type=float, default=0.0)
    parser.add_argument("--stage_c_div_weight", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--rare_threshold", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true", help="Resume from save_dir/dataset_name/seed*/last.pt.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP. Default is enabled on CUDA.")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU fallback when CUDA is unavailable.")

    parser.add_argument("--lambda_xyz", type=float, default=1.0)
    parser.add_argument("--lambda_fde", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=0.35)
    parser.add_argument("--lambda_res", type=float, default=0.2)
    parser.add_argument("--lambda_score", type=float, default=0.03)
    parser.add_argument("--lambda_div", type=float, default=0.05)
    parser.add_argument("--lambda_coeff", type=float, default=0.02)
    parser.add_argument("--lambda_smooth", type=float, default=0.10)
    parser.add_argument("--lambda_gt_proto_shape", type=float, default=None)
    parser.add_argument("--lambda_gt_proto_fde", type=float, default=None)
    parser.add_argument("--lambda_gt_proto_coeff", type=float, default=None)
    parser.add_argument("--lambda_anchor_recon", type=float, default=None)
    parser.add_argument(
        "--anchor_recon_supervision",
        type=str,
        default=None,
        choices=["none", "gt_proto", "all"],
        help="Coarse-path distillation to the LS reconstruction around each predicted endpoint anchor.",
    )
    parser.add_argument("--lambda_projection_coeff", type=float, default=None)
    parser.add_argument("--lambda_projection_path", type=float, default=None)
    parser.add_argument(
        "--projection_supervision",
        type=str,
        default=None,
        choices=["none", "winner", "gt_proto", "winner_gt_proto", "all"],
        help="Supervise selected candidates toward the LS coeff/path under each predicted endpoint anchor.",
    )
    parser.add_argument("--score_hard_mix", type=float, default=0.25)
    parser.add_argument("--score_fde_weight", type=float, default=0.75)
    parser.add_argument("--score_soft_temperature", type=float, default=0.35)
    parser.add_argument("--proto_focal_gamma", type=float, default=0.0)
    parser.add_argument("--proto_freq_weight_power", type=float, default=0.0)
    parser.add_argument("--proto_freq_weight_max", type=float, default=5.0)
    parser.add_argument(
        "--endpoint_residual_supervision",
        type=str,
        default="all",
        choices=["all", "hit_only"],
        help="Use legacy endpoint residual supervision for all samples or only samples whose GT prototype is in top-k.",
    )

    parser.add_argument("--disable_social", action="store_true")
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--disable_refiner", action="store_true")

    parser.add_argument("--save_dir", type=str, default="save_model")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Initialize model weights from a checkpoint.")
    parser.add_argument("--allow_partial_init", action="store_true", help="Allow missing/unexpected keys for init_checkpoint.")
    parser.add_argument(
        "--freeze_backbone_except_new_heads",
        action="store_true",
        help="Train only newly added heads after init_checkpoint.",
    )
    parser.add_argument(
        "--freeze_backbone_except_trajectory_control",
        action="store_true",
        help="Train only the trajectory-control refiner after init_checkpoint.",
    )
    parser.add_argument(
        "--freeze_backbone_except_coeff_correction",
        dest="freeze_backbone_except_new_heads",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument(
        "--artifact_max_samples",
        type=int,
        default=0,
        help="Cap train artifact fitting samples for smoke validation. 0 keeps full artifact fitting.",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_arg, allow_cpu=False):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. Use --allow_cpu only if you really want to run on CPU.")


def resolve_dataset_name(args):
    if args.dataset_name:
        return args.dataset_name
    if args.dataset:
        return args.dataset
    return "111_days"


def apply_protocol(args):
    spec = resolve_protocol(args.protocol_name)
    args.obs = spec.obs_steps
    args.preds = spec.pred_steps
    args.obs_stride = spec.obs_stride
    args.pred_stride = spec.pred_stride
    args.obs_horizon_sec = spec.obs_horizon_sec
    args.pred_horizon_sec = spec.pred_horizon_sec
    args.eval_topk_primary = spec.eval_topk_primary
    args.eval_topk_secondary = spec.eval_topk_secondary
    args.glev_topn_primary = spec.glev_topn_primary
    args.glev_topn_secondary = spec.glev_topn_secondary
    return spec


def apply_training_defaults(args):
    args.dataset_name = resolve_dataset_name(args)
    is_unified = args.protocol_name == "trajair_40to120_best20"
    is_main_dataset = args.dataset_name == "111_days"
    is_small_dataset = args.dataset_name.lower().startswith("7days")
    uses_validated_basis_profile = is_unified and (is_main_dataset or is_small_dataset)

    if args.n_proto is None:
        args.n_proto = 96 if is_unified and is_small_dataset else 64
    if args.topk_proto is None:
        args.topk_proto = 15 if is_unified and is_main_dataset else 5
    if args.micro_per_proto is None:
        args.micro_per_proto = 2 if is_unified and is_main_dataset else 4
    if args.candidate_dense_topk is None:
        args.candidate_dense_topk = 5 if is_unified and is_main_dataset else 0
    if args.d_model is None:
        args.d_model = 128 if is_unified and is_main_dataset else 96
    if args.ff_dim is None:
        args.ff_dim = 256 if is_unified and is_main_dataset else 192
    if args.encoder_layers is None:
        args.encoder_layers = 4 if is_unified and is_main_dataset else 3
    if args.local_basis_dim is None:
        args.local_basis_dim = 2 if uses_validated_basis_profile else 0
    if args.support_aware_local_basis is None:
        args.support_aware_local_basis = bool(uses_validated_basis_profile and args.local_basis_dim > 0)
    if args.two_stage_decoder is None:
        args.two_stage_decoder = bool(uses_validated_basis_profile)
    if args.coupled_decoder is None:
        args.coupled_decoder = bool(uses_validated_basis_profile)
    if args.coupled_decoder_iters is None:
        args.coupled_decoder_iters = 2 if args.coupled_decoder else 0
    if args.endpoint_shape_refiner is None:
        args.endpoint_shape_refiner = bool(uses_validated_basis_profile)
    if args.control_shape_refiner is None:
        args.control_shape_refiner = bool(uses_validated_basis_profile)
    if args.control_shape_points is None:
        args.control_shape_points = 32 if args.control_shape_refiner else 16
    if args.trajectory_control_refiner is None:
        args.trajectory_control_refiner = False
    if args.trajectory_control_points is None:
        args.trajectory_control_points = 32
    if args.micro_coeff_anchors is None:
        args.micro_coeff_anchors = True

    if args.batch_size <= 0:
        if is_unified and is_main_dataset:
            args.batch_size = 512
        elif is_unified:
            args.batch_size = 48
        else:
            args.batch_size = 48

    if args.eval_batch_size <= 0:
        if is_unified and is_main_dataset:
            args.eval_batch_size = 1024
        else:
            args.eval_batch_size = max(args.batch_size * 2, args.batch_size)

    if args.grad_accum <= 0:
        args.grad_accum = 1

    if args.phase_a_epochs is None:
        args.phase_a_epochs = 10
    if args.phase_b_epochs is None:
        args.phase_b_epochs = 4
    if args.phase_c_epochs is None:
        if is_unified and is_small_dataset:
            args.phase_c_epochs = 20
        else:
            args.phase_c_epochs = 51
    if args.extra_epochs is None:
        if is_unified and is_small_dataset:
            args.extra_epochs = 0
        elif is_unified and is_main_dataset:
            args.extra_epochs = 0
        else:
            args.extra_epochs = 35

    if args.early_stop_patience is None:
        args.early_stop_patience = 24 if is_unified and is_main_dataset else 0
    if args.early_stop_min_epoch is None:
        args.early_stop_min_epoch = 35 if is_unified and is_main_dataset else 0

    if args.lambda_gt_proto_shape is None:
        args.lambda_gt_proto_shape = 0.15 if is_unified and is_main_dataset else 0.0
    if args.lambda_gt_proto_fde is None:
        args.lambda_gt_proto_fde = 0.05 if is_unified and is_main_dataset else 0.0
    if args.lambda_gt_proto_coeff is None:
        args.lambda_gt_proto_coeff = 0.10 if is_unified and is_main_dataset else 0.0
    if args.lambda_anchor_recon is None:
        args.lambda_anchor_recon = 0.10 if is_unified and is_main_dataset else 0.0
    if args.anchor_recon_supervision is None:
        args.anchor_recon_supervision = "gt_proto" if is_unified and is_main_dataset else "none"
    use_projection_guidance = bool(is_unified and is_main_dataset and args.coupled_decoder)
    if args.lambda_projection_coeff is None:
        args.lambda_projection_coeff = 0.02 if use_projection_guidance else 0.0
    if args.lambda_projection_path is None:
        args.lambda_projection_path = 0.05 if use_projection_guidance else 0.0
    if args.projection_supervision is None:
        args.projection_supervision = "winner_gt_proto" if use_projection_guidance else "none"

    args.epochs = args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs + max(args.extra_epochs, 0)


def apply_runtime_defaults(args, device):
    if args.num_workers <= 0:
        if os.name == "nt":
            args.num_workers = 0
        else:
            cpu_count = os.cpu_count() or 4
            args.num_workers = min(8, max(2, cpu_count // 2))

    if device.type == "cuda":
        args.pin_memory = True

    if args.num_workers > 0:
        args.persistent_workers = True

def stage_config(epoch, args):
    if epoch <= args.phase_a_epochs:
        return {
            "name": "basis_warmup",
            "enable_refiner": False,
            "force_gt_proto": True,
            "div_weight": 0.0,
            "rare_weight": 1.0,
        }
    if epoch <= args.phase_a_epochs + args.phase_b_epochs:
        return {
            "name": "joint_no_refiner",
            "enable_refiner": False,
            "force_gt_proto": False,
            "div_weight": args.stage_b_div_weight,
            "rare_weight": 1.0,
        }
    stage_name = "joint_refiner"
    if epoch > args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs:
        stage_name = "joint_refiner_extra"
    return {
        "name": stage_name,
        "enable_refiner": True,
        "force_gt_proto": False,
        "div_weight": args.stage_c_div_weight,
        "rare_weight": 1.5,
    }


def set_epoch_lr(optimizer, epoch, args):
    if epoch <= args.phase_a_epochs:
        lr = args.stage_a_lr
    elif epoch <= args.phase_a_epochs + args.phase_b_epochs:
        phase_epoch = epoch - args.phase_a_epochs - 1
        denom = max(args.phase_b_epochs - 1, 1)
        progress = min(phase_epoch / denom, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.stage_c_lr + (args.stage_b_lr - args.stage_c_lr) * cosine
    elif epoch <= args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs:
        phase_epoch = epoch - args.phase_a_epochs - args.phase_b_epochs - 1
        denom = max(args.phase_c_epochs - 1, 1)
        progress = min(phase_epoch / denom, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.min_lr + (args.stage_c_lr - args.min_lr) * cosine
    else:
        lr = args.extra_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def build_train_loader(dataset, args, rare_weight):
    weights = dataset.sample_weights(rare_weight)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=int(len(weights)),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=args.max_agents),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def build_eval_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=args.max_agents),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def build_model(args, model_artifact):
    return ProtoBasisNet(
        obs_len=args.obs,
        pred_len=args.preds,
        d_model=args.d_model,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        encoder_layers=args.encoder_layers,
        social_layers=args.social_layers,
        topk_proto=args.topk_proto,
        n_micro=args.micro_per_proto,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        local_basis_dim=args.local_basis_dim,
        support_aware_local_basis=args.support_aware_local_basis,
        two_stage_decoder=bool(args.two_stage_decoder),
        two_stage_update_endpoint=not args.no_two_stage_update_endpoint,
        two_stage_update_coeff=not args.no_two_stage_update_coeff,
        dropout=args.dropout,
        proto_summary_5d=torch.tensor(model_artifact["summary_5d"], dtype=torch.float32),
        proto_frequency=torch.tensor(model_artifact["frequency"], dtype=torch.float32),
        basis_bank=torch.tensor(model_artifact["basis_bank"], dtype=torch.float32),
        prototype_mean_path=torch.tensor(model_artifact["prototype_mean_path"], dtype=torch.float32),
        local_basis_bank=torch.tensor(model_artifact["local_basis_bank"], dtype=torch.float32),
        micro_coeff_anchors=torch.tensor(
            model_artifact.get(
                "micro_coeff_anchors",
                np.zeros((args.n_proto, args.micro_per_proto, args.basis_dim), dtype=np.float32),
            ),
            dtype=torch.float32,
        ),
        use_micro_coeff_anchors=bool(args.micro_coeff_anchors),
        endpoint_conditioning=args.endpoint_conditioning,
        candidate_dense_topk=args.candidate_dense_topk,
        coupled_decoder=bool(args.coupled_decoder),
        coupled_decoder_iters=args.coupled_decoder_iters,
        endpoint_shape_refiner=bool(args.endpoint_shape_refiner),
        control_shape_refiner=bool(args.control_shape_refiner),
        control_shape_points=args.control_shape_points,
        trajectory_control_refiner=bool(args.trajectory_control_refiner),
        trajectory_control_points=args.trajectory_control_points,
        disable_social=args.disable_social,
        disable_router=args.disable_router,
        disable_refiner=args.disable_refiner,
    )


def initialize_from_checkpoint(model, checkpoint_path, device, allow_partial=False):
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    load_result = model.load_state_dict(state_dict, strict=not allow_partial)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if (missing or unexpected) and not allow_partial:
        raise RuntimeError(f"Checkpoint init mismatch: missing={missing}, unexpected={unexpected}")
    print(
        "[Init] loaded "
        f"{checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print("[Init] missing keys:", ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    if unexpected:
        print("[Init] unexpected keys:", ", ".join(unexpected[:12]) + (" ..." if len(unexpected) > 12 else ""))
    return checkpoint


def apply_freeze_policy(model, args):
    if args.freeze_backbone_except_trajectory_control:
        module = getattr(model, "trajectory_control_refiner", None)
        if module is None:
            raise RuntimeError(
                "--freeze_backbone_except_trajectory_control requires --trajectory_control_refiner."
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in module.parameters():
            parameter.requires_grad = True
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"[Freeze] trainable_params={trainable} total_params={total}")
        return
    if not args.freeze_backbone_except_new_heads:
        return
    trainable_modules = [
        module
        for module in (
            getattr(model, "endpoint_shape_refiner", None),
            getattr(model, "control_shape_refiner", None),
            getattr(model, "trajectory_control_refiner", None),
        )
        if module is not None
    ]
    if not trainable_modules:
        raise RuntimeError(
            "--freeze_backbone_except_new_heads requires at least one shape/control refiner head."
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[Freeze] trainable_params={trainable} total_params={total}")


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def tracked_best_specs(args, run_dir):
    secondary_k = args.eval_topk_secondary
    return {
        "best20": {
            "metric_key": "best20",
            "path": os.path.join(run_dir, f"best_best{secondary_k}.pt"),
        },
        "ade20": {
            "metric_key": f"ADE@{secondary_k}",
            "path": os.path.join(run_dir, f"best_ade{secondary_k}.pt"),
        }
    }


def init_best_records(args, run_dir):
    return {
        name: {"value": float("inf"), "epoch": 0, "path": spec["path"], "sort_key": None}
        for name, spec in tracked_best_specs(args, run_dir).items()
    }


def unified_best20_sort_key(metrics, args):
    secondary_k = args.eval_topk_secondary
    rare_k = secondary_k if secondary_k != args.eval_topk_primary else args.eval_topk_primary
    ade = float(metrics[f"ADE@{secondary_k}"])
    fde = float(metrics[f"FDE@{secondary_k}"])
    glev = float(metrics[f"GLeV@{secondary_k}"])
    rare_fde = float(metrics[f"rare_FDE@{rare_k}"])
    return (
        ade + fde,
        fde,
        ade,
        glev,
        rare_fde,
    )


def checkpoint_payload(model, optimizer, epoch, global_step, best_records, model_artifact, config, meta):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best": best_records,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "proto_summary_5d": model_artifact["summary_5d"],
        "proto_freq": model_artifact["frequency"],
        "basis_bank": model_artifact["basis_bank"],
        "prototype_mean_path": model_artifact["prototype_mean_path"],
        "local_basis_bank": model_artifact["local_basis_bank"],
        "micro_coeff_anchors": model_artifact.get("micro_coeff_anchors"),
        "config": config,
        "meta": meta,
    }


def save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def remove_legacy_best_checkpoints(run_dir, keep_names):
    if not os.path.isdir(run_dir):
        return
    if isinstance(keep_names, str):
        keep_names = {keep_names}
    else:
        keep_names = set(keep_names)
    for file_name in os.listdir(run_dir):
        if not file_name.startswith("best_") or not file_name.endswith(".pt"):
            continue
        if file_name in keep_names:
            continue
        path = os.path.join(run_dir, file_name)
        if os.path.isfile(path):
            os.remove(path)


def autocast_context(device, use_amp):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def evaluate(
    model,
    loader,
    device,
    args,
    enable_refiner,
    use_amp=False,
    limit_eval_batches=0,
    progress_desc="eval",
):
    metric_names = metric_names_for_protocol(args.eval_topk_primary, args.eval_topk_secondary)
    metric_sums = init_metric_sums(metric_names)
    total_count = 0
    rare_count = 0
    model.eval()
    total_batches = len(loader)
    if limit_eval_batches:
        total_batches = min(total_batches, limit_eval_batches)
    progress = tqdm(
        loader,
        leave=False,
        dynamic_ncols=True,
        total=total_batches,
        desc=progress_desc,
        file=sys.stdout,
        bar_format="{l_bar}{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(progress):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with autocast_context(device, use_amp):
                outputs = model(
                    batch["obs_xyz"],
                    batch["obs_mask"],
                    enable_refiner=enable_refiner,
                )
            metrics, count, rare = summarize_batch_metrics(
                outputs,
                batch,
                primary_k=args.eval_topk_primary,
                secondary_k=args.eval_topk_secondary,
                glev_topn_primary=args.glev_topn_primary,
                glev_topn_secondary=args.glev_topn_secondary,
            )
            update_metric_sums(metric_sums, metrics, count)
            total_count += count
            rare_count += rare
            postfix = {
                f"ADE@{args.eval_topk_primary}": format_scalar(metrics.get(f"ADE@{args.eval_topk_primary}"))
            }
            if args.eval_topk_secondary != args.eval_topk_primary:
                postfix[f"ADE@{args.eval_topk_secondary}"] = format_scalar(
                    metrics.get(f"ADE@{args.eval_topk_secondary}")
                )
            progress.set_postfix(postfix, refresh=False)
    progress.close()
    return average_metric_sums(metric_sums, total_count, rare_count), total_count, rare_count


def peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def load_resume_checkpoint(args, run_dir, model, optimizer, device):
    last_path = os.path.join(run_dir, "last.pt")
    best_records = init_best_records(args, run_dir)
    if not args.resume or not os.path.exists(last_path):
        return 1, 0, best_records

    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    saved_best = checkpoint.get("best", {})
    for key, record in best_records.items():
        if key not in saved_best:
            continue
        saved_record = saved_best[key]
        if isinstance(saved_record, dict):
            record["value"] = float(saved_record.get("value", record["value"]))
            record["epoch"] = int(saved_record.get("epoch", record["epoch"]))
            if saved_record.get("sort_key") is not None:
                record["sort_key"] = tuple(float(value) for value in saved_record["sort_key"])
        else:
            record["value"] = float(saved_record)
        record["path"] = record["path"]

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    return start_epoch, global_step, best_records


def init_loss_sums():
    return {name: 0.0 for name in LOSS_STAT_KEYS}


def update_loss_sums(loss_sums, loss_stats):
    for name in LOSS_STAT_KEYS:
        loss_sums[name] += float(loss_stats.get(name, 0.0))


def average_loss_sums(loss_sums, count):
    return {name: value / max(count, 1) for name, value in loss_sums.items()}


def format_scalar(value, precision=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "nan"
        return f"{value:.{precision}f}"
    return str(value)


def supports_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def main_metric_keys(args):
    return {"best20", "ADE@20", f"ADE@{args.eval_topk_secondary}"}


def has_main_line_best_update(args, best_updates):
    keys = main_metric_keys(args)
    return any(item.get("metric_key", "") in keys for item in best_updates)


def maybe_green(text, enabled):
    if not enabled or not supports_color():
        return text
    return f"{ANSI_GREEN}{text}{ANSI_RESET}"


def progress_postfix(batch_loss, loss_stats):
    return {
        "loss": format_scalar(batch_loss),
        "xyz": format_scalar(loss_stats.get("xyz")),
        "fde": format_scalar(loss_stats.get("fde")),
        "proto": format_scalar(loss_stats.get("proto")),
        "wADE": format_scalar(loss_stats.get("winner_ade")),
    }


def _metric_lines(args, metrics):
    primary_k = args.eval_topk_primary
    secondary_k = args.eval_topk_secondary
    rare_k = secondary_k if secondary_k != primary_k else primary_k

    eval_main = []
    if secondary_k != primary_k:
        eval_main.extend(
            [
                f"ADE@{secondary_k}={format_scalar(metrics[f'ADE@{secondary_k}'])}",
                f"FDE@{secondary_k}={format_scalar(metrics[f'FDE@{secondary_k}'])}",
                f"GLeV@{secondary_k}={format_scalar(metrics[f'GLeV@{secondary_k}'])}",
                f"rare_FDE@{rare_k}={format_scalar(metrics[f'rare_FDE@{rare_k}'])}",
            ]
        )
    else:
        eval_main.extend(
            [
                f"ADE@{primary_k}={format_scalar(metrics[f'ADE@{primary_k}'])}",
                f"FDE@{primary_k}={format_scalar(metrics[f'FDE@{primary_k}'])}",
                f"GLeV@{primary_k}={format_scalar(metrics[f'GLeV@{primary_k}'])}",
                f"rare_FDE@{rare_k}={format_scalar(metrics[f'rare_FDE@{rare_k}'])}",
            ]
        )

    eval_aux = [
        f"ADE@{primary_k}={format_scalar(metrics[f'ADE@{primary_k}'])}",
        f"FDE@{primary_k}={format_scalar(metrics[f'FDE@{primary_k}'])}",
        f"GLeV@{primary_k}={format_scalar(metrics[f'GLeV@{primary_k}'])}",
    ]

    eval_aux.extend(
        [
            f"Top1_ADE={format_scalar(metrics['Top1_ADE'])}",
            f"Top1_FDE={format_scalar(metrics['Top1_FDE'])}",
            f"proto_top1_acc={format_scalar(metrics['proto_top1_acc'])}",
            f"proto_rare_recall={format_scalar(metrics['proto_rare_recall'])}",
            f"score_entropy={format_scalar(metrics['score_entropy'])}",
            f"endpoint_var={format_scalar(metrics['endpoint_var'])}",
        ]
    )
    return eval_main, eval_aux


def format_epoch_summary(args, epoch, total_epochs, phase_name, train_loss, loss_stats, metrics, lr, memory_mb, best_updates):
    train_parts = [
        f"total={format_scalar(train_loss)}",
        f"xyz={format_scalar(loss_stats['xyz'])}",
        f"fde={format_scalar(loss_stats['fde'])}",
        f"proto={format_scalar(loss_stats['proto'])}",
        f"res={format_scalar(loss_stats['res'])}",
        f"res_hit={format_scalar(loss_stats['res_hit_rate'])}",
        f"score={format_scalar(loss_stats['score'])}",
        f"div={format_scalar(loss_stats['div'])}",
        f"coeff={format_scalar(loss_stats['coeff'])}",
        f"smooth={format_scalar(loss_stats['smooth'])}",
        f"gt_shape={format_scalar(loss_stats['gt_proto_shape'])}",
        f"gt_fde={format_scalar(loss_stats['gt_proto_fde'])}",
        f"gt_coeff={format_scalar(loss_stats['gt_proto_coeff'])}",
        f"anchor_recon={format_scalar(loss_stats['anchor_recon'])}",
        f"proj_coeff={format_scalar(loss_stats['projection_coeff'])}",
        f"proj_path={format_scalar(loss_stats['projection_path'])}",
        f"gt_hit={format_scalar(loss_stats['gt_proto_hit_rate'])}",
        f"winner_ADE={format_scalar(loss_stats['winner_ade'])}",
    ]
    eval_main, eval_aux = _metric_lines(args, metrics)
    best_text = ", ".join(item.get("metric_key", "?") for item in best_updates) if best_updates else "none"
    header = (
        f"[Epoch {epoch:03d}/{total_epochs:03d}] phase={phase_name} "
        f"lr={lr:.2e} mem={memory_mb:.0f}MB best_update={best_text}"
    )
    return "\n".join(
        [
            header,
            f"  train_loss: {' '.join(train_parts)}",
            maybe_green(f"  eval_main : {' '.join(eval_main)}", has_main_line_best_update(args, best_updates)),
            f"  eval_aux  : {' '.join(eval_aux)}",
        ]
    )


def main():
    args = build_parser().parse_args()
    apply_protocol(args)
    apply_training_defaults(args)
    set_seed(args.seed)
    device = select_device(args.device, allow_cpu=args.allow_cpu)
    apply_runtime_defaults(args, device)
    project_root = os.getcwd()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = build_grad_scaler(use_amp)

    train_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, "train")
    test_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, "test")

    train_dataset = ProtoBasisSceneDataset(
        data_dir=train_dir,
        split_name="train",
        obs_len=args.obs,
        pred_len=args.preds,
        obs_stride=args.obs_stride,
        pred_stride=args.pred_stride,
        max_agents=args.max_agents,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        local_basis_dim=args.local_basis_dim,
        micro_per_proto=args.micro_per_proto if args.micro_coeff_anchors else 0,
        artifact_max_samples=args.artifact_max_samples,
        rare_threshold=args.rare_threshold,
    )
    model_artifact = train_dataset.export_model_artifact()
    eval_dataset = ProtoBasisSceneDataset(
        data_dir=test_dir,
        split_name="test",
        obs_len=args.obs,
        pred_len=args.preds,
        obs_stride=args.obs_stride,
        pred_stride=args.pred_stride,
        max_agents=args.max_agents,
        model_artifact=model_artifact,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        local_basis_dim=args.local_basis_dim,
        micro_per_proto=args.micro_per_proto if args.micro_coeff_anchors else 0,
        rare_threshold=args.rare_threshold,
    )

    eval_loader = build_eval_loader(eval_dataset, args)
    model = build_model(args, model_artifact).to(device)
    initialize_from_checkpoint(
        model,
        args.init_checkpoint,
        device,
        allow_partial=args.allow_partial_init,
    )
    apply_freeze_policy(model, args)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze policy.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.stage_a_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    loss_fn = ProtoBasisLoss(
        lambda_xyz=args.lambda_xyz,
        lambda_fde=args.lambda_fde,
        lambda_proto=args.lambda_proto,
        lambda_res=args.lambda_res,
        lambda_score=args.lambda_score,
        lambda_div=args.lambda_div,
        lambda_coeff=args.lambda_coeff,
        lambda_smooth=args.lambda_smooth,
        lambda_gt_proto_shape=args.lambda_gt_proto_shape,
        lambda_gt_proto_fde=args.lambda_gt_proto_fde,
        lambda_gt_proto_coeff=args.lambda_gt_proto_coeff,
        score_hard_mix=args.score_hard_mix,
        score_fde_weight=args.score_fde_weight,
        score_soft_temperature=args.score_soft_temperature,
        proto_focal_gamma=args.proto_focal_gamma,
        proto_freq_weight_power=args.proto_freq_weight_power,
        proto_freq_weight_max=args.proto_freq_weight_max,
        endpoint_residual_supervision=args.endpoint_residual_supervision,
        lambda_anchor_recon=args.lambda_anchor_recon,
        anchor_recon_supervision=args.anchor_recon_supervision,
        lambda_projection_coeff=args.lambda_projection_coeff,
        lambda_projection_path=args.lambda_projection_path,
        projection_supervision=args.projection_supervision,
    )

    run_dir = os.path.join(args.save_dir, args.dataset_name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    best_specs = tracked_best_specs(args, run_dir)
    keep_best_names = [os.path.basename(spec["path"]) for spec in best_specs.values()]
    remove_legacy_best_checkpoints(run_dir, keep_names=keep_best_names)

    start_epoch, global_step, best_records = load_resume_checkpoint(args, run_dir, model, optimizer, device)

    meta = {
        "project_name": "ProtoBasis-Net",
        "protocol_name": args.protocol_name,
        "protocol_hash": protocol_hash(vars(args)),
        "artifact_hash": model_artifact_hash(model_artifact),
        "basis_hash": basis_hash(model_artifact),
        "git_commit": git_commit(project_root),
        "seed": args.seed,
        "rare_count": int(len(model_artifact["rare_ids"])),
        "n_proto": args.n_proto,
        "basis_dim": args.basis_dim,
        "endpoint_conditioning": args.endpoint_conditioning,
        "candidate_dense_topk": args.candidate_dense_topk,
        "coupled_decoder": bool(args.coupled_decoder),
        "coupled_decoder_iters": args.coupled_decoder_iters,
        "endpoint_shape_refiner": bool(args.endpoint_shape_refiner),
        "control_shape_refiner": bool(args.control_shape_refiner),
        "control_shape_points": args.control_shape_points,
        "trajectory_control_refiner": bool(args.trajectory_control_refiner),
        "trajectory_control_points": args.trajectory_control_points,
        "endpoint_residual_supervision": args.endpoint_residual_supervision,
        "anchor_recon_supervision": args.anchor_recon_supervision,
        "projection_supervision": args.projection_supervision,
        "micro_coeff_anchors": bool(args.micro_coeff_anchors),
        "lambda_gt_proto_shape": args.lambda_gt_proto_shape,
        "lambda_gt_proto_fde": args.lambda_gt_proto_fde,
        "lambda_gt_proto_coeff": args.lambda_gt_proto_coeff,
        "lambda_anchor_recon": args.lambda_anchor_recon,
        "lambda_projection_coeff": args.lambda_projection_coeff,
        "lambda_projection_path": args.lambda_projection_path,
        "proto_focal_gamma": args.proto_focal_gamma,
        "proto_freq_weight_power": args.proto_freq_weight_power,
        "init_checkpoint": args.init_checkpoint,
        "freeze_backbone_except_new_heads": bool(args.freeze_backbone_except_new_heads),
        "freeze_backbone_except_trajectory_control": bool(args.freeze_backbone_except_trajectory_control),
        "amp_enabled": use_amp,
    }
    recorder = RunRecorder(
        run_dir,
        vars(args),
        extra_metadata=meta,
        resume_state={
            "enabled": args.resume and start_epoch > 1,
            "last_epoch": start_epoch - 1,
            "best": best_records,
        },
    )

    if start_epoch > args.epochs:
        print(
            f"[Skip] run already reached epoch {start_epoch - 1}. "
            f"Use --extra_epochs N with --resume to continue."
        )
        return

    last_path = os.path.join(run_dir, "last.pt")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            cfg = stage_config(epoch, args)
            current_lr = set_epoch_lr(optimizer, epoch, args)
            train_loader = build_train_loader(train_dataset, args, rare_weight=cfg["rare_weight"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            epoch_loss_sums = init_loss_sums()
            seen_batches = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            progress = tqdm(
                train_loader,
                leave=False,
                dynamic_ncols=True,
                desc=f"E{epoch:03d}/{args.epochs:03d} {cfg['name']}",
                file=sys.stdout,
                bar_format="{l_bar}{bar:24}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            for batch_index, raw_batch in enumerate(progress):
                if args.limit_train_batches and batch_index >= args.limit_train_batches:
                    break
                batch = move_batch_to_device(raw_batch, device)
                with autocast_context(device, use_amp):
                    outputs = model(
                        batch["obs_xyz"],
                        batch["obs_mask"],
                        gt_proto_id=batch["gt_proto_id"],
                        force_gt_proto=cfg["force_gt_proto"],
                        enable_refiner=cfg["enable_refiner"],
                    )
                    loss, loss_stats = loss_fn(outputs, batch, cfg)

                batch_loss = float(loss.detach().item())
                loss = loss / max(args.grad_accum, 1)
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_index + 1) % args.grad_accum == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                train_loss_sum += batch_loss
                update_loss_sums(epoch_loss_sums, loss_stats)
                seen_batches += 1
                progress.set_postfix(progress_postfix(batch_loss, loss_stats), refresh=False)

            if seen_batches and seen_batches % args.grad_accum != 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            metrics, _, _ = evaluate(
                model,
                eval_loader,
                device=device,
                args=args,
                enable_refiner=cfg["enable_refiner"],
                use_amp=use_amp,
                limit_eval_batches=args.limit_eval_batches,
                progress_desc=f"V{epoch:03d}/{args.epochs:03d} eval",
            )
            memory_mb = peak_memory_mb(device)
            train_loss = train_loss_sum / max(seen_batches, 1)
            avg_loss_stats = average_loss_sums(epoch_loss_sums, seen_batches)

            best_updates = []
            secondary_k = args.eval_topk_secondary
            rare_k = secondary_k if secondary_k != args.eval_topk_primary else args.eval_topk_primary
            best_key = "best20"
            spec = best_specs[best_key]
            candidate_sort_key = unified_best20_sort_key(metrics, args)
            previous_sort_key = best_records[best_key].get("sort_key")
            if previous_sort_key is None or candidate_sort_key < tuple(previous_sort_key):
                previous_value = best_records[best_key]["value"]
                best_records[best_key]["value"] = float(metrics[f"ADE@{secondary_k}"])
                best_records[best_key]["epoch"] = epoch
                best_records[best_key]["sort_key"] = candidate_sort_key
                best_records[best_key]["fde_value"] = float(metrics[f"FDE@{secondary_k}"])
                best_records[best_key]["glev_value"] = float(metrics[f"GLeV@{secondary_k}"])
                best_records[best_key]["rare_fde_value"] = float(metrics[f"rare_FDE@{rare_k}"])
                best_records[best_key]["selection_score"] = candidate_sort_key[0]
                best_updates.append(
                    {
                        "record_key": best_key,
                        "metric_key": spec["metric_key"],
                        "previous_value": None if math.isinf(previous_value) else previous_value,
                        "value": float(metrics[f"ADE@{secondary_k}"]),
                        "epoch": epoch,
                    }
                )

            ade_key = "ade20"
            spec = best_specs[ade_key]
            ade_value = float(metrics[f"ADE@{secondary_k}"])
            if ade_value < best_records[ade_key]["value"]:
                previous_value = best_records[ade_key]["value"]
                best_records[ade_key]["value"] = ade_value
                best_records[ade_key]["epoch"] = epoch
                best_records[ade_key]["sort_key"] = (
                    ade_value,
                    float(metrics[f"FDE@{secondary_k}"]),
                    float(metrics[f"GLeV@{secondary_k}"]),
                    float(metrics[f"rare_FDE@{rare_k}"]),
                )
                best_records[ade_key]["fde_value"] = float(metrics[f"FDE@{secondary_k}"])
                best_records[ade_key]["glev_value"] = float(metrics[f"GLeV@{secondary_k}"])
                best_records[ade_key]["rare_fde_value"] = float(metrics[f"rare_FDE@{rare_k}"])
                best_records[ade_key]["selection_score"] = ade_value
                best_updates.append(
                    {
                        "record_key": ade_key,
                        "metric_key": spec["metric_key"],
                        "previous_value": None if math.isinf(previous_value) else previous_value,
                        "value": ade_value,
                        "epoch": epoch,
                    }
                )

            epoch_meta = dict(meta)
            epoch_meta["eval_enable_refiner"] = cfg["enable_refiner"]
            ckpt = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_records=best_records,
                model_artifact=model_artifact,
                config=vars(args),
                meta=epoch_meta,
            )
            save_checkpoint(last_path, ckpt)
            for update in best_updates:
                save_checkpoint(best_specs[update["record_key"]]["path"], ckpt)

            progress.close()
            recorder.log_epoch(
                epoch,
                cfg["name"],
                train_loss,
                metrics,
                current_lr,
                memory_mb,
                loss_stats=avg_loss_stats,
                best_updates=best_updates,
            )
            recorder.update_live_status(
                epoch,
                best_records,
                phase_name=cfg["name"],
                train_loss=train_loss,
                loss_stats=avg_loss_stats,
                metrics=metrics,
                lr=current_lr,
                peak_memory_mb=memory_mb,
                best_updates=best_updates,
            )
            print(
                format_epoch_summary(
                    args,
                    epoch,
                    args.epochs,
                    cfg["name"],
                    train_loss,
                    avg_loss_stats,
                    metrics,
                    current_lr,
                    memory_mb,
                    best_updates,
                ),
                flush=True,
            )

            patience = max(int(args.early_stop_patience), 0)
            min_epoch = max(int(args.early_stop_min_epoch), 0)
            best_epoch = max(
                int(best_records["best20"].get("epoch", 0)),
                int(best_records.get("ade20", {}).get("epoch", 0)),
            )
            if patience > 0 and epoch >= min_epoch and best_epoch > 0 and epoch - best_epoch >= patience:
                print(
                    f"[EarlyStop] tracked best metrics have not improved for {epoch - best_epoch} epochs "
                    f"(best epoch {best_epoch}); stopping at epoch {epoch}.",
                    flush=True,
                )
                break

        recorder.finalize(best_records)
        print("[Done] ProtoBasis-Net training finished.", flush=True)

    except KeyboardInterrupt:
        recorder.finalize_incomplete("interrupted", "Training interrupted by user.")
        raise
    except Exception as exc:
        recorder.finalize_incomplete("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
