from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _velocity_delta(xyz):
    return torch.cat([xyz[:, :, :1], xyz[:, :, 1:] - xyz[:, :, :-1]], dim=2)


def _velocity_to_angles(velocity):
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    yaw = torch.atan2(vy, vx + 1e-6)
    horizontal = torch.sqrt(vx.pow(2) + vy.pow(2) + 1e-6)
    pitch = torch.atan2(vz, horizontal)
    return yaw, pitch


def _build_rotation(yaw, pitch):
    cy, sy = torch.cos(-yaw), torch.sin(-yaw)
    cp, sp = torch.cos(-pitch), torch.sin(-pitch)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)
    rz = torch.stack(
        [
            torch.stack([cy, -sy, zeros], dim=-1),
            torch.stack([sy, cy, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cp, zeros, sp], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sp, zeros, cp], dim=-1),
        ],
        dim=-2,
    )
    return torch.bmm(ry, rz)


def _masked_mean(hidden, mask):
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom


class PoseNormalizer(nn.Module):
    def forward(self, obs_xyz):
        target = obs_xyz[:, 0]
        origin = target[:, -1]
        previous = target[:, -2] if target.size(1) > 1 else target[:, -1]
        direction = origin - previous
        yaw = torch.atan2(direction[:, 1], direction[:, 0] + 1e-6)
        horizontal = torch.sqrt(direction[:, 0].pow(2) + direction[:, 1].pow(2) + 1e-6)
        pitch = torch.atan2(direction[:, 2], horizontal)
        rotation = _build_rotation(yaw, pitch)

        centered = obs_xyz - origin[:, None, None, :]
        local_xyz = torch.einsum("bij,bntj->bnti", rotation, centered)
        return local_xyz, yaw, pitch, origin, rotation

    def inverse(self, local_xyz, origin, rotation):
        inverse_rotation = rotation.transpose(1, 2)
        global_xyz = torch.einsum("bij,bktj->bkti", inverse_rotation, local_xyz)
        return global_xyz + origin[:, None, None, :]


def build_local_features(local_xyz):
    delta = _velocity_delta(local_xyz)
    speed = torch.linalg.norm(delta, dim=-1, keepdim=True)
    yaw, pitch = _velocity_to_angles(delta)
    return torch.cat(
        [
            local_xyz,
            speed,
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(pitch).unsqueeze(-1),
            torch.cos(pitch).unsqueeze(-1),
        ],
        dim=-1,
    )


def build_global_features(obs_xyz):
    delta = _velocity_delta(obs_xyz)
    yaw, pitch = _velocity_to_angles(delta)
    return torch.cat(
        [
            obs_xyz,
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(pitch).unsqueeze(-1),
            torch.cos(pitch).unsqueeze(-1),
        ],
        dim=-1,
    )


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=96, nhead=4, ff_dim=192, layers=3, obs_len=40, dropout=0.1):
        super().__init__()
        local_dim = d_model // 2
        global_dim = d_model - local_dim
        self.local_proj = nn.Linear(8, local_dim)
        self.global_proj = nn.Linear(7, global_dim)
        self.time_emb = nn.Embedding(obs_len, d_model)
        self.register_buffer("time_ids", torch.arange(obs_len, dtype=torch.long), persistent=False)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.pool_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

    def forward(self, feats_local, feats_global):
        batch_size, num_agents, obs_len, _ = feats_local.shape
        local_h = self.local_proj(feats_local)
        global_h = self.global_proj(feats_global)
        hidden = torch.cat([local_h, global_h], dim=-1)
        hidden = hidden + self.time_emb(self.time_ids[:obs_len])[None, None, :, :]
        hidden = hidden.view(batch_size * num_agents, obs_len, -1)
        hidden = self.encoder(hidden)
        pooled = hidden.max(dim=1).values
        last = hidden[:, -1]
        out = self.pool_proj(torch.cat([pooled, last], dim=-1))
        agent_out = out.view(batch_size, num_agents, -1)
        return agent_out


class TargetSocialAggregator(nn.Module):
    def __init__(self, d_model=96, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(layers)]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 192),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(192, d_model),
                )
                for _ in range(layers)
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, agent_feat, obs_mask):
        query = agent_feat[:, :1]
        for attn, ffn, norm1, norm2 in zip(self.cross_attn, self.ffn, self.norm1, self.norm2):
            attended, _ = attn(query, agent_feat, agent_feat, key_padding_mask=~obs_mask)
            query = norm1(query + attended)
            query = norm2(query + ffn(query))
        scene_ctx = _masked_mean(agent_feat, obs_mask)
        return query.squeeze(1), scene_ctx


class PrototypeRouter(nn.Module):
    def __init__(self, n_proto=64, d_model=96, topk=5, dropout=0.1, endpoint_conditioning="rank"):
        super().__init__()
        if endpoint_conditioning not in {"rank", "proto"}:
            raise ValueError(f"Unsupported endpoint_conditioning: {endpoint_conditioning}")
        self.topk = topk
        self.endpoint_conditioning = endpoint_conditioning
        self.proto_emb = nn.Embedding(n_proto, d_model)
        self.proto_proj = nn.Linear(5, d_model)
        self.trunk = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )
        self.logit_head = nn.Linear(d_model, n_proto)
        self.endpoint_head = nn.Linear(d_model, topk * 3)
        if self.endpoint_conditioning == "proto":
            self.endpoint_proto_head = nn.Sequential(
                nn.Linear(d_model * 2 + 5, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )
        else:
            self.endpoint_proto_head = None
        self.generic_proto = nn.Parameter(torch.randn(topk, d_model))
        self.generic_endpoint_head = nn.Linear(d_model, topk * 3)

    def _rank_endpoint_residual(self, router_input, hidden):
        first_layer = self.endpoint_head[0] if isinstance(self.endpoint_head, nn.Sequential) else self.endpoint_head
        endpoint_source = router_input if first_layer.in_features == router_input.size(-1) else hidden
        endpoint_raw = self.endpoint_head(endpoint_source)
        if endpoint_raw.size(-1) == 3:
            return endpoint_raw[:, None, :].expand(-1, self.topk, -1)
        return endpoint_raw.view(hidden.size(0), self.topk, 3)

    def forward(self, target_ctx, scene_ctx, proto_summary_5d, gt_proto_id=None, force_gt_proto=False, disable_router=False):
        router_input = torch.cat([target_ctx, scene_ctx], dim=-1)
        hidden = self.trunk(router_input)
        logits = self.logit_head(hidden)
        top_idx = logits.topk(self.topk, dim=-1).indices

        if force_gt_proto and gt_proto_id is not None:
            top_idx = top_idx.clone()
            for batch_index in range(top_idx.size(0)):
                gt_idx = int(gt_proto_id[batch_index].item())
                if gt_idx not in top_idx[batch_index].tolist():
                    top_idx[batch_index, -1] = gt_idx

        if disable_router:
            proto_token = self.generic_proto[None, :, :].expand(hidden.size(0), -1, -1)
            endpoint_residual = self.generic_endpoint_head(hidden).view(hidden.size(0), self.topk, 3)
            endpoint_local = endpoint_residual
        else:
            selected_summary = proto_summary_5d[top_idx]
            proto_token = self.proto_emb(top_idx) + self.proto_proj(selected_summary)
            if self.endpoint_conditioning == "proto":
                hidden_rep = hidden[:, None, :].expand(-1, self.topk, -1)
                endpoint_input = torch.cat([hidden_rep, proto_token, selected_summary], dim=-1)
                endpoint_residual = self.endpoint_proto_head(endpoint_input)
            else:
                endpoint_residual = self._rank_endpoint_residual(router_input, hidden)
            endpoint_local = selected_summary[:, :, :3] + endpoint_residual
        return logits, top_idx, proto_token, endpoint_residual, endpoint_local


class PrototypeConditionedQueryDecoder(nn.Module):
    def __init__(self, d_model=96, n_micro=4, basis_dim=16, ff_dim=192, dropout=0.1):
        super().__init__()
        self.n_micro = n_micro
        self.endpoint_proj = nn.Linear(3, d_model)
        self.coeff_anchor_proj = nn.Linear(basis_dim, d_model)
        self.micro = nn.Parameter(torch.randn(n_micro, d_model))

        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d_model),
                )
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(6)])

        self.coeff_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, basis_dim))
        self.score_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, proto_token, endpoint_local, target_ctx, agent_feat, obs_mask, micro_coeff_anchor=None):
        batch_size = proto_token.size(0)
        topk_proto = proto_token.size(1)
        endpoint_token = self.endpoint_proj(endpoint_local)
        if micro_coeff_anchor is None:
            coeff_anchor_token = 0.0
        else:
            coeff_anchor_token = self.coeff_anchor_proj(micro_coeff_anchor)
        hidden = (
            proto_token[:, :, None, :]
            + endpoint_token[:, :, None, :]
            + self.micro[None, None, :, :]
            + coeff_anchor_token
            + target_ctx[:, None, None, :]
        )
        hidden = hidden.view(batch_size, proto_token.size(1) * self.n_micro, -1)
        norm_index = 0
        for layer_index in range(2):
            self_attended, _ = self.self_attn[layer_index](hidden, hidden, hidden)
            hidden = self.norms[norm_index](hidden + self_attended)
            norm_index += 1

            cross_attended, _ = self.cross_attn[layer_index](
                hidden,
                agent_feat,
                agent_feat,
                key_padding_mask=~obs_mask,
            )
            hidden = self.norms[norm_index](hidden + cross_attended)
            norm_index += 1

            hidden = self.norms[norm_index](hidden + self.ffn[layer_index](hidden))
            norm_index += 1

        coeff_delta = self.coeff_head(hidden)
        if micro_coeff_anchor is None:
            coeff = coeff_delta
        else:
            coeff = coeff_delta + micro_coeff_anchor.reshape(batch_size, topk_proto * self.n_micro, -1)
        score = self.score_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden)).squeeze(-1)
        return hidden, coeff, score, gate, coeff_delta


class BasisBank(nn.Module):
    def __init__(self, basis_bank):
        super().__init__()
        self.register_buffer("basis_bank", basis_bank.float())

    def forward(self, anchor, coeff):
        residual = torch.einsum("bkm,mtd->bktd", coeff, self.basis_bank)
        return anchor + residual


class TemporalResidualRefiner(nn.Module):
    def __init__(self, d_model=96):
        super().__init__()
        self.query_proj = nn.Linear(d_model, 32)
        self.depthwise = nn.Conv1d(38, 38, kernel_size=5, padding=2, groups=38)
        self.pointwise = nn.Conv1d(38, 64, kernel_size=1)
        self.out = nn.Conv1d(64, 3, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, coarse_local_xyz, query_feat, gate):
        delta = torch.cat(
            [coarse_local_xyz[:, :, :1], coarse_local_xyz[:, :, 1:] - coarse_local_xyz[:, :, :-1]],
            dim=2,
        )
        query = self.query_proj(query_feat)[:, :, None, :].expand(-1, -1, coarse_local_xyz.size(2), -1)
        features = torch.cat([coarse_local_xyz, delta, query], dim=-1)
        batch_size, num_modes, pred_len, channels = features.shape
        conv_in = features.reshape(batch_size * num_modes, pred_len, channels).transpose(1, 2)
        hidden = self.depthwise(conv_in)
        hidden = self.act(self.pointwise(hidden))
        update = self.out(hidden).transpose(1, 2).reshape(batch_size, num_modes, pred_len, 3)
        return coarse_local_xyz + gate[:, :, None, None] * update


class EndpointPreservingShapeRefiner(nn.Module):
    def __init__(self, d_model=96, pred_len=120, hidden_channels=128):
        super().__init__()
        self.query_proj = nn.Linear(d_model, 32)
        in_channels = 38
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        endpoint_envelope = 1.0 - torch.linspace(0.0, 1.0, steps=pred_len)
        self.register_buffer("endpoint_envelope", endpoint_envelope, persistent=False)

    def forward(self, local_xyz, query_feat):
        delta = torch.cat([local_xyz[:, :, :1], local_xyz[:, :, 1:] - local_xyz[:, :, :-1]], dim=2)
        query = self.query_proj(query_feat)[:, :, None, :].expand(-1, -1, local_xyz.size(2), -1)
        features = torch.cat([local_xyz, delta, query], dim=-1)
        batch_size, num_modes, pred_len, channels = features.shape
        conv_in = features.reshape(batch_size * num_modes, pred_len, channels).transpose(1, 2)
        update = self.net(conv_in).transpose(1, 2).reshape(batch_size, num_modes, pred_len, 3)
        envelope = self.endpoint_envelope.to(device=local_xyz.device, dtype=local_xyz.dtype)
        return local_xyz + envelope[None, None, :, None] * update


class EndpointPreservingControlPointRefiner(nn.Module):
    def __init__(self, d_model=96, pred_len=120, num_control_points=16, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.num_control_points = int(num_control_points)
        stats_dim = 24
        input_dim = d_model + stats_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_control_points * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        endpoint_envelope = 1.0 - torch.linspace(0.0, 1.0, steps=pred_len)
        self.register_buffer("endpoint_envelope", endpoint_envelope, persistent=False)

    def forward(self, local_xyz, query_feat):
        velocity = local_xyz[:, :, 1:] - local_xyz[:, :, :-1]
        start_velocity = velocity[:, :, 0]
        end_velocity = velocity[:, :, -1]
        mean_velocity = velocity.mean(dim=2)
        stats = torch.cat(
            [
                local_xyz[:, :, 0],
                local_xyz[:, :, local_xyz.size(2) // 3],
                local_xyz[:, :, (2 * local_xyz.size(2)) // 3],
                local_xyz[:, :, -1],
                local_xyz.mean(dim=2),
                start_velocity,
                end_velocity,
                mean_velocity,
            ],
            dim=-1,
        )
        control = self.net(torch.cat([query_feat, stats], dim=-1))
        batch_size, num_modes = local_xyz.shape[:2]
        control = control.reshape(batch_size * num_modes, self.num_control_points, 3).transpose(1, 2)
        residual = F.interpolate(control, size=local_xyz.size(2), mode="linear", align_corners=True)
        residual = residual.transpose(1, 2).reshape(batch_size, num_modes, local_xyz.size(2), 3)
        envelope = self.endpoint_envelope.to(device=local_xyz.device, dtype=local_xyz.dtype)
        return local_xyz + envelope[None, None, :, None] * residual


class EndpointSetRefiner(nn.Module):
    def __init__(self, d_model=96, pred_len=120, hidden_dim=192, max_delta=0.4, dropout=0.1):
        super().__init__()
        self.max_delta = float(max_delta)
        stats_dim = 19
        input_dim = d_model + stats_dim
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.set_attn = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.delta_head = nn.Linear(d_model, 3)
        self.gate_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)
        alpha = torch.linspace(0.0, 1.0, steps=pred_len)
        self.register_buffer("endpoint_alpha", alpha, persistent=False)

    def forward(self, local_xyz, query_feat, pred_score):
        endpoints = local_xyz[:, :, -1]
        endpoint_mean = endpoints.mean(dim=1, keepdim=True)
        endpoint_std = endpoints.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        centered_endpoint = endpoints - endpoint_mean
        path_mean = local_xyz.mean(dim=2)
        path_mid = local_xyz[:, :, local_xyz.size(2) // 2]
        score_prob = pred_score.softmax(dim=-1).unsqueeze(-1)
        stats = torch.cat(
            [
                endpoints,
                centered_endpoint,
                endpoint_mean.expand_as(endpoints),
                endpoint_std.expand_as(endpoints),
                path_mean,
                path_mid,
                score_prob,
            ],
            dim=-1,
        )
        hidden = self.in_proj(torch.cat([query_feat, stats], dim=-1))
        attended, _ = self.set_attn(hidden, hidden, hidden, need_weights=False)
        hidden = hidden + attended
        hidden = hidden + self.ffn(hidden)
        gate = torch.sigmoid(self.gate_head(hidden))
        endpoint_delta = torch.tanh(self.delta_head(hidden)) * self.max_delta * gate
        alpha = self.endpoint_alpha.to(device=local_xyz.device, dtype=local_xyz.dtype)
        refined = local_xyz + alpha[None, None, :, None] * endpoint_delta[:, :, None, :]
        return refined, endpoint_delta, gate.squeeze(-1)


class CoupledEndpointCoeffDecoder(nn.Module):
    def __init__(self, d_model=96, basis_dim=16, iters=1, dropout=0.1):
        super().__init__()
        self.iters = max(int(iters), 0)
        update_in_dim = d_model + basis_dim + 12
        self.update_blocks = nn.ModuleList()
        self.endpoint_heads = nn.ModuleList()
        self.coeff_heads = nn.ModuleList()
        for _ in range(self.iters):
            block = nn.Sequential(
                nn.Linear(update_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            endpoint_head = nn.Linear(d_model, 3)
            coeff_head = nn.Linear(d_model, basis_dim)
            nn.init.zeros_(endpoint_head.weight)
            nn.init.zeros_(endpoint_head.bias)
            nn.init.zeros_(coeff_head.weight)
            nn.init.zeros_(coeff_head.bias)
            self.update_blocks.append(block)
            self.endpoint_heads.append(endpoint_head)
            self.coeff_heads.append(coeff_head)

    def forward(self, query_feat, endpoint_local, coeff, coarse_local, basis_bank, anchor_alpha):
        active_query = query_feat
        for block, endpoint_head, coeff_head in zip(self.update_blocks, self.endpoint_heads, self.coeff_heads):
            coarse_end = coarse_local[:, :, -1]
            coarse_mean = coarse_local.mean(dim=2)
            endpoint_gap = coarse_end - endpoint_local
            update_input = torch.cat([active_query, endpoint_local, coarse_end, coarse_mean, endpoint_gap, coeff], dim=-1)
            update = block(update_input)
            endpoint_local = endpoint_local + endpoint_head(update)
            coeff = coeff + coeff_head(update)
            anchor_local = build_anchor(endpoint_local, anchor_alpha.to(endpoint_local))
            coarse_local = basis_bank(anchor_local, coeff)
        return active_query, endpoint_local, coeff, coarse_local


def build_anchor(endpoint_local, alpha):
    return alpha[None, None, :, None] * endpoint_local[:, :, None, :]


class ProtoBasisNet(nn.Module):
    def __init__(
        self,
        obs_len=40,
        pred_len=120,
        d_model=96,
        nhead=4,
        ff_dim=192,
        encoder_layers=3,
        social_layers=2,
        topk_proto=5,
        n_micro=4,
        n_proto=64,
        basis_dim=16,
        local_basis_dim=0,
        support_aware_local_basis=False,
        two_stage_decoder=False,
        two_stage_update_endpoint=True,
        two_stage_update_coeff=True,
        dropout=0.1,
        proto_summary_5d: Optional[torch.Tensor] = None,
        proto_frequency: Optional[torch.Tensor] = None,
        basis_bank: Optional[torch.Tensor] = None,
        prototype_mean_path: Optional[torch.Tensor] = None,
        local_basis_bank: Optional[torch.Tensor] = None,
        micro_coeff_anchors: Optional[torch.Tensor] = None,
        use_micro_coeff_anchors=False,
        endpoint_conditioning="rank",
        candidate_dense_topk=0,
        coupled_decoder=False,
        coupled_decoder_iters=0,
        micro_endpoint_offsets=False,
        endpoint_shape_refiner=False,
        control_shape_refiner=False,
        control_shape_points=16,
        endpoint_set_refiner=False,
        endpoint_set_max_delta=0.4,
        disable_social=False,
        disable_router=False,
        disable_refiner=False,
    ):
        super().__init__()
        if proto_summary_5d is None:
            raise ValueError("proto_summary_5d is required for ProtoBasis-Net.")
        if basis_bank is None:
            raise ValueError("basis_bank is required for ProtoBasis-Net.")
        if proto_frequency is None:
            proto_frequency = torch.ones(proto_summary_5d.size(0), dtype=torch.float32)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.topk_proto = topk_proto
        self.n_micro = n_micro
        self.num_modes = topk_proto * n_micro
        self.candidate_dense_topk = int(candidate_dense_topk or 0)
        self.disable_social = disable_social
        self.disable_router = disable_router
        self.disable_refiner = disable_refiner
        self.support_aware_local_basis = support_aware_local_basis
        self.two_stage_decoder = two_stage_decoder
        self.two_stage_update_endpoint = two_stage_update_endpoint
        self.two_stage_update_coeff = two_stage_update_coeff
        self.use_micro_coeff_anchors = bool(use_micro_coeff_anchors)
        self.coupled_decoder_enabled = bool(coupled_decoder) and int(coupled_decoder_iters or 0) > 0
        self.micro_endpoint_offsets_enabled = bool(micro_endpoint_offsets)
        self.endpoint_shape_refiner_enabled = bool(endpoint_shape_refiner)
        self.control_shape_refiner_enabled = bool(control_shape_refiner)
        self.endpoint_set_refiner_enabled = bool(endpoint_set_refiner)
        self.pose_normalizer = PoseNormalizer()
        self.register_buffer("anchor_alpha", torch.linspace(0.0, 1.0, steps=pred_len), persistent=False)
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            nhead=nhead,
            ff_dim=ff_dim,
            layers=encoder_layers,
            obs_len=obs_len,
            dropout=dropout,
        )
        self.social_aggregator = TargetSocialAggregator(
            d_model=d_model,
            nhead=nhead,
            layers=social_layers,
            dropout=dropout,
        )
        self.prototype_router = PrototypeRouter(
            n_proto=n_proto,
            d_model=d_model,
            topk=topk_proto,
            dropout=dropout,
            endpoint_conditioning=endpoint_conditioning,
        )
        self.query_decoder = PrototypeConditionedQueryDecoder(
            d_model=d_model,
            n_micro=n_micro,
            basis_dim=basis_dim,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.basis_bank = BasisBank(basis_bank)
        basis_matrix = self.basis_bank.basis_bank.reshape(self.basis_bank.basis_bank.size(0), -1)
        self.register_buffer("basis_matrix_flat", basis_matrix.float(), persistent=False)
        if basis_matrix.numel() > 0:
            basis_pinv = torch.linalg.pinv(basis_matrix)
        else:
            basis_pinv = torch.zeros(basis_matrix.size(1), 0, dtype=basis_matrix.dtype)
        self.register_buffer("basis_pinv", basis_pinv.float(), persistent=False)
        self.refiner = TemporalResidualRefiner(d_model=d_model)
        self.local_basis_dim = int(local_basis_dim)

        self.register_buffer("proto_summary_5d", proto_summary_5d.float())
        self.register_buffer("proto_frequency", proto_frequency.float())
        if prototype_mean_path is None:
            prototype_mean_path = torch.zeros(n_proto, pred_len, 3, dtype=torch.float32)
        if local_basis_bank is None:
            local_basis_bank = torch.zeros(n_proto, self.local_basis_dim, pred_len, 3, dtype=torch.float32)
        self.register_buffer("prototype_mean_path", prototype_mean_path.float())
        self.register_buffer("local_basis_bank", local_basis_bank.float())
        if micro_coeff_anchors is None:
            micro_coeff_anchors = torch.zeros(n_proto, n_micro, basis_dim, dtype=torch.float32)
        micro_coeff_anchors = micro_coeff_anchors.float()
        if micro_coeff_anchors.dim() != 3 or micro_coeff_anchors.size(0) != n_proto or micro_coeff_anchors.size(2) != basis_dim:
            micro_coeff_anchors = torch.zeros(n_proto, n_micro, basis_dim, dtype=torch.float32)
        elif micro_coeff_anchors.size(1) < n_micro:
            pad = torch.zeros(
                n_proto,
                n_micro - micro_coeff_anchors.size(1),
                basis_dim,
                dtype=micro_coeff_anchors.dtype,
                device=micro_coeff_anchors.device,
            )
            micro_coeff_anchors = torch.cat([micro_coeff_anchors, pad], dim=1)
        elif micro_coeff_anchors.size(1) > n_micro:
            micro_coeff_anchors = micro_coeff_anchors[:, :n_micro]
        self.register_buffer("micro_coeff_anchors", micro_coeff_anchors, persistent=False)
        self.has_micro_coeff_anchors = self.use_micro_coeff_anchors and self.micro_coeff_anchors.numel() > 0
        self.has_local_basis = self.local_basis_dim > 0 and self.local_basis_bank.numel() > 0
        if self.micro_endpoint_offsets_enabled:
            self.micro_endpoint_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 3),
            )
            nn.init.zeros_(self.micro_endpoint_head[-1].weight)
            nn.init.zeros_(self.micro_endpoint_head[-1].bias)
        else:
            self.micro_endpoint_head = None
        if self.has_local_basis:
            self.local_coeff_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, self.local_basis_dim),
            )
            self.local_mix_gate = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.local_coeff_head = None
            self.local_mix_gate = None
        if self.two_stage_decoder:
            self.stage2_proj = nn.Sequential(
                nn.Linear(d_model + 6, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.stage2_endpoint_head = nn.Linear(d_model, 3)
            self.stage2_coeff_head = nn.Linear(d_model, basis_dim)
        else:
            self.stage2_proj = None
            self.stage2_endpoint_head = None
            self.stage2_coeff_head = None
        if self.coupled_decoder_enabled:
            self.coupled_decoder = CoupledEndpointCoeffDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                iters=int(coupled_decoder_iters),
                dropout=dropout,
            )
        else:
            self.coupled_decoder = None
        if self.endpoint_shape_refiner_enabled:
            self.endpoint_shape_refiner = EndpointPreservingShapeRefiner(d_model=d_model, pred_len=pred_len)
        else:
            self.endpoint_shape_refiner = None
        if self.control_shape_refiner_enabled:
            self.control_shape_refiner = EndpointPreservingControlPointRefiner(
                d_model=d_model,
                pred_len=pred_len,
                num_control_points=control_shape_points,
                dropout=dropout,
            )
        else:
            self.control_shape_refiner = None
        if self.endpoint_set_refiner_enabled:
            self.endpoint_set_refiner = EndpointSetRefiner(
                d_model=d_model,
                pred_len=pred_len,
                hidden_dim=ff_dim,
                max_delta=endpoint_set_max_delta,
                dropout=dropout,
            )
        else:
            self.endpoint_set_refiner = None
        keep_indices = []
        if 0 < self.candidate_dense_topk < self.topk_proto and self.n_micro > 1:
            for proto_rank in range(self.topk_proto):
                micro_count = self.n_micro if proto_rank < self.candidate_dense_topk else 1
                for micro_rank in range(micro_count):
                    keep_indices.append(proto_rank * self.n_micro + micro_rank)
        if keep_indices and len(keep_indices) < self.topk_proto * self.n_micro:
            keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
        else:
            keep_tensor = torch.empty(0, dtype=torch.long)
        self.register_buffer("candidate_keep_indices", keep_tensor, persistent=False)

    def forward(self, obs_xyz, obs_mask, gt_proto_id=None, force_gt_proto=False, enable_refiner=True):
        local_xyz, _, _, origin, rotation = self.pose_normalizer(obs_xyz)
        feats_local = build_local_features(local_xyz)
        feats_global = build_global_features(obs_xyz)

        agent_feat = self.temporal_encoder(feats_local, feats_global)
        if self.disable_social:
            target_ctx = agent_feat[:, 0]
            scene_ctx = _masked_mean(agent_feat, obs_mask)
        else:
            target_ctx, scene_ctx = self.social_aggregator(agent_feat, obs_mask)

        proto_logits, top_proto_idx, proto_token, endpoint_residual, endpoint_local = self.prototype_router(
            target_ctx,
            scene_ctx,
            self.proto_summary_5d,
            gt_proto_id=gt_proto_id,
            force_gt_proto=force_gt_proto,
            disable_router=self.disable_router,
        )

        micro_coeff_anchor = self.micro_coeff_anchors[top_proto_idx] if self.has_micro_coeff_anchors else None
        query_feat, coeff, pred_score, difficulty_gate, coeff_delta = self.query_decoder(
            proto_token,
            endpoint_local,
            target_ctx,
            agent_feat,
            obs_mask,
            micro_coeff_anchor=micro_coeff_anchor,
        )
        endpoint_mode_local = endpoint_local.repeat_interleave(self.n_micro, dim=1)
        micro_endpoint_delta = None
        if self.micro_endpoint_head is not None:
            micro_endpoint_delta = self.micro_endpoint_head(query_feat)
            endpoint_mode_local = endpoint_mode_local + micro_endpoint_delta
        anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
        coarse_local = self.basis_bank(anchor_local, coeff)
        active_query = query_feat
        if self.two_stage_decoder:
            stage2_input = torch.cat([query_feat, coarse_local[:, :, -1], coarse_local.mean(dim=2)], dim=-1)
            stage2_query = query_feat + self.stage2_proj(stage2_input)
            if self.two_stage_update_endpoint:
                endpoint_mode_local = endpoint_mode_local + self.stage2_endpoint_head(stage2_query)
            if self.two_stage_update_coeff:
                stage2_coeff_delta = self.stage2_coeff_head(stage2_query)
                coeff = coeff + stage2_coeff_delta
                coeff_delta = coeff_delta + stage2_coeff_delta
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coarse_local = self.basis_bank(anchor_local, coeff)
            active_query = stage2_query
        if self.coupled_decoder is not None:
            coeff_before_coupled = coeff
            active_query, endpoint_mode_local, coeff, coarse_local = self.coupled_decoder(
                active_query,
                endpoint_mode_local,
                coeff,
                coarse_local,
                self.basis_bank,
                self.anchor_alpha,
            )
            coeff_delta = coeff_delta + (coeff - coeff_before_coupled)
        candidate_proto_idx = top_proto_idx.repeat_interleave(self.n_micro, dim=1)
        if self.candidate_keep_indices.numel() > 0:
            keep = self.candidate_keep_indices.to(device=coeff.device)
            query_feat = query_feat.index_select(1, keep)
            coeff = coeff.index_select(1, keep)
            pred_score = pred_score.index_select(1, keep)
            difficulty_gate = difficulty_gate.index_select(1, keep)
            coeff_delta = coeff_delta.index_select(1, keep)
            endpoint_mode_local = endpoint_mode_local.index_select(1, keep)
            coarse_local = coarse_local.index_select(1, keep)
            active_query = active_query.index_select(1, keep)
            candidate_proto_idx = candidate_proto_idx.index_select(1, keep)
        if self.has_local_basis:
            proto_mean_path = self.prototype_mean_path[candidate_proto_idx]
            proto_mean_endpoint = proto_mean_path[:, :, -1, :]
            aligned_proto_mean = proto_mean_path + (endpoint_mode_local - proto_mean_endpoint)[:, :, None, :]

            local_basis = self.local_basis_bank[candidate_proto_idx]
            local_coeff = self.local_coeff_head(active_query)
            local_residual = torch.einsum("bkm,bkmtd->bktd", local_coeff, local_basis)
            local_path = aligned_proto_mean + local_residual
            local_gate = torch.sigmoid(self.local_mix_gate(active_query)).unsqueeze(-1)
            if self.support_aware_local_basis:
                proto_support = self.proto_frequency[candidate_proto_idx]
                support_scale = (proto_support / self.proto_frequency.max().clamp_min(1e-6)).clamp_min(1e-6).sqrt()
                support_scale = support_scale.unsqueeze(-1).unsqueeze(-1)
                local_gate = local_gate * support_scale
            coarse_local = coarse_local + local_gate * (local_path - coarse_local)
        use_refiner = enable_refiner and (not self.disable_refiner)
        refined_local = self.refiner(coarse_local, active_query, difficulty_gate) if use_refiner else coarse_local
        if self.endpoint_shape_refiner is not None:
            refined_local = self.endpoint_shape_refiner(refined_local, active_query)
        if self.control_shape_refiner is not None:
            refined_local = self.control_shape_refiner(refined_local, active_query)
        endpoint_set_delta = None
        endpoint_set_gate = None
        if self.endpoint_set_refiner is not None:
            refined_local, endpoint_set_delta, endpoint_set_gate = self.endpoint_set_refiner(
                refined_local,
                active_query,
                pred_score,
            )

        pred_xyz = self.pose_normalizer.inverse(refined_local, origin, rotation)

        aux = {
            "coeff": coeff,
            "coeff_delta": coeff_delta,
            "endpoint_residual": endpoint_residual,
            "endpoint_mode_local": endpoint_mode_local,
            "coarse_local": coarse_local,
            "candidate_proto_idx": candidate_proto_idx,
            "basis_matrix": self.basis_matrix_flat,
            "basis_pinv": self.basis_pinv,
            "proto_frequency": self.proto_frequency,
            "proto_summary_5d": self.proto_summary_5d,
        }
        if micro_endpoint_delta is not None:
            aux["micro_endpoint_delta"] = micro_endpoint_delta
        if endpoint_set_delta is not None:
            aux["endpoint_set_delta"] = endpoint_set_delta
            aux["endpoint_set_gate"] = endpoint_set_gate

        return {
            "pred_xyz": pred_xyz,
            "pred_score": pred_score,
            "proto_logits": proto_logits,
            "top_proto_idx": top_proto_idx,
            "aux": aux,
        }


ProtoBasisFlight = ProtoBasisNet
