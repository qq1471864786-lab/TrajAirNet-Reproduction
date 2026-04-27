import math
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

    def forward(self, feats_local, feats_global, return_target_sequence=False):
        batch_size, num_agents, obs_len, _ = feats_local.shape
        local_h = self.local_proj(feats_local)
        global_h = self.global_proj(feats_global)
        hidden = torch.cat([local_h, global_h], dim=-1)
        hidden = hidden + self.time_emb(self.time_ids[:obs_len])[None, None, :, :]
        hidden = hidden.view(batch_size * num_agents, obs_len, -1)
        hidden = self.encoder(hidden)
        sequence_hidden = hidden.view(batch_size, num_agents, obs_len, -1)
        pooled = hidden.max(dim=1).values
        last = hidden[:, -1]
        out = self.pool_proj(torch.cat([pooled, last], dim=-1))
        agent_out = out.view(batch_size, num_agents, -1)
        if return_target_sequence:
            return agent_out, sequence_hidden[:, 0]
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


class BasisAwareCoeffDecoder(nn.Module):
    def __init__(self, d_model=96, basis_dim=16, pred_len=120, nhead=4, ff_dim=256, dropout=0.1, mode="residual"):
        super().__init__()
        if mode not in {"residual", "replace"}:
            raise ValueError(f"Unsupported basis-aware coeff mode: {mode}")
        self.mode = mode
        self.basis_dim = int(basis_dim)
        self.basis_proj = nn.Linear(pred_len * 3, d_model)
        self.basis_id = nn.Parameter(torch.randn(basis_dim, d_model) * 0.02)
        self.endpoint_proj = nn.Linear(3, d_model)
        self.coeff_proj = nn.Linear(basis_dim, d_model)
        self.stats_proj = nn.Linear(12, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.slot_norm = nn.LayerNorm(d_model)
        self.prior_proj = nn.Linear(1, d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim // 2),
            nn.GELU(),
            nn.Linear(ff_dim // 2, 1),
        )
        nn.init.zeros_(self.slot_mlp[-1].weight)
        nn.init.zeros_(self.slot_mlp[-1].bias)

    def forward(self, query_feat, endpoint_local, coeff, coarse_local, basis_bank):
        coarse_end = coarse_local[:, :, -1]
        coarse_mean = coarse_local.mean(dim=2)
        endpoint_gap = coarse_end - endpoint_local
        stats = torch.cat([endpoint_local, coarse_end, coarse_mean, endpoint_gap], dim=-1)
        candidate_query = self.query_norm(
            query_feat
            + self.endpoint_proj(endpoint_local)
            + self.coeff_proj(coeff)
            + self.stats_proj(stats)
        )
        basis_flat = basis_bank.reshape(self.basis_dim, -1).to(device=coeff.device, dtype=coeff.dtype)
        basis_tokens = self.basis_proj(basis_flat) + self.basis_id.to(device=coeff.device, dtype=coeff.dtype)
        batch_size, num_modes = coeff.shape[:2]
        query = candidate_query.reshape(batch_size * num_modes, 1, -1)
        tokens = basis_tokens[None].expand(batch_size * num_modes, -1, -1)
        attended, _ = self.cross_attn(query, tokens, tokens)
        refined = query + attended
        refined = refined + self.ffn(refined)
        refined = refined.reshape(batch_size, num_modes, -1)
        slot = (
            refined[:, :, None, :]
            + basis_tokens[None, None, :, :]
            + self.prior_proj(coeff.unsqueeze(-1))
        )
        coeff_update = self.slot_mlp(self.slot_norm(slot)).squeeze(-1)
        if self.mode == "replace":
            return coeff_update
        return coeff + coeff_update


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


class SoftPrototypeSetDecoder(nn.Module):
    def __init__(
        self,
        d_model=96,
        basis_dim=16,
        num_modes=20,
        nhead=4,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.num_modes = int(num_modes)
        self.mode_tokens = nn.Parameter(torch.randn(self.num_modes, d_model) * 0.02)
        self.target_proj = nn.Linear(d_model, d_model)
        self.scene_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.proto_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.agent_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
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
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(8)])
        self.endpoint_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, basis_dim),
        )
        self.score_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim // 2), nn.GELU(), nn.Linear(ff_dim // 2, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim // 2), nn.GELU(), nn.Linear(ff_dim // 2, 1))
        for head in (self.endpoint_head, self.coeff_head, self.score_head, self.gate_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        target_ctx,
        scene_ctx,
        agent_feat,
        obs_mask,
        proto_summary_5d,
        proto_embedding_weight,
        proto_proj,
        base_query=None,
        base_endpoint=None,
        base_coeff=None,
        base_score=None,
        base_gate=None,
    ):
        batch_size = target_ctx.size(0)
        context = self.target_proj(target_ctx)[:, None, :] + self.scene_proj(scene_ctx)[:, None, :]
        if base_query is None:
            hidden = self.mode_tokens.to(device=target_ctx.device, dtype=target_ctx.dtype)[None] + context
        else:
            hidden = base_query + context
        proto_memory = proto_embedding_weight + proto_proj(proto_summary_5d)
        proto_memory = proto_memory.to(device=target_ctx.device, dtype=target_ctx.dtype)
        proto_memory = proto_memory[None].expand(batch_size, -1, -1)
        proto_summary = proto_summary_5d.to(device=target_ctx.device, dtype=target_ctx.dtype)
        proto_attn_weights = None

        norm_index = 0
        for layer_index in range(2):
            attended, _ = self.self_attn[layer_index](hidden, hidden, hidden)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, proto_attn_weights = self.proto_attn[layer_index](hidden, proto_memory, proto_memory)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.agent_attn[layer_index](hidden, agent_feat, agent_feat, key_padding_mask=~obs_mask)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            hidden = self.norms[norm_index](hidden + self.ffn[layer_index](hidden))
            norm_index += 1

        if proto_attn_weights is None:
            proto_attn_weights = hidden.new_full(
                (batch_size, self.num_modes, proto_summary.size(0)),
                1.0 / max(proto_summary.size(0), 1),
            )
        endpoint_update = self.endpoint_head(hidden)
        coeff_update = self.coeff_head(hidden)
        score_update = self.score_head(hidden).squeeze(-1)
        gate_update = 0.25 * torch.tanh(self.gate_head(hidden)).squeeze(-1)
        if base_endpoint is None:
            proto_anchor = proto_attn_weights @ proto_summary[:, :3]
            endpoint_local = proto_anchor + endpoint_update
            coeff = coeff_update
            score = score_update
            gate = torch.sigmoid(self.gate_head(hidden)).squeeze(-1)
        else:
            endpoint_local = base_endpoint + endpoint_update
            coeff = base_coeff + coeff_update
            score = base_score + score_update
            gate = (base_gate + gate_update).clamp(0.0, 1.0)
        candidate_proto_idx = proto_attn_weights.argmax(dim=-1)
        query_out = base_query if base_query is not None else hidden
        return query_out, endpoint_local, coeff, score, gate, candidate_proto_idx


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


class BasisBridgeDecoder(nn.Module):
    def __init__(self, d_model=96, basis_dim=16, pred_len=120, num_control_points=16, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.num_control_points = int(num_control_points)
        stats_dim = 30
        input_dim = d_model + basis_dim + stats_dim
        self.control_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_control_points * 3),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.control_head[-1].weight)
        nn.init.zeros_(self.control_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        endpoint_envelope = 1.0 - torch.linspace(0.0, 1.0, steps=pred_len)
        self.register_buffer("endpoint_envelope", endpoint_envelope, persistent=False)

    def forward(self, query_feat, endpoint_local, coeff, coarse_local, anchor_local, basis_pinv, basis_bank):
        base_residual = coarse_local - anchor_local
        velocity = coarse_local[:, :, 1:] - coarse_local[:, :, :-1]
        start_velocity = velocity[:, :, 0]
        end_velocity = velocity[:, :, -1]
        mean_velocity = velocity.mean(dim=2)
        stats = torch.cat(
            [
                endpoint_local,
                coarse_local[:, :, 0],
                coarse_local[:, :, coarse_local.size(2) // 3],
                coarse_local[:, :, (2 * coarse_local.size(2)) // 3],
                coarse_local[:, :, -1],
                coarse_local.mean(dim=2),
                start_velocity,
                end_velocity,
                mean_velocity,
                base_residual.mean(dim=2),
            ],
            dim=-1,
        )
        bridge_input = torch.cat([query_feat, coeff, stats], dim=-1)
        batch_size, num_modes = coeff.shape[:2]
        control = self.control_head(bridge_input)
        control = control.reshape(batch_size * num_modes, self.num_control_points, 3).transpose(1, 2)
        residual_delta = F.interpolate(control, size=coarse_local.size(2), mode="linear", align_corners=True)
        residual_delta = residual_delta.transpose(1, 2).reshape(batch_size, num_modes, coarse_local.size(2), 3)
        envelope = self.endpoint_envelope.to(device=coarse_local.device, dtype=coarse_local.dtype)
        gate = 1.0 + 0.25 * torch.tanh(self.gate_head(bridge_input)).unsqueeze(-1)
        bridged_residual = base_residual + envelope[None, None, :, None] * gate * residual_delta
        flat_residual = bridged_residual.reshape(batch_size * num_modes, -1)
        bridge_coeff = flat_residual @ basis_pinv.to(device=coarse_local.device, dtype=coarse_local.dtype)
        bridge_coeff = bridge_coeff.reshape(batch_size, num_modes, -1)
        bridge_local = basis_bank(anchor_local, bridge_coeff)
        bridge_control_local = anchor_local + bridged_residual
        return bridge_coeff, bridge_local, bridge_control_local, gate.squeeze(-1).squeeze(-1)


class TemporalBasisDynamicsDecoder(nn.Module):
    def __init__(
        self,
        d_model=96,
        basis_dim=16,
        pred_len=120,
        num_control_points=24,
        nhead=4,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.num_control_points = int(num_control_points)
        self.endpoint_proj = nn.Linear(3, d_model)
        self.coeff_proj = nn.Linear(basis_dim, d_model)
        self.stats_proj = nn.Linear(12, d_model)
        self.control_tokens = nn.Parameter(torch.randn(self.num_control_points, d_model) * 0.02)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.time_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.agent_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.control_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        nn.init.zeros_(self.control_head[-1].weight)
        nn.init.zeros_(self.control_head[-1].bias)
        endpoint_envelope = 1.0 - torch.linspace(0.0, 1.0, steps=pred_len)
        self.register_buffer("endpoint_envelope", endpoint_envelope, persistent=False)

    def forward(
        self,
        query_feat,
        endpoint_local,
        coeff,
        coarse_local,
        target_temporal_feat,
        agent_feat,
        obs_mask,
        anchor_local,
        basis_pinv,
        basis_bank,
    ):
        coarse_end = coarse_local[:, :, -1]
        coarse_mean = coarse_local.mean(dim=2)
        endpoint_gap = coarse_end - endpoint_local
        stats = torch.cat([endpoint_local, coarse_end, coarse_mean, endpoint_gap], dim=-1)
        base_query = query_feat + self.endpoint_proj(endpoint_local) + self.coeff_proj(coeff) + self.stats_proj(stats)

        batch_size, num_modes, hidden_dim = base_query.shape
        control = base_query[:, :, None, :] + self.control_tokens.to(device=base_query.device, dtype=base_query.dtype)
        control = control.reshape(batch_size * num_modes, self.num_control_points, hidden_dim)

        attended, _ = self.self_attn(control, control, control)
        control = self.norms[0](control + attended)

        time_memory = target_temporal_feat[:, None].expand(-1, num_modes, -1, -1)
        time_memory = time_memory.reshape(batch_size * num_modes, target_temporal_feat.size(1), hidden_dim)
        attended, _ = self.time_attn(control, time_memory, time_memory)
        control = self.norms[1](control + attended)

        agent_memory = agent_feat[:, None].expand(-1, num_modes, -1, -1)
        agent_memory = agent_memory.reshape(batch_size * num_modes, agent_feat.size(1), hidden_dim)
        agent_mask = obs_mask[:, None].expand(-1, num_modes, -1).reshape(batch_size * num_modes, obs_mask.size(1))
        attended, _ = self.agent_attn(control, agent_memory, agent_memory, key_padding_mask=~agent_mask)
        control = self.norms[2](control + attended)
        control = self.norms[3](control + self.ffn(control))

        control_delta = self.control_head(control).transpose(1, 2)
        residual_delta = F.interpolate(control_delta, size=coarse_local.size(2), mode="linear", align_corners=True)
        residual_delta = residual_delta.transpose(1, 2).reshape(batch_size, num_modes, coarse_local.size(2), 3)
        envelope = self.endpoint_envelope.to(device=coarse_local.device, dtype=coarse_local.dtype)
        residual = coarse_local - anchor_local + envelope[None, None, :, None] * residual_delta
        flat_residual = residual.reshape(batch_size * num_modes, -1)
        dynamics_coeff = flat_residual @ basis_pinv.to(device=coarse_local.device, dtype=coarse_local.dtype)
        dynamics_coeff = dynamics_coeff.reshape(batch_size, num_modes, -1)
        dynamics_local = basis_bank(anchor_local, dynamics_coeff)
        return dynamics_coeff, dynamics_local


class PrototypeDynamicsRolloutDecoder(nn.Module):
    def __init__(
        self,
        d_model=96,
        basis_dim=16,
        pred_len=120,
        num_control_points=40,
        nhead=4,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.num_control_points = int(num_control_points)
        self.endpoint_proj = nn.Linear(3, d_model)
        self.coeff_proj = nn.Linear(basis_dim, d_model)
        self.stats_proj = nn.Linear(30, d_model)
        self.control_tokens = nn.Parameter(torch.randn(self.num_control_points, d_model) * 0.02)
        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.time_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.agent_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(8)])
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
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim // 2),
            nn.GELU(),
            nn.Linear(ff_dim // 2, 1),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        alpha = torch.linspace(1.0 / pred_len, 1.0, steps=pred_len)
        self.register_buffer("endpoint_alpha", alpha, persistent=False)

    def forward(
        self,
        query_feat,
        endpoint_local,
        coeff,
        coarse_local,
        target_temporal_feat,
        agent_feat,
        obs_mask,
        anchor_local,
        basis_pinv,
    ):
        coarse_velocity = torch.cat(
            [coarse_local[:, :, :1], coarse_local[:, :, 1:] - coarse_local[:, :, :-1]],
            dim=2,
        )
        start_velocity = coarse_velocity[:, :, 0]
        end_velocity = coarse_velocity[:, :, -1]
        mean_velocity = coarse_velocity.mean(dim=2)
        endpoint_gap = coarse_local[:, :, -1] - endpoint_local
        stats = torch.cat(
            [
                endpoint_local,
                coarse_local[:, :, 0],
                coarse_local[:, :, self.pred_len // 3],
                coarse_local[:, :, (2 * self.pred_len) // 3],
                coarse_local[:, :, -1],
                coarse_local.mean(dim=2),
                start_velocity,
                end_velocity,
                mean_velocity,
                endpoint_gap,
            ],
            dim=-1,
        )
        base_query = query_feat + self.endpoint_proj(endpoint_local) + self.coeff_proj(coeff) + self.stats_proj(stats)
        batch_size, num_modes, hidden_dim = base_query.shape
        hidden = base_query[:, :, None, :] + self.control_tokens.to(device=base_query.device, dtype=base_query.dtype)
        hidden = hidden.reshape(batch_size * num_modes, self.num_control_points, hidden_dim)

        time_memory = target_temporal_feat[:, None].expand(-1, num_modes, -1, -1)
        time_memory = time_memory.reshape(batch_size * num_modes, target_temporal_feat.size(1), hidden_dim)
        agent_memory = agent_feat[:, None].expand(-1, num_modes, -1, -1)
        agent_memory = agent_memory.reshape(batch_size * num_modes, agent_feat.size(1), hidden_dim)
        agent_mask = obs_mask[:, None].expand(-1, num_modes, -1).reshape(batch_size * num_modes, obs_mask.size(1))

        norm_index = 0
        for layer_index in range(2):
            attended, _ = self.self_attn[layer_index](hidden, hidden, hidden)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.time_attn[layer_index](hidden, time_memory, time_memory)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.agent_attn[layer_index](
                hidden,
                agent_memory,
                agent_memory,
                key_padding_mask=~agent_mask,
            )
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            hidden = self.norms[norm_index](hidden + self.ffn[layer_index](hidden))
            norm_index += 1

        velocity_control = self.velocity_head(hidden).transpose(1, 2)
        velocity_delta = F.interpolate(velocity_control, size=self.pred_len, mode="linear", align_corners=True)
        velocity_delta = velocity_delta.transpose(1, 2).reshape(batch_size, num_modes, self.pred_len, 3)
        gate = (0.25 + 0.25 * torch.tanh(self.gate_head(base_query))).unsqueeze(-1)
        velocity = coarse_velocity + gate * velocity_delta
        direct_local = velocity.cumsum(dim=2)
        alpha = self.endpoint_alpha.to(device=direct_local.device, dtype=direct_local.dtype)
        endpoint_error = direct_local[:, :, -1] - endpoint_local
        direct_local = direct_local - alpha[None, None, :, None] * endpoint_error[:, :, None, :]

        residual = direct_local - anchor_local
        flat_residual = residual.reshape(batch_size * num_modes, -1)
        direct_coeff = flat_residual @ basis_pinv.to(device=direct_local.device, dtype=direct_local.dtype)
        direct_coeff = direct_coeff.reshape(batch_size, num_modes, -1)
        return direct_coeff, direct_local, gate.squeeze(-1).squeeze(-1)


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


def _select_diverse_anchor_ids(proto_summary_5d, proto_frequency, num_modes):
    num_proto = int(proto_summary_5d.size(0))
    if num_proto == 0 or num_modes <= 0:
        return torch.empty(0, dtype=torch.long)
    endpoints = proto_summary_5d[:, :3].float()
    frequency = proto_frequency.float().clamp_min(1e-6)
    selected = [int(frequency.argmax().item())]
    min_dist = torch.cdist(endpoints[selected], endpoints).squeeze(0)
    freq_scale = (frequency / frequency.max().clamp_min(1e-6)).sqrt()
    while len(selected) < min(num_modes, num_proto):
        score = min_dist * freq_scale
        score[selected] = -1.0
        next_idx = int(score.argmax().item())
        selected.append(next_idx)
        min_dist = torch.minimum(min_dist, torch.cdist(endpoints[next_idx : next_idx + 1], endpoints).squeeze(0))
    while len(selected) < num_modes:
        selected.append(selected[len(selected) % len(selected)])
    return torch.tensor(selected, dtype=torch.long)


class AnchorSetTrajectoryDecoder(nn.Module):
    def __init__(self, d_model=96, basis_dim=16, num_modes=20, nhead=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.num_modes = int(num_modes)
        self.mode_tokens = nn.Parameter(torch.randn(self.num_modes, d_model) * 0.02)
        self.target_proj = nn.Linear(d_model, d_model)
        self.scene_proj = nn.Linear(d_model, d_model)
        self.anchor_summary_proj = nn.Linear(5, d_model)
        self.anchor_coeff_proj = nn.Linear(basis_dim, d_model)
        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
        )
        self.agent_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(2)]
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
        self.endpoint_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        self.coeff_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, basis_dim),
        )
        self.score_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim // 2), nn.GELU(), nn.Linear(ff_dim // 2, 1))
        self.gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim // 2), nn.GELU(), nn.Linear(ff_dim // 2, 1))
        nn.init.zeros_(self.endpoint_head[-1].weight)
        nn.init.zeros_(self.endpoint_head[-1].bias)
        nn.init.zeros_(self.coeff_head[-1].weight)
        nn.init.zeros_(self.coeff_head[-1].bias)

    def forward(self, target_ctx, scene_ctx, agent_feat, obs_mask, anchor_summary, anchor_coeff):
        batch_size = target_ctx.size(0)
        anchor_summary = anchor_summary.to(device=target_ctx.device, dtype=target_ctx.dtype)
        anchor_coeff = anchor_coeff.to(device=target_ctx.device, dtype=target_ctx.dtype)
        hidden = (
            self.mode_tokens.to(device=target_ctx.device, dtype=target_ctx.dtype)[None]
            + self.target_proj(target_ctx)[:, None, :]
            + self.scene_proj(scene_ctx)[:, None, :]
            + self.anchor_summary_proj(anchor_summary)[None]
            + self.anchor_coeff_proj(anchor_coeff)[None]
        )
        norm_index = 0
        for layer_index in range(2):
            attended, _ = self.self_attn[layer_index](hidden, hidden, hidden)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.agent_attn[layer_index](hidden, agent_feat, agent_feat, key_padding_mask=~obs_mask)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            hidden = self.norms[norm_index](hidden + self.ffn[layer_index](hidden))
            norm_index += 1

        endpoint = anchor_summary[None, :, :3] + self.endpoint_head(hidden)
        coeff = anchor_coeff[None, :, :] + self.coeff_head(hidden)
        score = self.score_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden)).squeeze(-1)
        return hidden, endpoint, coeff, score, gate


class IntentionTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        d_model=96,
        pred_len=120,
        num_modes=20,
        nhead=4,
        ff_dim=256,
        layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.num_modes = num_modes
        self.mode_tokens = nn.Parameter(torch.randn(num_modes, d_model) * 0.02)
        self.target_proj = nn.Linear(d_model, d_model)
        self.scene_proj = nn.Linear(d_model, d_model)
        self.velocity_proj = nn.Linear(3, d_model)
        self.time_emb = nn.Embedding(pred_len, d_model)
        self.register_buffer("time_ids", torch.arange(pred_len, dtype=torch.long), persistent=False)
        self.register_buffer(
            "step_ids",
            torch.arange(1, pred_len + 1, dtype=torch.float32).view(1, 1, pred_len, 1),
            persistent=False,
        )
        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(layers)]
        )
        self.target_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(layers)]
        )
        self.agent_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout) for _ in range(layers)]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d_model),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers * 4)])
        self.traj_norm = nn.LayerNorm(d_model)
        self.path_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 3),
        )
        self.endpoint_blend = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, target_ctx, scene_ctx, target_temporal_feat, agent_feat, obs_mask, target_local_obs):
        last_velocity = target_local_obs[:, -1] - target_local_obs[:, -2]
        hidden = (
            self.mode_tokens.to(device=target_ctx.device, dtype=target_ctx.dtype)[None]
            + self.target_proj(target_ctx)[:, None, :]
            + self.scene_proj(scene_ctx)[:, None, :]
            + self.velocity_proj(last_velocity)[:, None, :]
        )
        norm_index = 0
        for layer_index in range(len(self.self_attn)):
            attended, _ = self.self_attn[layer_index](hidden, hidden, hidden)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.target_attn[layer_index](hidden, target_temporal_feat, target_temporal_feat)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            attended, _ = self.agent_attn[layer_index](hidden, agent_feat, agent_feat, key_padding_mask=~obs_mask)
            hidden = self.norms[norm_index](hidden + attended)
            norm_index += 1

            hidden = self.norms[norm_index](hidden + self.ffn[layer_index](hidden))
            norm_index += 1

        time_feat = self.time_emb(self.time_ids[: self.pred_len]).to(dtype=hidden.dtype)
        traj_hidden = self.traj_norm(hidden[:, :, None, :] + time_feat[None, None, :, :])
        residual_path = self.path_head(traj_hidden)
        velocity_path = self.step_ids.to(device=hidden.device, dtype=hidden.dtype) * last_velocity[:, None, None, :]
        direct_path = velocity_path + residual_path

        endpoint = self.endpoint_head(hidden) + self.pred_len * last_velocity[:, None, :]
        alpha = torch.linspace(1.0 / self.pred_len, 1.0, steps=self.pred_len, device=hidden.device, dtype=hidden.dtype)
        endpoint_path = alpha[None, None, :, None] * endpoint[:, :, None, :]
        blend = self.endpoint_blend(hidden)[:, :, None, :]
        direct_path = blend * endpoint_path + (1.0 - blend) * direct_path
        score = self.score_head(hidden).squeeze(-1)
        return hidden, direct_path, score, endpoint


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
        basis_coeff_decoder=False,
        basis_coeff_mode="residual",
        coupled_decoder=False,
        coupled_decoder_iters=0,
        basis_bridge_decoder=False,
        bridge_control_points=16,
        temporal_dynamics_decoder=False,
        dynamics_control_points=24,
        direct_dynamics_decoder=False,
        direct_dynamics_control_points=40,
        soft_proto_decoder=False,
        soft_proto_modes=20,
        anchor_set_decoder=False,
        anchor_set_modes=20,
        intention_trajectory_decoder=False,
        intention_modes=20,
        intention_decoder_layers=3,
        tail_rescue_candidates=False,
        tail_rescue_extra_proto=5,
        tail_rescue_selection="oracle_train",
        tail_rescue_threshold=0.5,
        micro_endpoint_offsets=False,
        endpoint_shape_refiner=False,
        control_shape_refiner=False,
        control_shape_points=16,
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
        self.basis_coeff_decoder_enabled = bool(basis_coeff_decoder)
        self.coupled_decoder_enabled = bool(coupled_decoder) and int(coupled_decoder_iters or 0) > 0
        self.basis_bridge_decoder_enabled = bool(basis_bridge_decoder)
        self.temporal_dynamics_decoder_enabled = bool(temporal_dynamics_decoder)
        self.direct_dynamics_decoder_enabled = bool(direct_dynamics_decoder)
        self.soft_proto_decoder_enabled = bool(soft_proto_decoder)
        self.anchor_set_decoder_enabled = bool(anchor_set_decoder)
        self.intention_trajectory_decoder_enabled = bool(intention_trajectory_decoder)
        self.tail_rescue_candidates_enabled = bool(tail_rescue_candidates)
        self.tail_rescue_extra_proto = max(int(tail_rescue_extra_proto or 0), 0)
        if tail_rescue_selection not in {"gate", "oracle_train", "always"}:
            raise ValueError(f"Unsupported tail_rescue_selection: {tail_rescue_selection}")
        self.tail_rescue_selection = tail_rescue_selection
        self.tail_rescue_threshold = float(tail_rescue_threshold)
        self.micro_endpoint_offsets_enabled = bool(micro_endpoint_offsets)
        self.endpoint_shape_refiner_enabled = bool(endpoint_shape_refiner)
        self.control_shape_refiner_enabled = bool(control_shape_refiner)
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
        if self.soft_proto_decoder_enabled:
            self.soft_proto_decoder = SoftPrototypeSetDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                num_modes=soft_proto_modes,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        else:
            self.soft_proto_decoder = None
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
        anchor_set_ids = _select_diverse_anchor_ids(self.proto_summary_5d, self.proto_frequency, int(anchor_set_modes))
        self.register_buffer("anchor_set_ids", anchor_set_ids, persistent=False)
        if self.anchor_set_decoder_enabled:
            self.anchor_set_decoder = AnchorSetTrajectoryDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                num_modes=int(anchor_set_modes),
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        else:
            self.anchor_set_decoder = None
        if self.intention_trajectory_decoder_enabled:
            self.intention_trajectory_decoder = IntentionTrajectoryDecoder(
                d_model=d_model,
                pred_len=pred_len,
                num_modes=int(intention_modes),
                nhead=nhead,
                ff_dim=ff_dim,
                layers=int(intention_decoder_layers),
                dropout=dropout,
            )
        else:
            self.intention_trajectory_decoder = None
        if self.tail_rescue_candidates_enabled:
            rescue_endpoint_in = d_model * 3 + 5
            self.tail_rescue_endpoint_head = nn.Sequential(
                nn.LayerNorm(rescue_endpoint_in),
                nn.Linear(rescue_endpoint_in, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 3),
            )
            self.tail_rescue_gate_head = nn.Sequential(
                nn.LayerNorm(d_model * 2 + 4),
                nn.Linear(d_model * 2 + 4, ff_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim // 2, 1),
            )
            nn.init.zeros_(self.tail_rescue_endpoint_head[-1].weight)
            nn.init.zeros_(self.tail_rescue_endpoint_head[-1].bias)
            nn.init.zeros_(self.tail_rescue_gate_head[-1].weight)
            nn.init.zeros_(self.tail_rescue_gate_head[-1].bias)
        else:
            self.tail_rescue_endpoint_head = None
            self.tail_rescue_gate_head = None
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
        if self.basis_coeff_decoder_enabled:
            self.basis_coeff_decoder = BasisAwareCoeffDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                pred_len=pred_len,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
                mode=basis_coeff_mode,
            )
        else:
            self.basis_coeff_decoder = None
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
        if self.basis_bridge_decoder_enabled:
            self.basis_bridge_decoder = BasisBridgeDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                pred_len=pred_len,
                num_control_points=bridge_control_points,
                dropout=dropout,
            )
        else:
            self.basis_bridge_decoder = None
        if self.temporal_dynamics_decoder_enabled:
            self.temporal_dynamics_decoder = TemporalBasisDynamicsDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                pred_len=pred_len,
                num_control_points=dynamics_control_points,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        else:
            self.temporal_dynamics_decoder = None
        if self.direct_dynamics_decoder_enabled:
            self.direct_dynamics_decoder = PrototypeDynamicsRolloutDecoder(
                d_model=d_model,
                basis_dim=basis_dim,
                pred_len=pred_len,
                num_control_points=direct_dynamics_control_points,
                nhead=nhead,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        else:
            self.direct_dynamics_decoder = None
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

    def _decode_tail_rescue_candidates(self, proto_logits, target_ctx, scene_ctx, agent_feat, obs_mask):
        if self.tail_rescue_endpoint_head is None or self.tail_rescue_extra_proto <= 0:
            return None
        lookahead = min(self.topk_proto + self.tail_rescue_extra_proto, proto_logits.size(-1))
        if lookahead <= self.topk_proto:
            return None
        rescue_proto_idx = proto_logits.topk(lookahead, dim=-1).indices[:, self.topk_proto:lookahead]
        rescue_summary = self.proto_summary_5d[rescue_proto_idx]
        rescue_proto_token = self.prototype_router.proto_emb(rescue_proto_idx) + self.prototype_router.proto_proj(rescue_summary)
        target_rep = target_ctx[:, None, :].expand(-1, rescue_proto_idx.size(1), -1)
        scene_rep = scene_ctx[:, None, :].expand(-1, rescue_proto_idx.size(1), -1)
        endpoint_input = torch.cat([target_rep, scene_rep, rescue_proto_token, rescue_summary], dim=-1)
        rescue_endpoint_local = rescue_summary[:, :, :3] + self.tail_rescue_endpoint_head(endpoint_input)
        rescue_micro_coeff_anchor = self.micro_coeff_anchors[rescue_proto_idx] if self.has_micro_coeff_anchors else None
        rescue_query, rescue_coeff, rescue_score, rescue_gate, rescue_coeff_delta = self.query_decoder(
            rescue_proto_token,
            rescue_endpoint_local,
            target_ctx,
            agent_feat,
            obs_mask,
            micro_coeff_anchor=rescue_micro_coeff_anchor,
        )
        rescue_endpoint_mode = rescue_endpoint_local.repeat_interleave(self.n_micro, dim=1)
        if self.micro_endpoint_head is not None:
            rescue_endpoint_mode = rescue_endpoint_mode + self.micro_endpoint_head(rescue_query)
        rescue_anchor = build_anchor(rescue_endpoint_mode, self.anchor_alpha.to(rescue_endpoint_mode))
        rescue_coarse = self.basis_bank(rescue_anchor, rescue_coeff)
        rescue_candidate_proto = rescue_proto_idx.repeat_interleave(self.n_micro, dim=1)
        return {
            "query_feat": rescue_query,
            "coeff": rescue_coeff,
            "pred_score": rescue_score,
            "difficulty_gate": rescue_gate,
            "coeff_delta": rescue_coeff_delta,
            "endpoint_mode_local": rescue_endpoint_mode,
            "coarse_local": rescue_coarse,
            "candidate_proto_idx": rescue_candidate_proto,
        }

    def _tail_rescue_decision(self, proto_logits, target_ctx, scene_ctx, gt_proto_id=None):
        if self.tail_rescue_gate_head is None:
            return None, None, None
        prob = torch.softmax(proto_logits.float(), dim=-1).to(dtype=target_ctx.dtype)
        lookahead = min(self.topk_proto + self.tail_rescue_extra_proto, proto_logits.size(-1))
        top_values, natural_idx = prob.topk(lookahead, dim=-1)
        top1_prob = top_values[:, :1]
        kept_mass = top_values[:, : self.topk_proto].sum(dim=-1, keepdim=True)
        rescue_mass = top_values[:, self.topk_proto:].sum(dim=-1, keepdim=True) if lookahead > self.topk_proto else top1_prob * 0.0
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / max(math.log(max(proto_logits.size(-1), 2)), 1e-6)
        gate_input = torch.cat([target_ctx, scene_ctx, top1_prob, kept_mass, rescue_mass, entropy], dim=-1)
        gate_logit = self.tail_rescue_gate_head(gate_input).squeeze(-1)
        gate_active = torch.sigmoid(gate_logit) > self.tail_rescue_threshold
        target = None
        if gt_proto_id is not None:
            natural_top = natural_idx[:, : self.topk_proto]
            target = (~natural_top.eq(gt_proto_id[:, None]).any(dim=1)).float()
        if self.tail_rescue_selection == "always":
            active = torch.ones_like(gate_active)
        elif self.tail_rescue_selection == "oracle_train" and self.training and target is not None:
            active = target.bool()
        else:
            active = gate_active
        return active, gate_logit, target

    def _apply_tail_rescue_selection(
        self,
        base_tensors,
        rescue_tensors,
        rescue_active,
    ):
        if rescue_tensors is None or rescue_active is None or self.n_micro <= 0:
            return base_tensors, False
        if self.candidate_keep_indices.numel() == 0:
            return base_tensors, False
        default_keep = self.candidate_keep_indices.to(device=base_tensors["coeff"].device)
        base_first = torch.arange(self.topk_proto, device=base_tensors["coeff"].device) * self.n_micro
        rescue_proto_count = rescue_tensors["candidate_proto_idx"].size(1) // self.n_micro
        rescue_first = torch.arange(rescue_proto_count, device=base_tensors["coeff"].device) * self.n_micro
        if base_first.numel() + rescue_first.numel() != default_keep.numel():
            return base_tensors, False

        active_float = rescue_active.to(device=base_tensors["coeff"].device).view(-1, 1)
        active_bool = rescue_active.to(device=base_tensors["coeff"].device).view(-1, 1)
        selected = {}
        for key, tensor in base_tensors.items():
            default_value = tensor.index_select(1, default_keep)
            rescue_base = tensor.index_select(1, base_first)
            rescue_value = torch.cat([rescue_base, rescue_tensors[key].index_select(1, rescue_first)], dim=1)
            if tensor.dtype == torch.bool:
                mask = active_bool
            elif tensor.is_floating_point():
                mask = active_float
                while mask.dim() < default_value.dim():
                    mask = mask.unsqueeze(-1)
            else:
                mask = active_bool
            if tensor.is_floating_point():
                selected[key] = torch.where(mask.bool(), rescue_value, default_value)
            else:
                selected[key] = torch.where(mask, rescue_value, default_value)
        return selected, True

    def forward(self, obs_xyz, obs_mask, gt_proto_id=None, force_gt_proto=False, enable_refiner=True):
        local_xyz, _, _, origin, rotation = self.pose_normalizer(obs_xyz)
        feats_local = build_local_features(local_xyz)
        feats_global = build_global_features(obs_xyz)

        agent_feat, target_temporal_feat = self.temporal_encoder(
            feats_local,
            feats_global,
            return_target_sequence=True,
        )
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

        candidate_proto_idx = None
        candidate_selection_applied = False
        tail_rescue_active = None
        tail_rescue_logit = None
        tail_rescue_target = None

        if self.intention_trajectory_decoder is not None:
            active_query, direct_local, pred_score, endpoint_mode_local = self.intention_trajectory_decoder(
                target_ctx,
                scene_ctx,
                target_temporal_feat,
                agent_feat,
                obs_mask,
                local_xyz[:, 0],
            )
            batch_size, num_modes = direct_local.shape[:2]
            coeff = direct_local.new_zeros(batch_size, num_modes, self.basis_bank.basis_bank.size(0))
            pred_xyz = self.pose_normalizer.inverse(direct_local, origin, rotation)
            aux = {
                "coeff": coeff,
                "coeff_delta": coeff,
                "endpoint_residual": endpoint_residual,
                "endpoint_mode_local": endpoint_mode_local,
                "coarse_local": direct_local,
                "direct_dynamics_local": direct_local,
                "candidate_proto_idx": None,
                "basis_matrix": self.basis_matrix_flat,
                "basis_pinv": self.basis_pinv,
                "proto_frequency": self.proto_frequency,
                "proto_summary_5d": self.proto_summary_5d,
            }
            return {
                "pred_xyz": pred_xyz,
                "pred_score": pred_score,
                "proto_logits": proto_logits,
                "top_proto_idx": top_proto_idx,
                "aux": aux,
            }

        if self.anchor_set_decoder is not None:
            anchor_ids = self.anchor_set_ids.to(device=target_ctx.device)
            anchor_summary = self.proto_summary_5d.index_select(0, anchor_ids)
            if self.has_micro_coeff_anchors:
                anchor_coeff = self.micro_coeff_anchors.index_select(0, anchor_ids)[:, 0]
            else:
                anchor_coeff = torch.zeros(
                    anchor_ids.size(0),
                    self.basis_bank.basis_bank.size(0),
                    device=target_ctx.device,
                    dtype=target_ctx.dtype,
                )
            query_feat, endpoint_mode_local, coeff, pred_score, difficulty_gate = self.anchor_set_decoder(
                target_ctx,
                scene_ctx,
                agent_feat,
                obs_mask,
                anchor_summary,
                anchor_coeff,
            )
            candidate_proto_idx = anchor_ids[None, :].expand(target_ctx.size(0), -1)
            coeff_delta = coeff - anchor_coeff.to(device=coeff.device, dtype=coeff.dtype)[None]
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coarse_local = self.basis_bank(anchor_local, coeff)
            active_query = query_feat
        else:
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
            candidate_proto_idx = top_proto_idx.repeat_interleave(self.n_micro, dim=1)
            if self.tail_rescue_candidates_enabled:
                rescue_tensors = self._decode_tail_rescue_candidates(
                    proto_logits,
                    target_ctx,
                    scene_ctx,
                    agent_feat,
                    obs_mask,
                )
                tail_rescue_active, tail_rescue_logit, tail_rescue_target = self._tail_rescue_decision(
                    proto_logits,
                    target_ctx,
                    scene_ctx,
                    gt_proto_id=gt_proto_id,
                )
                base_tensors = {
                    "query_feat": query_feat,
                    "coeff": coeff,
                    "pred_score": pred_score,
                    "difficulty_gate": difficulty_gate,
                    "coeff_delta": coeff_delta,
                    "endpoint_mode_local": endpoint_mode_local,
                    "coarse_local": coarse_local,
                    "candidate_proto_idx": candidate_proto_idx,
                }
                selected_tensors, candidate_selection_applied = self._apply_tail_rescue_selection(
                    base_tensors,
                    rescue_tensors,
                    tail_rescue_active,
                )
                if candidate_selection_applied:
                    query_feat = selected_tensors["query_feat"]
                    coeff = selected_tensors["coeff"]
                    pred_score = selected_tensors["pred_score"]
                    difficulty_gate = selected_tensors["difficulty_gate"]
                    coeff_delta = selected_tensors["coeff_delta"]
                    endpoint_mode_local = selected_tensors["endpoint_mode_local"]
                    coarse_local = selected_tensors["coarse_local"]
                    candidate_proto_idx = selected_tensors["candidate_proto_idx"]
                    active_query = query_feat
        if self.anchor_set_decoder is not None:
            micro_endpoint_delta = None
        basis_coeff = None
        if self.basis_coeff_decoder is not None:
            coeff_before_basis = coeff
            coeff = self.basis_coeff_decoder(
                query_feat,
                endpoint_mode_local,
                coeff,
                coarse_local,
                self.basis_bank.basis_bank,
            )
            basis_coeff = coeff
            coeff_delta = coeff_delta + (coeff - coeff_before_basis)
            coarse_local = self.basis_bank(anchor_local, coeff)
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
        if self.anchor_set_decoder is None and candidate_proto_idx is None:
            candidate_proto_idx = top_proto_idx.repeat_interleave(self.n_micro, dim=1)
        if (
            self.anchor_set_decoder is None
            and self.candidate_keep_indices.numel() > 0
            and not candidate_selection_applied
        ):
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
        soft_proto_idx = None
        if self.soft_proto_decoder is not None:
            coeff_before_soft = coeff
            active_query, endpoint_mode_local, coeff, pred_score, difficulty_gate, soft_proto_idx = self.soft_proto_decoder(
                target_ctx,
                scene_ctx,
                agent_feat,
                obs_mask,
                self.proto_summary_5d,
                self.prototype_router.proto_emb.weight,
                self.prototype_router.proto_proj,
                base_query=active_query,
                base_endpoint=endpoint_mode_local,
                base_coeff=coeff,
                base_score=pred_score,
                base_gate=difficulty_gate,
            )
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coarse_local = self.basis_bank(anchor_local, coeff)
            coeff_delta = coeff_delta + (coeff - coeff_before_soft)
        bridge_local = None
        bridge_control_local = None
        bridge_coeff = None
        bridge_gate = None
        dynamics_local = None
        dynamics_coeff = None
        direct_dynamics_local = None
        direct_dynamics_coeff = None
        direct_dynamics_gate = None
        if self.basis_bridge_decoder is not None:
            coeff_before_bridge = coeff
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coeff, coarse_local, bridge_control_local, bridge_gate = self.basis_bridge_decoder(
                active_query,
                endpoint_mode_local,
                coeff,
                coarse_local,
                anchor_local,
                self.basis_pinv,
                self.basis_bank,
            )
            bridge_coeff = coeff
            bridge_local = coarse_local
            coeff_delta = coeff_delta + (coeff - coeff_before_bridge)
        if self.temporal_dynamics_decoder is not None:
            coeff_before_dynamics = coeff
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coeff, coarse_local = self.temporal_dynamics_decoder(
                active_query,
                endpoint_mode_local,
                coeff,
                coarse_local,
                target_temporal_feat,
                agent_feat,
                obs_mask,
                anchor_local,
                self.basis_pinv,
                self.basis_bank,
            )
            dynamics_coeff = coeff
            dynamics_local = coarse_local
            coeff_delta = coeff_delta + (coeff - coeff_before_dynamics)
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
        if self.direct_dynamics_decoder is not None:
            coeff_before_direct = coeff
            anchor_local = build_anchor(endpoint_mode_local, self.anchor_alpha.to(endpoint_mode_local))
            coeff, coarse_local, direct_dynamics_gate = self.direct_dynamics_decoder(
                active_query,
                endpoint_mode_local,
                coeff,
                coarse_local,
                target_temporal_feat,
                agent_feat,
                obs_mask,
                anchor_local,
                self.basis_pinv,
            )
            direct_dynamics_coeff = coeff
            direct_dynamics_local = coarse_local
            coeff_delta = coeff_delta + (coeff - coeff_before_direct)
        use_refiner = enable_refiner and (not self.disable_refiner)
        refined_local = self.refiner(coarse_local, active_query, difficulty_gate) if use_refiner else coarse_local
        if self.endpoint_shape_refiner is not None:
            refined_local = self.endpoint_shape_refiner(refined_local, active_query)
        if self.control_shape_refiner is not None:
            refined_local = self.control_shape_refiner(refined_local, active_query)

        pred_xyz = self.pose_normalizer.inverse(refined_local, origin, rotation)

        aux = {
            "coeff": coeff,
            "coeff_delta": coeff_delta,
            "basis_coeff": basis_coeff,
            "endpoint_residual": endpoint_residual,
            "endpoint_mode_local": endpoint_mode_local,
            "coarse_local": coarse_local,
            "candidate_proto_idx": candidate_proto_idx,
            "basis_matrix": self.basis_matrix_flat,
            "basis_pinv": self.basis_pinv,
            "proto_frequency": self.proto_frequency,
            "proto_summary_5d": self.proto_summary_5d,
        }
        if tail_rescue_logit is not None:
            aux["tail_rescue_logit"] = tail_rescue_logit
        if tail_rescue_active is not None:
            aux["tail_rescue_active"] = tail_rescue_active.float()
        if tail_rescue_target is not None:
            aux["tail_rescue_target"] = tail_rescue_target
        if micro_endpoint_delta is not None:
            aux["micro_endpoint_delta"] = micro_endpoint_delta
        if soft_proto_idx is not None:
            aux["soft_proto_idx"] = soft_proto_idx
        if bridge_local is not None:
            aux.update(
                {
                    "bridge_coeff": bridge_coeff,
                    "bridge_local": bridge_local,
                    "bridge_control_local": bridge_control_local,
                    "bridge_gate": bridge_gate,
                }
            )
        if dynamics_local is not None:
            aux.update(
                {
                    "dynamics_coeff": dynamics_coeff,
                    "dynamics_local": dynamics_local,
                }
            )
        if direct_dynamics_local is not None:
            aux.update(
                {
                    "direct_dynamics_coeff": direct_dynamics_coeff,
                    "direct_dynamics_local": direct_dynamics_local,
                    "direct_dynamics_gate": direct_dynamics_gate,
                }
            )

        return {
            "pred_xyz": pred_xyz,
            "pred_score": pred_score,
            "proto_logits": proto_logits,
            "top_proto_idx": top_proto_idx,
            "aux": aux,
        }


ProtoBasisFlight = ProtoBasisNet
