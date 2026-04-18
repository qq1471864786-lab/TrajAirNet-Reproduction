from typing import Optional

import torch
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
        self.local_proj = nn.Linear(8, 48)
        self.global_proj = nn.Linear(7, 48)
        self.time_emb = nn.Embedding(obs_len, d_model)
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
        time_ids = torch.arange(obs_len, device=hidden.device)
        hidden = hidden + self.time_emb(time_ids)[None, None, :, :]
        hidden = hidden.view(batch_size * num_agents, obs_len, -1)
        hidden = self.encoder(hidden)
        pooled = hidden.max(dim=1).values
        last = hidden[:, -1]
        out = self.pool_proj(torch.cat([pooled, last], dim=-1))
        return out.view(batch_size, num_agents, -1)


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
    def __init__(self, n_proto=64, d_model=96, topk=5, dropout=0.1):
        super().__init__()
        self.topk = topk
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
        self.generic_proto = nn.Parameter(torch.randn(topk, d_model))
        self.generic_endpoint_head = nn.Linear(d_model, topk * 3)

    def forward(self, target_ctx, scene_ctx, proto_summary_5d, gt_proto_id=None, force_gt_proto=False, disable_router=False):
        hidden = self.trunk(torch.cat([target_ctx, scene_ctx], dim=-1))
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
            proto_token = self.proto_emb(top_idx) + self.proto_proj(proto_summary_5d[top_idx])
            endpoint_residual = self.endpoint_head(hidden).view(hidden.size(0), self.topk, 3)
            endpoint_local = proto_summary_5d[top_idx, :3] + endpoint_residual
        return logits, top_idx, proto_token, endpoint_residual, endpoint_local


class PrototypeConditionedQueryDecoder(nn.Module):
    def __init__(self, d_model=96, n_micro=4, basis_dim=16, dropout=0.1):
        super().__init__()
        self.n_micro = n_micro
        self.endpoint_proj = nn.Linear(3, d_model)
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
                    nn.Linear(d_model, 192),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(192, d_model),
                )
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(6)])

        self.coeff_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, basis_dim))
        self.score_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, proto_token, endpoint_local, target_ctx, agent_feat, obs_mask):
        batch_size = proto_token.size(0)
        endpoint_token = self.endpoint_proj(endpoint_local)
        hidden = (
            proto_token[:, :, None, :]
            + endpoint_token[:, :, None, :]
            + self.micro[None, None, :, :]
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

        coeff = self.coeff_head(hidden)
        score = self.score_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden)).squeeze(-1)
        return hidden, coeff, score, gate


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


def build_anchor(endpoint_local, pred_len):
    alpha = torch.linspace(0.0, 1.0, steps=pred_len, device=endpoint_local.device, dtype=endpoint_local.dtype)
    return alpha[None, None, :, None] * endpoint_local[:, :, None, :]


class ProtoBasisFlight(nn.Module):
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
        dropout=0.1,
        proto_summary_5d: Optional[torch.Tensor] = None,
        proto_frequency: Optional[torch.Tensor] = None,
        basis_bank: Optional[torch.Tensor] = None,
        disable_social=False,
        disable_router=False,
        disable_refiner=False,
    ):
        super().__init__()
        if proto_summary_5d is None:
            raise ValueError("proto_summary_5d is required for ProtoBasis-Flight.")
        if basis_bank is None:
            raise ValueError("basis_bank is required for ProtoBasis-Flight.")
        if proto_frequency is None:
            proto_frequency = torch.ones(proto_summary_5d.size(0), dtype=torch.float32)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.topk_proto = topk_proto
        self.n_micro = n_micro
        self.num_modes = topk_proto * n_micro
        self.disable_social = disable_social
        self.disable_router = disable_router
        self.disable_refiner = disable_refiner

        self.pose_normalizer = PoseNormalizer()
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
        )
        self.query_decoder = PrototypeConditionedQueryDecoder(
            d_model=d_model,
            n_micro=n_micro,
            basis_dim=basis_dim,
            dropout=dropout,
        )
        self.basis_bank = BasisBank(basis_bank)
        self.refiner = TemporalResidualRefiner(d_model=d_model)

        self.register_buffer("proto_summary_5d", proto_summary_5d.float())
        self.register_buffer("proto_frequency", proto_frequency.float())

    def forward(self, obs_xyz, obs_mask, gt_proto_id=None, force_gt_proto=False, enable_refiner=True):
        local_xyz, yaw, pitch, origin, rotation = self.pose_normalizer(obs_xyz)
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

        query_feat, coeff, pred_score, difficulty_gate = self.query_decoder(
            proto_token,
            endpoint_local,
            target_ctx,
            agent_feat,
            obs_mask,
        )
        endpoint_mode_local = endpoint_local.repeat_interleave(self.n_micro, dim=1)
        anchor_local = build_anchor(endpoint_mode_local, pred_len=self.pred_len)
        coarse_local = self.basis_bank(anchor_local, coeff)
        use_refiner = enable_refiner and (not self.disable_refiner)
        refined_local = self.refiner(coarse_local, query_feat, difficulty_gate) if use_refiner else coarse_local

        pred_xyz = self.pose_normalizer.inverse(refined_local, origin, rotation)
        coarse_xyz = self.pose_normalizer.inverse(coarse_local, origin, rotation)

        return {
            "pred_xyz": pred_xyz,
            "pred_score": pred_score,
            "proto_logits": proto_logits,
            "top_proto_idx": top_proto_idx,
            "aux": {
                "query_feat": query_feat,
                "coeff": coeff,
                "difficulty_gate": difficulty_gate,
                "endpoint_residual": endpoint_residual,
                "endpoint_local": endpoint_local,
                "endpoint_mode_local": endpoint_mode_local,
                "anchor_local": anchor_local,
                "coarse_local_xyz": coarse_local,
                "coarse_xyz": coarse_xyz,
                "refined_local_xyz": refined_local,
                "target_yaw": yaw,
                "target_pitch": pitch,
                "origin": origin,
                "used_refiner": use_refiner,
            },
        }
