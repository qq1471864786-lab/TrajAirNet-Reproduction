"""
TrajAir rebuild model.

Core idea:
  - keep a TrajAirNet-style CVAE + best-of-5 evaluation path
  - replace the old altitude branch patch with
    vertical-state-conditioned relation gating
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for index, out_channels in enumerate(num_channels):
            dilation = 2 ** index
            in_channels = num_inputs if index == 0 else num_channels[index - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CVAEEncoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()
        self.conditional = conditional
        adjusted_sizes = list(layer_sizes)
        if self.conditional:
            adjusted_sizes[0] += num_labels

        layers = []
        for in_size, out_size in zip(adjusted_sizes[:-1], adjusted_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.linear_means = nn.Linear(adjusted_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(adjusted_sizes[-1], latent_size)

    def forward(self, x, condition=None):
        if self.conditional:
            x = torch.cat((x, condition), dim=-1)
        hidden = self.mlp(x)
        return self.linear_means(hidden), self.linear_log_var(hidden)


class CVAEDecoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()
        self.conditional = conditional
        input_size = latent_size + num_labels if conditional else latent_size
        layers = []
        current_size = input_size
        for index, out_size in enumerate(layer_sizes):
            layers.append(nn.Linear(current_size, out_size))
            if index < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())
            current_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent, condition=None):
        if self.conditional:
            latent = torch.cat((latent, condition), dim=-1)
        return self.mlp(latent)


class CVAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditional=True, num_labels=0):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = CVAEEncoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = CVAEDecoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition=None):
        means, log_var = self.encoder(x, condition)
        latent = self.reparameterize(means, log_var)
        recon = self.decoder(latent, condition)
        return recon, means, log_var, latent

    def inference(self, latent, condition=None):
        return self.decoder(latent, condition)


def first_difference(sequence):
    if sequence.size(-1) <= 1:
        return torch.zeros_like(sequence)
    diff = sequence[:, :, 1:] - sequence[:, :, :-1]
    return torch.cat([diff[:, :, :1], diff], dim=-1)


def build_motion_channels(obs_input):
    velocity = first_difference(obs_input)
    return torch.cat([obs_input, velocity], dim=1)


def build_vertical_channels(obs_input):
    altitude = obs_input[:, 2:3, :]
    vertical_velocity = first_difference(altitude)
    vertical_trend = altitude - altitude[:, :, :1]
    return torch.cat([altitude, vertical_velocity, vertical_trend], dim=1)


def future_to_deltas(target, last_observation):
    future = target.permute(1, 0, 2)
    deltas = torch.empty_like(future)
    deltas[:, 0] = future[:, 0] - last_observation
    if future.size(1) > 1:
        deltas[:, 1:] = future[:, 1:] - future[:, :-1]
    return deltas


def deltas_to_positions(deltas, last_observation):
    cumulative = torch.cumsum(deltas, dim=1)
    prediction = last_observation.unsqueeze(1) + cumulative
    return prediction.permute(1, 0, 2)


class SequenceEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size=4, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_channels,
            num_channels=[hidden_dim, hidden_dim],
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence):
        encoded = self.tcn(sequence)
        pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=-1)], dim=-1)
        return self.proj(pooled)


class ContextEncoder(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, context):
        encoded = torch.relu(self.conv(context))
        pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=-1)], dim=-1)
        return self.proj(pooled)


class VerticalStateEncoder(nn.Module):
    def __init__(self, hidden_dim=128, kernel_size=3, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=3,
            num_channels=[hidden_dim, hidden_dim],
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, vertical_channels):
        encoded = self.tcn(vertical_channels)
        pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=-1)], dim=-1)
        state_feature = self.proj(pooled)
        summary = torch.stack(
            [
                vertical_channels[:, 0, -1],
                vertical_channels[:, 1, -1],
                vertical_channels[:, 2, -1],
            ],
            dim=-1,
        )
        return state_feature, summary


class RelationAwareInteractionBlock(nn.Module):
    def __init__(self, agent_dim, out_dim, n_heads=8, dropout=0.05, conditioned=False, neighbor_topk=3):
        super().__init__()
        self.conditioned = conditioned
        self.neighbor_topk = neighbor_topk
        if out_dim % n_heads != 0:
            raise ValueError("out_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.feature_proj = nn.Linear(agent_dim, out_dim, bias=False)
        self.att_vector = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.att_vector.unsqueeze(0))
        edge_dim = 5 + (3 if conditioned else 0)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_heads),
        )
        self.edge_gate = None
        if conditioned:
            self.edge_gate = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, out_dim),
                nn.Sigmoid(),
            )
        self.distance_scale = nn.Parameter(torch.tensor(0.2))
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        edge_norm = [5.0, 2.0, 5.0, 0.1, 0.1]
        if conditioned:
            edge_norm.extend([0.4, 0.05, 0.4])
        self.register_buffer("edge_norm", torch.tensor(edge_norm, dtype=torch.float32))

    def _adjacency_mask(self, scene_ids):
        same_scene = scene_ids.unsqueeze(0) == scene_ids.unsqueeze(1)
        not_self = ~torch.eye(scene_ids.size(0), dtype=torch.bool, device=scene_ids.device)
        return same_scene & not_self

    def _edge_features(self, last_pos, last_vel, state_summary=None):
        pos_i = last_pos.unsqueeze(1)
        pos_j = last_pos.unsqueeze(0)
        vel_i = last_vel.unsqueeze(1)
        vel_j = last_vel.unsqueeze(0)
        dx = pos_j[..., 0] - pos_i[..., 0]
        dy = pos_j[..., 1] - pos_i[..., 1]
        horizontal_distance = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-8)
        dvx = vel_j[..., 0] - vel_i[..., 0]
        dvy = vel_j[..., 1] - vel_i[..., 1]
        base_edge = torch.stack([dx, dy, horizontal_distance, dvx, dvy], dim=-1)
        vertical_edge = None
        if self.conditioned:
            if state_summary is None:
                raise ValueError("state_summary is required when conditioned=True")
            state_i = state_summary.unsqueeze(1)
            state_j = state_summary.unsqueeze(0)
            dz = state_j[..., 0] - state_i[..., 0]
            dvz = state_j[..., 1] - state_i[..., 1]
            dtrend = state_j[..., 2] - state_i[..., 2]
            vertical_edge = torch.stack([dz, dvz, dtrend], dim=-1)
        return base_edge, vertical_edge, horizontal_distance

    def _neighbor_mask(self, adjacency, horizontal_distance, vertical_edge=None):
        if self.neighbor_topk <= 0:
            return adjacency

        neighbor_mask = torch.zeros_like(adjacency)
        for index in range(adjacency.size(0)):
            valid = torch.nonzero(adjacency[index], as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                continue
            if valid.numel() <= self.neighbor_topk:
                neighbor_mask[index, valid] = True
                continue
            distances = horizontal_distance[index, valid]
            if vertical_edge is not None:
                distances = distances + 0.5 * vertical_edge[index, valid, 0].abs()
            chosen = valid[torch.topk(distances, k=self.neighbor_topk, largest=False).indices]
            neighbor_mask[index, chosen] = True
        return adjacency & neighbor_mask

    def forward(self, agent_feature, last_pos, last_vel, scene_ids, state_summary=None):
        agent_count = agent_feature.size(0)
        if scene_ids is None or agent_count <= 1:
            return torch.zeros(agent_count, self.output_proj.out_features, device=agent_feature.device)

        adjacency = self._adjacency_mask(scene_ids)
        feature = self.feature_proj(agent_feature).view(agent_count, self.n_heads, self.head_dim)

        base_edge, vertical_edge, horizontal_distance = self._edge_features(
            last_pos,
            last_vel,
            state_summary=state_summary,
        )
        adjacency = self._neighbor_mask(adjacency, horizontal_distance, vertical_edge=vertical_edge)
        edge_input = base_edge if vertical_edge is None else torch.cat([base_edge, vertical_edge], dim=-1)
        edge_input = edge_input / self.edge_norm.view(1, 1, -1)

        feature_i = feature.unsqueeze(1).expand(agent_count, agent_count, self.n_heads, self.head_dim)
        feature_j = feature.unsqueeze(0).expand(agent_count, agent_count, self.n_heads, self.head_dim)
        pair_feature = torch.cat([feature_i, feature_j], dim=-1)
        scores = (pair_feature * self.att_vector.view(1, 1, self.n_heads, -1)).sum(dim=-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        scores = scores + self.edge_bias(edge_input)
        scores = scores - F.softplus(self.distance_scale) * torch.log1p(horizontal_distance).unsqueeze(-1)
        scores = scores.masked_fill(~adjacency.unsqueeze(-1), -1e9)
        attention = F.softmax(scores, dim=1)
        attention = attention * adjacency.unsqueeze(-1).float()
        normalizer = attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attention = self.dropout(attention / normalizer)

        if self.edge_gate is not None:
            gate = self.edge_gate(edge_input).view(agent_count, agent_count, self.n_heads, self.head_dim)
            feature_j = feature_j * gate

        messages = (attention.unsqueeze(-1) * feature_j).sum(dim=1).reshape(agent_count, -1)
        return self.output_proj(messages)


class VerticalRelationTrajectoryModel(nn.Module):
    def __init__(
        self,
        variant="base",
        obs_len=11,
        pred_len=120,
        pred_step=10,
        traj_hidden=256,
        context_hidden=32,
        state_hidden=128,
        interaction_hidden=256,
        interaction_heads=8,
        interaction_topk=3,
        tcn_kernel=4,
        dropout=0.1,
        cvae_latent=128,
        cvae_layers=2,
        cvae_channel_size=128,
        condition_dropout=0.0,
        state_scale_init=None,
        social_scale_init=None,
        interaction_integration="concat",
    ):
        super().__init__()
        valid_variants = {"base", "interaction", "state", "full", "naive"}
        if variant not in valid_variants:
            raise ValueError(f"Unsupported variant: {variant}")
        valid_integrations = {"concat", "residualgate"}
        if interaction_integration not in valid_integrations:
            raise ValueError(f"Unsupported interaction integration: {interaction_integration}")

        self.variant = variant
        self.obs_len = obs_len
        self.pred_steps = int(math.ceil(pred_len / pred_step))
        self.condition_dropout = condition_dropout
        self.interaction_integration = interaction_integration

        self.traj_encoder = SequenceEncoder(
            input_channels=6,
            hidden_dim=traj_hidden,
            kernel_size=tcn_kernel,
            dropout=dropout,
        )
        self.context_encoder = ContextEncoder(hidden_dim=context_hidden, dropout=dropout)
        self.base_dim = traj_hidden + context_hidden

        self.state_encoder = None
        self.state_scale = None
        self.state_condition_proj = None
        self.state_condition_dim = 0
        if variant in {"state", "full", "naive"}:
            self.state_encoder = VerticalStateEncoder(
                hidden_dim=state_hidden,
                kernel_size=3,
                dropout=dropout,
            )
            self.state_condition_proj = nn.Sequential(
                nn.Linear(state_hidden, state_hidden),
                nn.LayerNorm(state_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            default_state_scale = 0.15 if variant == "full" else 0.25
            if state_scale_init is None:
                state_scale_init = default_state_scale
            self.state_scale = nn.Parameter(torch.tensor(state_scale_init))
            self.state_condition_dim = state_hidden

        self.interaction_block = None
        self.social_scale = None
        self.social_condition_proj = None
        self.social_condition_dim = 0
        self.social_gate = None
        self.social_update = None
        if variant in {"interaction", "naive"}:
            self.interaction_block = RelationAwareInteractionBlock(
                agent_dim=self.base_dim,
                out_dim=interaction_hidden,
                n_heads=interaction_heads,
                dropout=dropout,
                conditioned=False,
                neighbor_topk=interaction_topk,
            )
            if interaction_integration == "concat":
                self.social_condition_proj = nn.Sequential(
                    nn.Linear(interaction_hidden, interaction_hidden),
                    nn.LayerNorm(interaction_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.social_condition_dim = interaction_hidden
            else:
                self.social_gate = nn.Sequential(
                    nn.Linear(self.base_dim + interaction_hidden, self.base_dim),
                    nn.Sigmoid(),
                )
                self.social_update = nn.Sequential(
                    nn.Linear(interaction_hidden, self.base_dim),
                    nn.LayerNorm(self.base_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            if social_scale_init is None:
                social_scale_init = 0.10
            self.social_scale = nn.Parameter(torch.tensor(social_scale_init))

        self.full_interaction_block = None
        if variant == "full":
            self.full_interaction_block = RelationAwareInteractionBlock(
                agent_dim=self.base_dim,
                out_dim=interaction_hidden,
                n_heads=interaction_heads,
                dropout=dropout,
                conditioned=True,
                neighbor_topk=interaction_topk,
            )
            if interaction_integration == "concat":
                self.social_condition_proj = nn.Sequential(
                    nn.Linear(interaction_hidden, interaction_hidden),
                    nn.LayerNorm(interaction_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.social_condition_dim = interaction_hidden
            else:
                self.social_gate = nn.Sequential(
                    nn.Linear(self.base_dim + interaction_hidden, self.base_dim),
                    nn.Sigmoid(),
                )
                self.social_update = nn.Sequential(
                    nn.Linear(interaction_hidden, self.base_dim),
                    nn.LayerNorm(self.base_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            if social_scale_init is None:
                social_scale_init = 0.10
            self.social_scale = nn.Parameter(torch.tensor(social_scale_init))

        self.condition_dim = self.base_dim + self.state_condition_dim + self.social_condition_dim

        encoder_sizes = [self.pred_steps * 3] + [cvae_channel_size] * cvae_layers
        decoder_sizes = [cvae_channel_size] * (cvae_layers + 1)
        self.cvae = CVAE(
            encoder_layer_sizes=encoder_sizes,
            latent_size=cvae_latent,
            decoder_layer_sizes=decoder_sizes,
            conditional=True,
            num_labels=self.condition_dim,
        )
        self.delta_decoder = nn.Linear(cvae_channel_size, self.pred_steps * 3)

    def _sample_latent(self, agent_count, scene_slices, device, latent_mode):
        if latent_mode not in {"sample", "zero"}:
            raise ValueError(f"Unsupported latent_mode: {latent_mode}")
        if scene_slices is None or scene_slices.numel() == 0:
            if latent_mode == "zero":
                return torch.zeros(agent_count, self.cvae.latent_size, device=device)
            return torch.randn(agent_count, self.cvae.latent_size, device=device)

        latent = torch.empty(agent_count, self.cvae.latent_size, device=device)
        for start, end in scene_slices.tolist():
            if latent_mode == "zero":
                scene_latent = torch.zeros(1, self.cvae.latent_size, device=device)
            else:
                scene_latent = torch.randn(1, self.cvae.latent_size, device=device)
            latent[start:end] = scene_latent.expand(end - start, -1)
        return latent

    def forward(self, obs, context, target=None, scene_ids=None, scene_slices=None, latent_mode="sample"):
        obs_input = obs.permute(1, 2, 0)
        context_input = context.permute(1, 2, 0)

        motion_channels = build_motion_channels(obs_input)
        traj_feature = self.traj_encoder(motion_channels)
        context_feature = self.context_encoder(context_input)
        base_feature = torch.cat([traj_feature, context_feature], dim=-1)

        vertical_state = None
        vertical_summary = None
        state_condition = None
        if self.state_encoder is not None:
            vertical_channels = build_vertical_channels(obs_input)
            vertical_state, vertical_summary = self.state_encoder(vertical_channels)
            state_condition = self.state_scale * self.state_condition_proj(vertical_state)

        last_position = obs[-1]
        if obs.size(0) > 1:
            last_velocity = obs[-1] - obs[-2]
        else:
            last_velocity = torch.zeros_like(obs[-1])

        social_condition = None
        if self.variant == "base":
            pass
        elif self.variant == "state":
            pass
        elif self.variant == "interaction":
            social_feature = self.interaction_block(
                base_feature,
                last_position,
                last_velocity,
                scene_ids,
                state_summary=None,
            )
            if self.social_condition_proj is not None:
                social_condition = self.social_scale * self.social_condition_proj(social_feature)
        elif self.variant == "naive":
            social_feature = self.interaction_block(
                base_feature,
                last_position,
                last_velocity,
                scene_ids,
                state_summary=None,
            )
            if self.social_condition_proj is not None:
                social_condition = self.social_scale * self.social_condition_proj(social_feature)
        else:
            social_feature = self.full_interaction_block(
                base_feature,
                last_position,
                last_velocity,
                scene_ids,
                state_summary=vertical_summary,
            )
            if self.social_condition_proj is not None:
                social_condition = self.social_scale * self.social_condition_proj(social_feature)

        condition_parts = [base_feature]
        if state_condition is not None:
            condition_parts.append(state_condition)
        if social_condition is not None:
            condition_parts.append(social_condition)
        condition = torch.cat(condition_parts, dim=-1)
        if social_condition is None and self.social_gate is not None and self.social_update is not None:
            condition = condition
        elif social_condition is not None and self.social_gate is not None and self.social_update is not None:
            raise RuntimeError("concat and residualgate should not be active at the same time")

        if self.social_gate is not None and self.social_update is not None and self.variant in {"interaction", "naive", "full"}:
            social_source = None
            if self.variant in {"interaction", "naive"}:
                social_source = social_feature
            else:
                social_source = social_feature
            gate = self.social_gate(torch.cat([base_feature, social_source], dim=-1))
            update = self.social_update(social_source)
            condition = base_feature + self.social_scale * gate * update
            if state_condition is not None:
                condition = torch.cat([condition, state_condition], dim=-1)

        if self.training and self.condition_dropout > 0:
            condition = F.dropout(condition, p=self.condition_dropout, training=True)

        last_observation = obs[-1]
        if target is not None:
            future_deltas = future_to_deltas(target, last_observation)
            future_flat = future_deltas.reshape(future_deltas.size(0), -1)
            decoder_hidden, means, log_var, _ = self.cvae(future_flat, condition)
            decoded_deltas = self.delta_decoder(decoder_hidden).view(-1, self.pred_steps, 3)
            prediction = deltas_to_positions(decoded_deltas, last_observation)
            return prediction, means, log_var, decoded_deltas

        latent = self._sample_latent(obs.size(1), scene_slices, obs.device, latent_mode=latent_mode)
        decoder_hidden = self.cvae.inference(latent, condition)
        decoded_deltas = self.delta_decoder(decoder_hidden).view(-1, self.pred_steps, 3)
        prediction = deltas_to_positions(decoded_deltas, last_observation)
        return prediction
