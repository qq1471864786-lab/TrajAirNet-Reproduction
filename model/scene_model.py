"""
HAINet: Height-Aware Interaction Network for Aircraft Trajectory Prediction.

Base components strictly replicated from ACTrajNet:
  - TCN encoder (3-layer, weight_norm, Tanh, flatten full sequence)
  - CAF (Channel Attention Fusion, pre-flatten gating)
  - CVAE (Tanh decoder + unconstrained linear output)
  - Wind/Context encoder (Conv1d + Linear)
  - Verlet integration (acc_to_abs)

Innovation modules (our contribution):
  - AC-GAT: altitude-conditioned graph attention
  - Interaction->Height feedback (bidirectional coupling)
  - Second-round CAF with updated altitude
"""

import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TCN — exact copy from ACTrajNet (weight_norm + Tanh)
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.tanh(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# CVAE — replicated from ACTrajNet (Tanh decoder + unconstrained linear)
# ---------------------------------------------------------------------------

class CVAE(nn.Module):
    """
    ACTrajNet-style CVAE.
    Encoder: MLP with ReLU -> mu, logvar
    Decoder: MLP with Tanh (all layers) -> unconstrained output
    """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, num_labels=0):
        super().__init__()
        if conditional:
            assert num_labels > 0
        self.latent_size = latent_size

        self.encoder = CVAEEncoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = CVAEDecoder(decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(c, z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        return self.decoder(c, z)


class CVAEEncoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(f"L{i}", nn.Linear(in_size, out_size))
            self.MLP.add_module(f"A{i}", nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)
        return self.linear_means(x), self.linear_log_var(x)


class CVAEDecoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):
        super().__init__()
        self.conditional = conditional
        input_size = (num_labels + latent_size) if conditional else latent_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(f"L{i}", nn.Linear(in_size, out_size))
            # All layers use Tanh (matching ACTrajNet original)
            self.MLP.add_module(f"A{i}", nn.Tanh())

    def forward(self, c, z):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        return self.MLP(z)


# ---------------------------------------------------------------------------
# Verlet Integration — from ACTrajNet/TrajAirNet
# ---------------------------------------------------------------------------

def acc_to_abs(acc, obs):
    """
    Convert predicted accelerations to absolute positions via Verlet.
    acc: (pred_steps, channels, batch) — permuted from (batch, channels, pred_steps)
    obs: (obs_len, channels, batch)
    Returns: (pred_steps, channels, batch)
    """
    pred = torch.empty_like(acc)
    pred[0] = 2 * obs[-1] - obs[0] + acc[0]
    if acc.size(0) > 1:
        pred[1] = 2 * pred[0] - obs[-1] + acc[1]
    for i in range(2, acc.size(0)):
        pred[i] = 2 * pred[i - 1] - pred[i - 2] + acc[i]
    return pred


# ---------------------------------------------------------------------------
# AC-GAT (Altitude-Conditioned Graph Attention) — our innovation
# ---------------------------------------------------------------------------

class AltitudeConditionedGAT(nn.Module):
    """
    GAT with altitude-conditioned attention bias.
    alpha_ij = softmax(e_ij + phi(|dz|, dv_z, d_xy))
    """

    def __init__(self, in_features, out_features, n_heads=8, dropout=0.05, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.out_features = out_features
        assert out_features % n_heads == 0

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        # altitude interaction bias MLP
        self.phi = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads),
        )

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, positions, velocities, adj_mask):
        """
        h: (N, in_features)
        positions: (N, 3) last observed [x, y, z]
        velocities: (N, 3) last observed velocity
        adj_mask: (N, N) bool
        Returns: (N, out_features)
        """
        N = h.size(0)
        Wh = self.W(h).view(N, self.n_heads, self.head_dim)

        Wh_i = Wh.unsqueeze(1).expand(N, N, self.n_heads, self.head_dim)
        Wh_j = Wh.unsqueeze(0).expand(N, N, self.n_heads, self.head_dim)
        cat_ij = torch.cat([Wh_i, Wh_j], dim=-1)
        e = (cat_ij * self.a).sum(dim=-1)
        e = self.leaky_relu(e)

        # altitude interaction bias
        dz = (positions[:, 2:3].unsqueeze(1) - positions[:, 2:3].unsqueeze(0)).abs()
        dv_z = velocities[:, 2:3].unsqueeze(1) - velocities[:, 2:3].unsqueeze(0)
        d_xy = torch.linalg.norm(
            positions[:, :2].unsqueeze(1) - positions[:, :2].unsqueeze(0),
            dim=-1, keepdim=True)
        phi_bias = self.phi(torch.cat([dz, dv_z, d_xy], dim=-1))

        e = e + phi_bias
        e = e.masked_fill(~adj_mask.unsqueeze(-1), -1e9)
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum('ijk,ijkd->ikd', alpha, Wh_j)
        return out.reshape(N, -1)


# ---------------------------------------------------------------------------
# Interaction-to-Height Feedback — our innovation
# ---------------------------------------------------------------------------

class InteractionHeightFeedback(nn.Module):
    """Social message gates and updates altitude representation."""

    def __init__(self, social_dim, alt_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(social_dim, alt_dim), nn.Sigmoid())
        self.update = nn.Sequential(nn.Linear(social_dim, alt_dim), nn.ReLU())

    def forward(self, h_alt, h_social):
        return h_alt + self.gate(h_social) * self.update(h_social)


# ---------------------------------------------------------------------------
# HAINet: Main Model
# ---------------------------------------------------------------------------

class HAINet(nn.Module):
    """
    Height-Aware Interaction Network.

    Base: ACTrajNet (TCN + CAF + CVAE + Verlet)
    Innovation: AC-GAT + bidirectional height-interaction coupling
    """

    def __init__(
        self,
        obs_len=11,
        pred_len=120,
        pred_step=10,
        input_channels=3,
        tcn_channel_size=256,
        tcn_layers=2,
        tcn_kernel=4,
        dropout=0.2,
        cvae_hidden=128,
        cvae_layers=2,
        cvae_channel_size=128,
        mlp_layer=32,
        num_context_input_c=2,
        num_context_output_c=7,
        cnn_kernels=2,
        gat_hidden=256,
        gat_heads=8,
        gat_dropout=0.05,
        use_interaction=True,
        use_height_conditioning=True,
        use_height_feedback=True,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.n_classes = int(math.ceil(pred_len / pred_step))  # 12
        self.input_channels = input_channels
        self.mlp_layer = mlp_layer
        self.use_interaction = use_interaction
        self.use_height_conditioning = use_height_conditioning
        self.use_height_feedback = use_height_feedback

        # --- TCN channels (ACTrajNet style: last layer outputs n_classes) ---
        tcn_alti_out = 1
        num_channels_alt = [tcn_channel_size] * tcn_layers
        num_channels_alt.append(tcn_alti_out)  # [256, 256, 1]

        num_channels = [tcn_channel_size] * tcn_layers
        num_channels.append(self.n_classes)  # [256, 256, 12]

        # --- Trajectory TCN (obs) ---
        self.tcn_encoder_x = TemporalConvNet(
            input_channels, num_channels, kernel_size=tcn_kernel, dropout=dropout)
        # --- Altitude TCN (obs) ---
        self.tcn_encoder_altitude_X = TemporalConvNet(
            1, num_channels_alt, kernel_size=tcn_kernel, dropout=dropout)
        # --- CAF gate ---
        self.fc_a_X = nn.Linear(obs_len, self.n_classes)
        self.sig_a_X = nn.Sigmoid()

        # --- Future trajectory TCN (training only) ---
        self.tcn_encoder_y = TemporalConvNet(
            input_channels, num_channels, kernel_size=tcn_kernel, dropout=dropout)
        # --- Future altitude TCN (training only) ---
        self.tcn_encoder_altitude_Y = TemporalConvNet(
            1, num_channels_alt, kernel_size=tcn_kernel, dropout=dropout)
        self.fc_a_Y = nn.Linear(self.n_classes, self.n_classes)
        self.sig_a_Y = nn.Sigmoid()

        # --- Context/Wind encoder ---
        self.context_conv = nn.Conv1d(
            in_channels=num_context_input_c, out_channels=1, kernel_size=cnn_kernels)
        self.context_linear = nn.Linear(obs_len - 1, num_context_output_c)
        self.relu = nn.ReLU()

        # --- Condition dimension ---
        # h_fused_flat = n_classes * obs_len = 132
        # h_wind = num_context_output_c = 7
        # Total base condition = 139 (matches ACTrajNet gat_in)
        base_condition_dim = self.n_classes * obs_len + num_context_output_c  # 139

        # --- AC-GAT (innovation) ---
        self.gat = AltitudeConditionedGAT(
            in_features=base_condition_dim,  # 139
            out_features=gat_hidden,
            n_heads=gat_heads,
            dropout=gat_dropout,
        ) if use_interaction else None

        # --- Interaction -> Height feedback (innovation) ---
        # alt_dim = tcn_alti_out * obs_len = 1 * 11 = 11
        alt_flat_dim = tcn_alti_out * obs_len  # 11
        self.height_feedback = InteractionHeightFeedback(
            gat_hidden, alt_flat_dim
        ) if (use_interaction and use_height_feedback) else None

        # Second CAF gate for after feedback
        self.fc_a_X2 = nn.Linear(obs_len, self.n_classes) if (
            use_interaction and use_height_feedback) else None
        self.sig_a_X2 = nn.Sigmoid() if (
            use_interaction and use_height_feedback) else None

        # --- CVAE ---
        # Gate-based fusion: social info modulates base condition additively
        # CVAE condition stays at base_condition_dim (139), no wasted zero dims
        if use_interaction:
            self.social_gate = nn.Sequential(
                nn.Linear(gat_hidden, base_condition_dim), nn.Sigmoid())
            self.social_proj = nn.Linear(gat_hidden, base_condition_dim)
        else:
            self.social_gate = None
            self.social_proj = None

        condition_dim = base_condition_dim

        # Encoder input: future_flat_dim = n_classes * n_classes = 144
        future_flat_dim = self.n_classes * self.n_classes  # 144
        cvae_encoder_sizes = [future_flat_dim]
        for _ in range(cvae_layers):
            cvae_encoder_sizes.append(cvae_channel_size)

        cvae_decoder_sizes = [cvae_channel_size] * cvae_layers
        cvae_decoder_sizes.append(input_channels * mlp_layer)  # 3 * 32 = 96

        self.cvae = CVAE(
            encoder_layer_sizes=cvae_encoder_sizes,
            latent_size=cvae_hidden,
            decoder_layer_sizes=cvae_decoder_sizes,
            conditional=True,
            num_labels=condition_dim,
        )

        # Final linear decoder: mlp_layer -> n_classes (unconstrained)
        self.linear_decoder = nn.Linear(mlp_layer, self.n_classes)

        self.init_weights()

    def init_weights(self):
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)

    def _build_adj_mask(self, scene_ids):
        N = scene_ids.size(0)
        same_scene = scene_ids.unsqueeze(0) == scene_ids.unsqueeze(1)
        not_self = ~torch.eye(N, dtype=torch.bool, device=scene_ids.device)
        return same_scene & not_self

    def _encode_agent(self, x_agent, context_agent, is_obs=True):
        """
        Encode a single agent's trajectory (batched).
        x_agent: (batch, channels, seq_len) — trajectory
        context_agent: (batch, 2, seq_len) — wind context
        Returns: h_fused_flat (batch, n_classes*obs_len), h_alt_flat (batch, 1*obs_len),
                 encoded_traj (batch, n_classes, seq_len) — raw TCN output before flatten
        """
        # Altitude: (batch, 1, seq_len)
        altitude = x_agent[:, 2:3, :]

        # TCN encode
        if is_obs:
            encoded_alt = self.tcn_encoder_altitude_X(altitude)  # (batch, 1, seq_len)
            encoded_traj = self.tcn_encoder_x(x_agent)  # (batch, n_classes, seq_len)
            # CAF: altitude gates trajectory (pre-flatten)
            gate = self.sig_a_X(self.fc_a_X(encoded_alt))  # (batch, 1, n_classes)
            gate = gate.transpose(1, 2)  # (batch, n_classes, 1)
            # Broadcast gate across time dimension
            fused_traj = encoded_traj * gate  # (batch, n_classes, seq_len)
        else:
            encoded_alt = self.tcn_encoder_altitude_Y(altitude)
            encoded_traj = self.tcn_encoder_y(x_agent)
            gate = self.sig_a_Y(self.fc_a_Y(encoded_alt))
            gate = gate.transpose(1, 2)
            fused_traj = encoded_traj * gate

        # Flatten full sequence
        h_fused_flat = fused_traj.reshape(x_agent.size(0), -1)  # (batch, n_classes*seq_len)
        h_alt_flat = encoded_alt.reshape(x_agent.size(0), -1)  # (batch, 1*seq_len)

        return h_fused_flat, h_alt_flat, encoded_traj

    def forward(self, obs, context, target=None, scene_ids=None, scene_slices=None):
        """
        obs: (obs_len, N, 3) observed positions [x, y, z]
        context: (obs_len, N, 2) wind [w_x, w_y]
        target: (pred_steps, N, 3) future positions (training only)
        scene_ids: (N,) scene index per agent
        scene_slices: (S, 2) start/end indices per scene
        Returns:
          training: (prediction, mu, logvar, acc)
          inference: prediction
        """
        N = obs.size(1)

        # --- Encode obs ---
        # TCN expects (batch, channels, seq_len)
        obs_input = obs.permute(1, 2, 0)  # (N, 3, obs_len)
        ctx_input = context.permute(1, 2, 0)  # (N, 2, obs_len)

        h_fused_flat, h_alt_flat, encoded_traj_cache = self._encode_agent(obs_input, ctx_input, is_obs=True)
        # h_fused_flat: (N, 132), h_alt_flat: (N, 11)

        # --- Wind/Context encoding ---
        encoded_context = self.context_conv(ctx_input)  # (N, 1, obs_len-1)
        h_wind = self.relu(self.context_linear(encoded_context))  # (N, 1, 7)
        h_wind = h_wind.squeeze(1)  # (N, 7)

        # --- Base condition (matches ACTrajNet) ---
        condition_base = torch.cat([h_fused_flat, h_wind], dim=-1)  # (N, 139)

        # --- AC-GAT (innovation) ---
        h_social = None
        if self.use_interaction and self.gat is not None and N > 1 and scene_ids is not None:
            adj_mask = self._build_adj_mask(scene_ids)
            has_neighbor = adj_mask.any(dim=1)

            if has_neighbor.any():
                last_pos = obs[-1]  # (N, 3)
                last_vel = obs[-1] - obs[-2] if obs.size(0) > 1 else torch.zeros_like(obs[-1])

                if self.use_height_conditioning:
                    h_social_raw = self.gat(condition_base, last_pos, last_vel, adj_mask)
                else:
                    pos_no_z = last_pos.clone()
                    pos_no_z[:, 2] = 0
                    vel_no_z = last_vel.clone()
                    vel_no_z[:, 2] = 0
                    h_social_raw = self.gat(condition_base, pos_no_z, vel_no_z, adj_mask)

                h_social = h_social_raw * has_neighbor.unsqueeze(-1).float()

                # --- Interaction -> Height feedback (innovation) ---
                if self.height_feedback is not None:
                    h_alt_updated = self.height_feedback(h_alt_flat, h_social)
                    # Second-round CAF with updated altitude, reuse cached TCN output
                    h_alt_updated_2d = h_alt_updated.unsqueeze(1)  # (N, 1, 11)
                    gate2 = self.sig_a_X2(self.fc_a_X2(h_alt_updated_2d))
                    gate2 = gate2.transpose(1, 2)
                    fused_traj2 = encoded_traj_cache * gate2
                    h_fused_flat = fused_traj2.reshape(N, -1)
                    condition_base = torch.cat([h_fused_flat, h_wind], dim=-1)

        # --- Gate-based social fusion ---
        if h_social is not None:
            condition = condition_base + self.social_gate(h_social) * self.social_proj(h_social)
        else:
            condition = condition_base

        # --- CVAE ---
        if target is not None:
            # Training: encode future
            target_input = target.permute(1, 2, 0)  # (N, 3, pred_steps)
            h_future_flat, _, _ = self._encode_agent(target_input, ctx_input, is_obs=False)
            # h_future_flat: (N, n_classes * n_classes) = (N, 144)

            H_yy, means, log_var, z = self.cvae(
                h_future_flat.unsqueeze(1), condition.unsqueeze(1))
            # H_yy: (N, 1, 96)

            # Reshape and decode
            H_yy = H_yy.squeeze(1)  # (N, 96)
            H_yy = H_yy.view(N, self.input_channels, -1)  # (N, 3, 32)
            recon_y = self.linear_decoder(H_yy)  # (N, 3, 12)

            # Verlet integration
            acc = recon_y.permute(2, 1, 0)  # (12, 3, N)
            prediction = acc_to_abs(acc, obs.permute(0, 2, 1))  # obs: (11, 3, N)
            prediction = prediction.permute(0, 2, 1)  # (12, N, 3)

            return prediction, means.squeeze(1), log_var.squeeze(1), acc.permute(0, 2, 1)
        else:
            # Inference: sample one latent per scene, matching baseline scene-level sampling.
            if scene_slices is not None and scene_slices.numel() > 0:
                z = torch.empty(N, self.cvae.latent_size, device=obs.device)
                for start, end in scene_slices.tolist():
                    scene_z = torch.randn(1, self.cvae.latent_size, device=obs.device)
                    z[start:end] = scene_z.expand(end - start, -1)
            else:
                z = torch.randn(N, self.cvae.latent_size, device=obs.device)
            H_yy = self.cvae.inference(z.unsqueeze(1), condition.unsqueeze(1))
            H_yy = H_yy.squeeze(1)
            H_yy = H_yy.view(N, self.input_channels, -1)
            recon_y = self.linear_decoder(H_yy)

            acc = recon_y.permute(2, 1, 0)
            prediction = acc_to_abs(acc, obs.permute(0, 2, 1))
            prediction = prediction.permute(0, 2, 1)

            return prediction
