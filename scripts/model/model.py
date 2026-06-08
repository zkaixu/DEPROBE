import torch
import torch.nn as nn
import torch.nn.functional as F


class GRL(torch.autograd.Function):
    """
    Core Gradient Reversal implementation.
    Inverts gradients during backpropagation to force domain invariance.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by alpha to confuse the domain classifier
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, alpha=None):
        # Allow dynamic override of alpha during training steps (e.g., progressive growing)
        a = alpha if alpha is not None else self.alpha
        return GRL.apply(x, a)


class DEPROBE(nn.Module):
    """
    DEPROBE: Physics-informed Transformer-CNN.

    Architecture:
    - Masked Mean Pooling avoids sequence-length artifacts.
    - Receptive-field mask alignment for CNN downsampling.
    - Early fusion of 12D physical priors as a prompt token.
    - LayerNorm on priors stabilizes attention gradients.
    """

    def __init__(self, max_seq_len=120, num_platforms=10, prior_dim=12, num_modalities=5, d_model=256):
        super().__init__()

        # 1. Base Token Embedding (0 is reserved for 'N'/Padding)
        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=d_model, padding_idx=0)

        # 2. Local Feature Extraction (1D CNN)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=d_model, out_channels=d_model // 2, kernel_size=5, padding=2)

        self.conv_bn = nn.BatchNorm1d(d_model)
        self.conv_relu = nn.ReLU()
        self.conv_pool = nn.MaxPool1d(kernel_size=2)

        # Learnable Positional Encoding (adapts to variable lengths with/without physics token)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len // 2 + 2, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # =====================================================================
        # 3. Physical Prior Projection Layer (Early Fusion)
        # LayerNorm stabilizes the 12D numerical values before projection
        # =====================================================================
        self.priors_norm = nn.LayerNorm(prior_dim)
        self.physics_proj = nn.Linear(prior_dim, d_model)

        # 4. Global Context Modeling (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 5. Modality Integration & Normalization
        self.modality_embedding = nn.Embedding(num_modalities, 32)
        self.layer_norm = nn.LayerNorm(d_model)

        # 6. Efficiency Regression Branch (Late Fusion included)
        combined_dim = d_model + prior_dim + 32
        self.efficiency_regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 7. Platform Adversarial Branch (DANN)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_platforms)
        )

        self.attention_weights = None
        self._hook_handle = None

    def register_attention_hook(self, layer_idx=0):
        """
        Registers a forward hook to capture attention weights (For Inference/Visualization ONLY).
        Defaults to targeting the self-attention module of the specified Transformer layer.
        """
        self.remove_attention_hook()  # Prevent duplicate hooks and potential memory leaks

        def hook_fn(module, input, output):
            # PyTorch MultiheadAttention returns a tuple: (attn_output, attn_output_weights)
            # attn_output_weights shape: (Batch_Size, Target_Len, Source_Len)
            self.attention_weights = output[1].detach().cpu()

        # Attach the hook to the targeted Transformer Encoder layer
        target_layer = self.transformer.layers[layer_idx].self_attn
        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def remove_attention_hook(self):
        """
        Removes the active attention hook and frees up memory.
        CRITICAL: Must be called before switching back to training mode.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self.attention_weights = None

    def encode(self, x, priors, modality_labels, pad_mask=None, inject_physics=True):
        """
        Core Latent Extraction Pipeline.
        Args:
            inject_physics: If False, skips the physics prompt token (for contrastive branch).
        """
        x_emb = self.embedding(x).transpose(1, 2)

        # --- Multiscale Feature Fusion ---
        x3 = self.conv3(x_emb)
        x5 = self.conv5(x_emb)
        x_cat = torch.cat([x3, x5], dim=1)

        x_conv = self.conv_pool(self.conv_relu(self.conv_bn(x_cat)))
        x_seq = x_conv.transpose(1, 2)

        # --- Receptive Field Mask Alignment ---
        downsampled_mask = None
        if pad_mask is not None:
            float_mask = pad_mask.float().unsqueeze(1)
            pooled_mask = F.max_pool1d(float_mask, kernel_size=2, stride=2)
            downsampled_mask = pooled_mask.squeeze(1).bool()
            downsampled_mask = downsampled_mask[:, :x_seq.size(1)]

        # =====================================================================
        # --- Early Fusion of Physical Priors (Conditional) ---
        # =====================================================================
        priors_normed = self.priors_norm(priors)

        if inject_physics:
            physics_token = self.physics_proj(priors_normed).unsqueeze(1)  # (Batch, 1, d_model)
            x_fused = torch.cat([physics_token, x_seq], dim=1)

            final_mask = None
            if downsampled_mask is not None:
                physics_mask = torch.zeros((x_seq.size(0), 1), dtype=torch.bool, device=x.device)
                final_mask = torch.cat([physics_mask, downsampled_mask], dim=1)
        else:
            x_fused = x_seq
            final_mask = downsampled_mask

        # --- Positional Encoding ---
        x_fused = x_fused + self.pos_encoding[:, :x_fused.size(1), :]

        # --- Global Context Modeling ---
        t_out = self.transformer(x_fused, src_key_padding_mask=final_mask)

        # --- Masked Mean Pooling ---
        if final_mask is not None:
            valid_tokens = (~final_mask).float().unsqueeze(-1)  # (Batch, Seq_Len, 1)
            z = (t_out * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1e-9)
        else:
            z = t_out.mean(dim=1)

        z = self.layer_norm(z)

        # --- Late Fusion (Residual Physics & Modality) ---
        mod_emb = self.modality_embedding(modality_labels)
        z_fused = torch.cat([z, priors_normed, mod_emb], dim=1)

        return z, z_fused

    def forward(self, x, priors, modality_labels, pad_mask=None, alpha=1.0):
        """
        Forward pass executing multi-task learning.
        """
        z_latent, z_fused = self.encode(x, priors, modality_labels, pad_mask)

        # Task 1: Efficiency score prediction
        eff_pred = self.efficiency_regressor(z_fused).squeeze(-1)

        # Task 2: Domain (Platform) prediction (Adversarial)
        z_reversed = self.grl(z_latent, alpha=alpha)
        domain_pred = self.domain_classifier(z_reversed)

        return z_latent, eff_pred, domain_pred