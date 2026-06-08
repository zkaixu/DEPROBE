import torch
import torch.nn.functional as F


def rank_n_contrast_loss(z: torch.Tensor, y: torch.Tensor,
                         tau: float = 0.1, margin: float = 0.1) -> torch.Tensor:
    """
    DEPROBE-DNA: Rank-N-Contrast Loss for Regression.

    Zha et al., "Rank-N-Contrast: Learning Continuous Representations
    for Regression", NeurIPS 2023.

    Within each mini-batch, label (efficiency) distance determines pairs:
      |y_i - y_j| <= margin  ->  positive pair (pull embeddings together)
      |y_i - y_j| > margin   ->  negative pair (push embeddings apart)

    This preserves the ranking order of capture efficiency in embedding
    space WITHOUT pre-computed neg_indices, avoiding false negative
    collision entirely.

    Args:
        z:      Latent embeddings [Batch, Dim]
        y:      Capture efficiency labels [Batch]
        tau:    Temperature coefficient (lower = sharper distribution)
        margin: Efficiency distance threshold for positive/negative split.
                With QuantileTransformer-normalized efficiency in [0,1],
                margin=0.1 means probes within 10% efficiency are positives.

    Returns:
        Scalar loss (0 if no valid positive pairs in batch)
    """
    z = F.normalize(z, p=2, dim=-1)
    B = z.size(0)

    if B < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # Pairwise cosine similarity scaled by temperature
    logits = torch.mm(z, z.t()) / tau  # (B, B)

    # Pairwise label distance
    y_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))  # (B, B)

    # Masks
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = (y_dist <= margin) & ~self_mask
    neg_mask = (y_dist > margin) & ~self_mask

    # Numerical stability: subtract row max
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(self_mask, 0)

    # Per-anchor: -log(sum_pos / sum_all)
    pos_sum = (exp_logits * pos_mask.float()).sum(dim=1)
    all_sum = exp_logits.sum(dim=1)

    # Only compute loss for anchors that have at least one positive
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    loss = -torch.log(pos_sum[valid] / (all_sum[valid] + 1e-8) + 1e-8)
    return loss.mean()
