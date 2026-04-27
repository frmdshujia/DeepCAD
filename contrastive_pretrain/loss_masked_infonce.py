"""loss_masked_infonce.py
Masked InfoNCE loss with PC14-guided positive / negative selection.

For each anchor i in a batch of B pairs:
  - Positives:  diagonal (i == j, same person) + K_pos nearest PC14 neighbors
  - Negatives:  K_neg farthest PC14 neighbors
  - Middle:     masked out — produce no gradient

Loss = SupCon-style CE averaged over both f->c and c->f directions.
Reference: Khosla et al. NeurIPS 2020, Supervised Contrastive Learning.
"""
from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_masks(pc14: torch.Tensor, k_pos: int, k_neg: int):
    """Return (pos_mask, active_mask) both shape (B, B) bool.

    pos_mask[i,j]    = True  iff j is a positive for anchor i
    active_mask[i,j] = True  iff j contributes to denominator for anchor i
    """
    B = pc14.size(0)
    device = pc14.device

    # Pairwise PC14 Euclidean distances (B, B)
    dist = torch.cdist(pc14.float(), pc14.float(), p=2)

    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
    neg_mask = torch.zeros(B, B, dtype=torch.bool, device=device)

    # Diagonal is always a positive (same person)
    pos_mask.fill_diagonal_(True)

    if k_pos > 0 and B > 1:
        dist_ex = dist.clone()
        dist_ex.fill_diagonal_(float('inf'))          # exclude self
        k_eff = min(k_pos, B - 1)
        nn_idx = dist_ex.topk(k_eff, dim=1, largest=False).indices  # (B, k_eff)
        pos_mask.scatter_(1, nn_idx, True)

    if k_neg > 0 and B > 1:
        k_eff = min(k_neg, B - 1)
        fn_idx = dist.topk(k_eff, dim=1, largest=True).indices      # (B, k_eff)
        neg_mask.scatter_(1, fn_idx, True)
        neg_mask.fill_diagonal_(False)   # self cannot be a negative
        neg_mask &= ~pos_mask            # positives take precedence

    active_mask = pos_mask | neg_mask
    return pos_mask, active_mask


def _masked_ce(logits: torch.Tensor,
               pos_mask: torch.Tensor,
               active_mask: torch.Tensor) -> torch.Tensor:
    """SupCon-style cross-entropy restricted to active set.

    For each anchor i with at least one negative in active set:
      loss_i = -1/|P_i| * sum_{p in P_i} (logits[i,p] - logsumexp(logits[i, active]))

    Anchors with no negatives (|active| == |pos|) are skipped.
    """
    # Fill inactive entries with -inf so they don't affect logsumexp
    NEG_INF = torch.finfo(logits.dtype).min / 2
    masked = logits.masked_fill(~active_mask, NEG_INF)
    log_denom = torch.logsumexp(masked, dim=1)   # (B,)

    n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)      # (B,)
    log_probs = logits - log_denom.unsqueeze(1)              # (B, B)
    loss_per = -(log_probs * pos_mask.float()).sum(dim=1) / n_pos  # (B,)

    # Only back-propagate through anchors that have at least one negative
    n_active = active_mask.float().sum(dim=1)   # (B,)
    valid = n_active > pos_mask.float().sum(dim=1)  # has negatives
    if not valid.any():
        return logits.sum() * 0.0               # keep graph, zero gradient
    return loss_per[valid].mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def masked_infonce_loss(
    zf: torch.Tensor,    # (B, d) L2-normalised fundus embeddings
    zc: torch.Tensor,    # (B, d) L2-normalised CMR   embeddings
    pc14: torch.Tensor,  # (B, 14) PC14 vectors (float32, same-batch CMR subjects)
    temperature: float = 0.07,
    k_pos: int = 2,
    k_neg: int = 4,
) -> torch.Tensor:
    """Masked InfoNCE: symmetric, PC14-guided.

    Args:
        zf:          (B, d) fundus embeddings, L2-normalised
        zc:          (B, d) CMR embeddings,    L2-normalised
        pc14:        (B, 14) CMR PC14 feature vectors for the B subjects
        temperature: InfoNCE temperature (default 0.07)
        k_pos:       extra positive neighbours beyond the diagonal (default 2)
        k_neg:       hard negative neighbours (default 4)

    Returns:
        scalar loss
    """
    assert zf.size(0) == zc.size(0) == pc14.size(0), \
        "zf, zc, pc14 must have same batch size"

    pos_mask, active_mask = _build_masks(pc14, k_pos, k_neg)

    logits_fc = zf @ zc.T / temperature   # f -> c
    logits_cf = zc @ zf.T / temperature   # c -> f

    loss_fc = _masked_ce(logits_fc, pos_mask, active_mask)
    loss_cf = _masked_ce(logits_cf, pos_mask, active_mask)
    return 0.5 * (loss_fc + loss_cf)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    B, d = 8, 256
    zf = torch.randn(B, d); zf = zf / zf.norm(dim=1, keepdim=True)
    zc = torch.randn(B, d); zc = zc / zc.norm(dim=1, keepdim=True)
    pc = torch.randn(B, 14)
    loss = masked_infonce_loss(zf, zc, pc, k_pos=2, k_neg=4)
    print(f'[sanity] loss={loss.item():.4f}  (expected < ln({B})={torch.tensor(B).float().log():.3f})')
    loss.backward()
    print('[sanity] backward ok')
