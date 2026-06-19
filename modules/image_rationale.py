import torch
import torch.nn.functional as F
from config import Config

cfg = Config()


def ipot(cost_matrix, beta=None, iterations=None):
    """
    Inexact Proximal point method for Optimal Transport (IPOT).
    Computes soft alignment between rows and columns of a cost matrix.

    Args:
        cost_matrix : (n, m)  pairwise cosine distance between
                               rationale tokens (n) and image patches (m)
        beta        : regularisation strength
        iterations  : number of Sinkhorn-like iterations

    Returns:
        transport   : (n, m)  soft transport plan
    """
    beta       = beta or cfg.IPOT_BETA
    iterations = iterations or cfg.IPOT_ITERATIONS

    n, m  = cost_matrix.shape
    # uniform marginals
    a = torch.ones(n, device=cost_matrix.device) / n
    b = torch.ones(m, device=cost_matrix.device) / m

    # Gibbs kernel
    K = torch.exp(-cost_matrix / beta)
    u = torch.ones(n, device=cost_matrix.device) / n

    for _ in range(iterations):
        v = b / (K.t() @ u + 1e-9)
        u = a / (K @ v + 1e-9)

    transport = torch.diag(u) @ K @ torch.diag(v)
    return transport


class ImageRationaleExtractor:
    """
    M5 — Image Rationale Extractor (no learned weights).

    Uses IPOT to align predicted text rationale token embeddings
    with image patch embeddings and produces a heatmap.

    Input:
        token_embeddings  : (B, seq_len, hidden)
        patch_embeddings  : (B, num_patches, hidden)
        rationale_preds   : (B, seq_len)   binary — which tokens are rationales

    Output:
        heatmap           : (B, num_patches)   importance score per patch [0,1]
    """

    @staticmethod
    def compute(token_embeddings, patch_embeddings, attention_mask=None, rationale_preds=None):
        B = token_embeddings.shape[0]
        heatmaps = []
        text_pseudo_labels = []

        for b in range(B):
            # CROSS-MODAL TRANSFER: We use ALL text tokens to align with image patches.
            tok_embs    = token_embeddings[b]              # (seq_len, hidden)
            patch_embs  = patch_embeddings[b]              # (num_patches, hidden)

            # normalise embeddings for cosine similarity
            tok_norm   = F.normalize(tok_embs, dim=-1)
            patch_norm = F.normalize(patch_embs,  dim=-1)

            # cost matrix: 1 - cosine_similarity  (seq_len × num_patches)
            sim_matrix  = tok_norm @ patch_norm.t()        # (seq_len, num_patches)
            cost_matrix = 1.0 - sim_matrix

            # run IPOT
            transport   = ipot(cost_matrix)                # (seq_len, num_patches)

            # heatmap = max transport weight across text tokens per patch
            patch_scores = transport.max(dim=0).values     # (num_patches,)

            # normalise to [0, 1]
            mn, mx = patch_scores.min(), patch_scores.max()
            if mx > mn:
                patch_scores = (patch_scores - mn) / (mx - mn + 1e-9)

            heatmaps.append(patch_scores)

            # Cross-Modal Transfer: Extract text pseudo-labels from the IPOT alignment
            # Find the max transport weight across image patches per text token
            tok_scores = transport.max(dim=1).values       # (seq_len,)
            
            # Mask out [CLS] (index 0), [SEP], and padding tokens
            if attention_mask is not None:
                mask = attention_mask[b].bool()
                # [CLS] is at 0
                mask[0] = False
                # [SEP] is the last True in the mask
                sep_idx = mask.nonzero().max().item()
                mask[sep_idx] = False
                
                tok_scores = tok_scores.masked_fill(~mask, -1e9)
            
            mn_t, mx_t = tok_scores[tok_scores > -1e8].min(), tok_scores[tok_scores > -1e8].max()
            if mx_t > mn_t:
                tok_scores = torch.where(tok_scores > -1e8, (tok_scores - mn_t) / (mx_t - mn_t + 1e-9), tok_scores)
            
            # The top 25% of mathematically aligned tokens become our "ground truth" pseudo-label
            # We calculate k based on the number of actual text tokens, not the total seq_len
            if attention_mask is not None:
                num_text_tokens = mask.sum().item()
                k = max(1, int(cfg.TOP_PATCH_RATIO * num_text_tokens))
            else:
                k = max(1, int(cfg.TOP_PATCH_RATIO * tok_scores.shape[0]))
                
            pseudo_label = torch.zeros_like(tok_scores)
            if k > 0:
                pseudo_label[tok_scores.topk(k).indices] = 1.0
            
            text_pseudo_labels.append(pseudo_label)

        return torch.stack(heatmaps, dim=0), torch.stack(text_pseudo_labels, dim=0)


if __name__ == "__main__":
    print("Testing ImageRationaleExtractor...")
    B, T, P, H = 2, 40, 144, 768
    tok  = torch.randn(B, T, H)
    pat  = torch.randn(B, P, H)
    pred = torch.zeros(B, T)
    pred[:, 3:7] = 1.0     # pretend tokens 3-6 are rationales

    heatmap = ImageRationaleExtractor.compute(tok, pat, pred)
    print("heatmap shape:", heatmap.shape)   # (2, 144)
    print("heatmap range: [{:.3f}, {:.3f}]".format(
        heatmap.min().item(), heatmap.max().item()))
    print("ImageRationale OK")