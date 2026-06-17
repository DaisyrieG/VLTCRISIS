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
    def compute(token_embeddings, patch_embeddings, rationale_preds):
        B = token_embeddings.shape[0]
        heatmaps = []

        for b in range(B):
            # select only predicted rationale tokens
            mask        = rationale_preds[b].bool()
            rat_embeds  = token_embeddings[b][mask]        # (n_rat, hidden)
            patch_embs  = patch_embeddings[b]              # (num_patches, hidden)

            if rat_embeds.shape[0] == 0:
                # no rationale predicted — zero heatmap
                heatmaps.append(
                    torch.zeros(patch_embs.shape[0], device=patch_embs.device)
                )
                continue

            # normalise embeddings for cosine similarity
            rat_norm   = F.normalize(rat_embeds, dim=-1)
            patch_norm = F.normalize(patch_embs,  dim=-1)

            # cost matrix: 1 - cosine_similarity  (n_rat × num_patches)
            sim_matrix  = rat_norm @ patch_norm.t()        # (n_rat, num_patches)
            cost_matrix = 1.0 - sim_matrix

            # run IPOT
            transport   = ipot(cost_matrix)                # (n_rat, num_patches)

            # heatmap = max transport weight across rationale tokens per patch
            patch_scores = transport.max(dim=0).values     # (num_patches,)

            # normalise to [0, 1]
            mn, mx = patch_scores.min(), patch_scores.max()
            if mx > mn:
                patch_scores = (patch_scores - mn) / (mx - mn + 1e-9)

            heatmaps.append(patch_scores)

        return torch.stack(heatmaps, dim=0)   # (B, num_patches)


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