import torch
from config import Config

cfg = Config()

# Token ID for the mask symbol '*'
# ViLT uses BERT tokenizer; '[UNK]' token (id=100) is our fallback mask
MASK_TOKEN_ID = 100


class RationaleMasker:
    """
    M6 — Rationale Masker (no learned weights — purely deterministic).

    Takes original token ids and image patches, applies rationale masks:
      - Non-rationale text tokens are replaced with MASK_TOKEN_ID
      - Non-rationale image patches are zeroed out (blurred)

    Input:
        input_ids        : (B, seq_len)
        pixel_values     : (B, 3, H, W)
        rationale_preds  : (B, seq_len)      binary token labels
        heatmap          : (B, num_patches)  patch importance scores [0,1]

    Output:
        masked_input_ids   : (B, seq_len)
        masked_pixel_values: (B, 3, H, W)
    """

    @staticmethod
    def mask_text(input_ids: torch.Tensor, rationale_preds: torch.Tensor) -> torch.Tensor:
        masked = input_ids.clone()
        # where prediction is 0 (non-rationale), replace with mask token
        non_rationale = (rationale_preds == 0)
        masked[non_rationale] = MASK_TOKEN_ID
        return masked

    @staticmethod
    def mask_image(pixel_values: torch.Tensor, heatmap: torch.Tensor, patch_size: int = None) -> torch.Tensor:
        """
        Blur (zero out) patches whose heatmap score is below the threshold.
        Top TOP_PATCH_RATIO patches are kept; the rest are zeroed.
        """
        patch_size = patch_size or cfg.PATCH_SIZE
        B, C, H, W = pixel_values.shape
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        masked = pixel_values.clone()

        for b in range(B):
            scores = heatmap[b]                          # (num_patches,)
            k      = max(1, int(cfg.TOP_PATCH_RATIO * scores.shape[0]))
            # indices of top-k patches to KEEP
            keep   = scores.topk(k).indices.tolist()
            keep_set = set(keep)

            patch_idx = 0
            for pi in range(num_patches_h):
                for pj in range(num_patches_w):
                    if patch_idx not in keep_set:
                        r0 = pi * patch_size
                        r1 = r0 + patch_size
                        c0 = pj * patch_size
                        c1 = c0 + patch_size
                        masked[b, :, r0:r1, c0:c1] = 0.0
                    patch_idx += 1

        return masked

    @classmethod
    def apply(cls, input_ids: torch.Tensor, pixel_values: torch.Tensor, rationale_preds: torch.Tensor, heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked_ids    = cls.mask_text(input_ids, rationale_preds)
        masked_pixels = cls.mask_image(pixel_values, heatmap)
        return masked_ids, masked_pixels


if __name__ == "__main__":
    print("Testing RationaleMasker...")
    B, T, H, W = 2, 40, 384, 384
    ids    = torch.randint(0, 1000, (B, T))
    pixels = torch.randn(B, 3, H, W)
    preds  = torch.zeros(B, T)
    preds[:, 5:10] = 1.0
    heat   = torch.rand(B, 144)

    m_ids, m_pix = RationaleMasker.apply(ids, pixels, preds, heat)
    print("masked_input_ids shape   :", m_ids.shape)
    print("masked_pixel_values shape:", m_pix.shape)
    print("Masker OK")