import torch
import torch.nn as nn
from transformers import ViltModel
from config import Config

cfg = Config()


class ViltEncoder(nn.Module):
    """
    M2 — ViLT Encoder (shared between Stage 1 and Stage 2).

    Input:
        input_ids       : (B, seq_len)        token ids
        attention_mask  : (B, seq_len)
        token_type_ids  : (B, seq_len)
        pixel_values    : (B, 3, H, W)        image patches
        pixel_mask      : (B, H, W)

    Output (dict):
        token_embeddings : (B, seq_len, hidden)   per-token embeddings
        patch_embeddings : (B, num_patches, hidden) per-patch embeddings
        pooler_output    : (B, hidden)             CLS pooled embedding
    """

    def __init__(self):
        super().__init__()
        self.vilt = ViltModel.from_pretrained(cfg.VILT_MODEL_NAME)
        self.hidden_size = self.vilt.config.hidden_size   # 768

    def forward(self, input_ids, attention_mask, token_type_ids,
                pixel_values, pixel_mask):

        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=False,
        )

        # full sequence: [CLS] + text tokens + [SEP] + image patches
        sequence_output = outputs.last_hidden_state  # (B, full_seq, hidden)
        pooler_output   = outputs.pooler_output      # (B, hidden)

        # split text tokens from image patch tokens
        # text occupies positions 0 .. input_ids.shape[1]-1
        text_len = input_ids.shape[1]
        token_embeddings = sequence_output[:, :text_len, :]
        patch_embeddings = sequence_output[:, text_len:, :]

        return {
            "token_embeddings": token_embeddings,
            "patch_embeddings": patch_embeddings,
            "pooler_output":    pooler_output,
        }


if __name__ == "__main__":
    print("Testing ViltEncoder...")
    encoder = ViltEncoder().to(cfg.DEVICE)
    B, T, H, W = 2, 40, 384, 384
    dummy = {
        "input_ids":      torch.zeros(B, T, dtype=torch.long).to(cfg.DEVICE),
        "attention_mask": torch.ones(B, T, dtype=torch.long).to(cfg.DEVICE),
        "token_type_ids": torch.zeros(B, T, dtype=torch.long).to(cfg.DEVICE),
        "pixel_values":   torch.randn(B, 3, H, W).to(cfg.DEVICE),
        "pixel_mask":     torch.ones(B, H, W, dtype=torch.long).to(cfg.DEVICE),
    }
    out = encoder(**dummy)
    print("token_embeddings:", out["token_embeddings"].shape)
    print("patch_embeddings:", out["patch_embeddings"].shape)
    print("pooler_output   :", out["pooler_output"].shape)
    print("Encoder OK")