import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from modules.encoder import ViltEncoder
from modules.text_rationale import TextRationaleExtractor
from modules.aux_classifier import AuxClassifier
from modules.image_rationale import ImageRationaleExtractor
from modules.masker import RationaleMasker
from modules.stage2_classifier import Stage2Classifier

cfg = Config()


class VLTCrisis(nn.Module):
    """
    Full VLTCrisis model — wires M2 through M7.

    Stage 1:
        M2 ViLT encoder  →  token/patch/pooler embeddings
        M3 Text rationale extractor  →  token rationale probs + preds
        M4 Auxiliary classifier  →  stage1 class logits (for training only)
        M5 Image rationale extractor  →  patch heatmap

    Stage 2:
        M6 Masker  →  masked token ids + masked pixel values
        M2 ViLT encoder (same weights)  →  rationale-only embeddings
        M7 Stage 2 classifier  →  final class logits

    Returns (during training):
        stage2_logits    : (B, num_classes)   main prediction
        stage1_logits    : (B, num_classes)   aux prediction
        rationale_probs  : (B, seq_len)       for rationale loss
        heatmap          : (B, num_patches)   image rationale scores

    Returns (during inference):
        stage2_logits    : (B, num_classes)
        rationale_preds  : (B, seq_len)
        heatmap          : (B, num_patches)
    """

    def __init__(self):
        super().__init__()
        self.encoder      = ViltEncoder()
        hidden            = self.encoder.hidden_size

        self.text_rat     = TextRationaleExtractor(hidden_size=hidden)
        self.aux_clf      = AuxClassifier(hidden_size=hidden)
        self.stage2_clf   = Stage2Classifier(hidden_size=hidden)

        # M5 and M6 have no parameters
        self.img_rat      = ImageRationaleExtractor()
        self.masker       = RationaleMasker()

    def forward(self, input_ids, attention_mask, token_type_ids,
                pixel_values, pixel_mask, training=True):

        # ── Stage 1: encode full tweet ─────────────────────────────────────
        s1_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )
        token_embs  = s1_out["token_embeddings"]   # (B, seq_len, H)
        patch_embs  = s1_out["patch_embeddings"]   # (B, num_patches, H)
        pooler      = s1_out["pooler_output"]      # (B, H)

        # M3 — text rationales
        rat_probs, rat_preds = self.text_rat(token_embs)

        # M4 — auxiliary class prediction
        stage1_logits = self.aux_clf(pooler)

        # M5 — image rationale heatmap and cross-modal text pseudo-labels
        with torch.no_grad():
            heatmap, text_pseudo_labels = ImageRationaleExtractor.compute(
                token_embs, patch_embs, attention_mask=attention_mask
            )

        # ── Stage 2: re-encode masked rationales only ──────────────────────
        masked_ids, masked_pixels = RationaleMasker.apply(
            input_ids, pixel_values, rat_preds, heatmap
        )

        s2_out = self.encoder(
            input_ids=masked_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=masked_pixels,
            pixel_mask=pixel_mask,
        )
        pooler2 = s2_out["pooler_output"]

        # M7 — final classifier
        stage2_logits = self.stage2_clf(pooler2)

        if training:
            return stage2_logits, stage1_logits, rat_probs, heatmap, text_pseudo_labels
        else:
            return stage2_logits, rat_preds, heatmap


if __name__ == "__main__":
    print("Testing VLTCrisis full model...")
    model = VLTCrisis().to(cfg.DEVICE)
    B, T, H, W = 2, 40, 384, 384

    dummy_input = {
        "input_ids":      torch.zeros(B, T, dtype=torch.long).to(cfg.DEVICE),
        "attention_mask": torch.ones(B, T, dtype=torch.long).to(cfg.DEVICE),
        "token_type_ids": torch.zeros(B, T, dtype=torch.long).to(cfg.DEVICE),
        "pixel_values":   torch.randn(B, 3, H, W).to(cfg.DEVICE),
        "pixel_mask":     torch.ones(B, H, W, dtype=torch.long).to(cfg.DEVICE),
    }

    s2_logits, s1_logits, rat_probs, heatmap = model(**dummy_input)
    print("stage2_logits :", s2_logits.shape)
    print("stage1_logits :", s1_logits.shape)
    print("rat_probs     :", rat_probs.shape)
    print("heatmap       :", heatmap.shape)
    print("VLTCrisis full model OK")