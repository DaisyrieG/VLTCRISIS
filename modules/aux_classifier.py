import torch
import torch.nn as nn
from config import Config

cfg = Config()


class AuxClassifier(nn.Module):
    """
    M4 — Auxiliary Classifier (Stage 1).
    Predicts humanitarian class from the ViLT pooler embedding.
    This loss guides the rationale extractor to pick class-relevant tokens.

    Input:
        pooler_output : (B, hidden_size)

    Output:
        logits        : (B, num_classes)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, cfg.NUM_CLASSES)

    def forward(self, pooler_output):
        return self.fc(pooler_output)


if __name__ == "__main__":
    print("Testing AuxClassifier...")
    model  = AuxClassifier(hidden_size=768)
    x      = torch.randn(2, 768)
    logits = model(x)
    print("logits shape:", logits.shape)   # (2, 5)
    print("AuxClassifier OK")