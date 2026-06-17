import torch
import torch.nn as nn
from config import Config

cfg = Config()


class Stage2Classifier(nn.Module):
    """
    M7 — Stage 2 Classifier.
    Final humanitarian class prediction based solely on rationale-only input.
    This is what makes VLTCrisis interpretable-by-design.

    Input:
        pooler_output : (B, hidden_size)   from ViLT re-encoding of masked input

    Output:
        logits        : (B, num_classes)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc      = nn.Linear(hidden_size, cfg.NUM_CLASSES)
        self.dropout = nn.Dropout(0.1)

    def forward(self, pooler_output):
        return self.fc(self.dropout(pooler_output))


if __name__ == "__main__":
    print("Testing Stage2Classifier...")
    model  = Stage2Classifier(hidden_size=768)
    x      = torch.randn(2, 768)
    logits = model(x)
    print("logits shape:", logits.shape)   # (2, 5)
    print("Stage2Classifier OK")