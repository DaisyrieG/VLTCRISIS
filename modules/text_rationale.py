import torch
import torch.nn as nn
from config import Config

cfg = Config()


class TextRationaleExtractor(nn.Module):
    """
    M3 — Text Rationale Extractor.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=cfg.GRU_HIDDEN_SIZE,
            num_layers=cfg.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        # bidirectional → output is 2 * GRU_HIDDEN_SIZE
        self.fc = nn.Linear(cfg.GRU_HIDDEN_SIZE * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, token_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings: Tensor of shape (B, seq_len, hidden_size)
            
        Returns:
            rationale_probs: Float tensor (B, seq_len) in [0, 1]
            rationale_preds: Float tensor (B, seq_len) binary {0, 1}
        """
        gru_out, _ = self.gru(token_embeddings)
        # logits  : (B, seq_len, 1) → squeeze → (B, seq_len)
        logits = self.fc(gru_out).squeeze(-1)
        rationale_probs = self.sigmoid(logits)

        # Now that we have Cross-Modal Transfer, we don't need the forced 25% hack anymore!
        # We can just threshold the probabilities at 0.5 like a normal classifier.
        rationale_preds = (rationale_probs > 0.5).float()

        return rationale_probs, rationale_preds


if __name__ == "__main__":
    print("Testing TextRationaleExtractor...")
    model = TextRationaleExtractor(hidden_size=768)
    x     = torch.randn(2, 40, 768)
    probs, preds = model(x)
    print("rationale_probs:", probs.shape)
    print("rationale_preds:", preds.shape)
    print("TextRationale OK")