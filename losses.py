import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

cfg = Config()


class VLTCrisisLoss(nn.Module):
    """
    Combined loss:
        Loss = Loss_l + alpha * Loss_r

    where:
        Loss_l  = cross-entropy on humanitarian class (stage2 + aux stage1)
        Loss_r  = weighted binary cross-entropy on token rationale labels
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def rationale_loss(self, rationale_probs, rationale_labels):
        """
        Weighted BCE loss to handle imbalance between rationale/non-rationale.
        Paper eq. (1): Loss_r = -sum_j (n/n_j) * BCE(y_j, p_j)
        """
        B, T = rationale_labels.shape
        total_loss = 0.0

        for b in range(B):
            y = rationale_labels[b]    # (T,)
            p = rationale_probs[b]     # (T,)
            n = T

            # compute per-class weights
            n1 = y.sum().item()        # number of rationale tokens
            n0 = n - n1               # number of non-rationale tokens

            if n1 == 0 or n0 == 0:
                # skip if only one class present
                continue

            # weight for class 1 tokens: n/n1, weight for class 0: n/n0
            weights = torch.where(y == 1,
                                  torch.tensor(n / n1, device=y.device),
                                  torch.tensor(n / n0, device=y.device))

            bce = F.binary_cross_entropy(p, y, weight=weights, reduction="mean")
            total_loss += bce

        return total_loss / B

    def forward(self, stage2_logits, stage1_logits,
                rationale_probs, rationale_labels, class_labels):
        """
        Args:
            stage2_logits   : (B, num_classes)  main prediction
            stage1_logits   : (B, num_classes)  aux stage1 prediction
            rationale_probs : (B, seq_len)       token rationale probabilities
            rationale_labels: (B, seq_len)       ground truth binary token labels
            class_labels    : (B,)               ground truth humanitarian class

        Returns:
            total_loss, loss_l, loss_r
        """
        # classification losses
        loss_l2  = self.ce_loss(stage2_logits, class_labels)
        loss_l1  = self.ce_loss(stage1_logits, class_labels)
        loss_l   = loss_l2 + loss_l1

        # rationale loss
        loss_r   = self.rationale_loss(rationale_probs, rationale_labels)

        total    = loss_l + cfg.ALPHA * loss_r
        return total, loss_l, loss_r


if __name__ == "__main__":
    print("Testing VLTCrisisLoss...")
    loss_fn = VLTCrisisLoss()
    B, T, C = 2, 40, 5
    s2  = torch.randn(B, C)
    s1  = torch.randn(B, C)
    rp  = torch.sigmoid(torch.randn(B, T))
    rl  = torch.zeros(B, T)
    rl[:, 3:7] = 1.0
    cl  = torch.tensor([0, 2])

    total, ll, lr = loss_fn(s2, s1, rp, rl, cl)
    print(f"total={total:.4f}  loss_l={ll:.4f}  loss_r={lr:.4f}")
    print("Loss OK")