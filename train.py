import os
import torch
import random
import numpy as np
from torch.optim import AdamW

from config import Config
from dataset import get_dataloader
from models.vltcrisis import VLTCrisis
from losses import VLTCrisisLoss
from evaluate import evaluate_model

cfg = Config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader = get_dataloader(cfg.TRAIN_FILE, split="train", shuffle=True)
    dev_loader   = get_dataloader(cfg.DEV_FILE,   split="dev",   shuffle=False)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches  : {len(dev_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model...")
    model    = VLTCrisis().to(cfg.DEVICE)
    loss_fn  = VLTCrisisLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_macro_f1  = 0.0
    patience_count = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            token_type_ids = batch["token_type_ids"].to(cfg.DEVICE)
            pixel_values   = batch["pixel_values"].to(cfg.DEVICE)
            pixel_mask     = batch["pixel_mask"].to(cfg.DEVICE)
            class_labels   = batch["label"].to(cfg.DEVICE)
            rat_labels     = batch["rationale_labels"].to(cfg.DEVICE)

            # forward
            s2_logits, s1_logits, rat_probs, heatmap = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                training=True,
            )

            # loss
            total_loss, loss_l, loss_r = loss_fn(
                s2_logits, s1_logits, rat_probs, rat_labels, class_labels
            )

            # backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Epoch {epoch} | Step {step+1} | "
                      f"loss={total_loss:.4f} "
                      f"loss_l={loss_l:.4f} "
                      f"loss_r={loss_r:.4f}")

            # ── Save Visual Sample ─────────────────────────────────────────────
            # Save a sample image and its heatmap at the end of the epoch
            if step == len(train_loader) - 1:
                try:
                    import torchvision
                    sample_dir = os.path.join(cfg.CHECKPOINT_DIR, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    sample_hm = heatmap[0].detach().cpu()
                    n_patches = len(sample_hm)
                    grid_size = int(n_patches ** 0.5)
                    
                    if grid_size * grid_size == n_patches:
                        hm_img = sample_hm.view(1, 1, grid_size, grid_size)
                        # Resize heatmap to match image size (384x384)
                        hm_img_resized = torch.nn.functional.interpolate(hm_img, size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), mode='bilinear')
                        
                        img_tensor = pixel_values[0].detach().cpu()
                        # ViLT standard un-normalization (approximate for display)
                        img_tensor = img_tensor * 0.5 + 0.5 
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Add heatmap as a red overlay
                        overlay = img_tensor.clone()
                        overlay[0] = torch.clamp(overlay[0] + hm_img_resized[0, 0] * 0.6, 0, 1)
                        
                        # Concatenate side by side (Original | Heatmap)
                        combined = torch.cat([img_tensor, overlay], dim=2)
                        
                        save_path = os.path.join(sample_dir, f"epoch_{epoch}_sample.jpg")
                        torchvision.utils.save_image(combined, save_path)
                except ImportError:
                    pass # Skip if torchvision is not available

        avg_loss = epoch_loss / len(train_loader)

        # ── Evaluation ─────────────────────────────────────────────────────
        mf1, tf1 = evaluate_model(model, dev_loader, cfg.DEVICE)
        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS} "
              f"| avg_loss={avg_loss:.4f} "
              f"| dev Macro-F1={mf1:.4f} "
              f"| dev Token-F1={tf1:.4f}")

        # ── Checkpoint & early stopping ────────────────────────────────────
        if mf1 > best_macro_f1:
            best_macro_f1  = mf1
            patience_count = 0
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model (Macro-F1={mf1:.4f}) → {ckpt_path}")
        else:
            patience_count += 1
            print(f"  No improvement. Patience {patience_count}/{cfg.PATIENCE}")
            if patience_count >= cfg.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best dev Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    train()