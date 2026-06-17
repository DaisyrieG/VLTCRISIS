import os
import ast
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor
from config import Config

cfg = Config()


class CrisisMMDDataset(Dataset):
    """
    Loads CrisisMMD CSV files.
    Expected CSV columns:
        text        : raw tweet string
        image_path  : filename of the image (inside cfg.IMAGE_DIR)
        label       : humanitarian class string
        rationales  : space-separated token indices that are rationales
                      (optional, only in train/dev)
    """

    def __init__(self, csv_file, split="train"):
        self.df        = pd.read_csv(csv_file)
        self.split     = split
        self.image_dir = cfg.IMAGE_DIR
        self.processor = ViltProcessor.from_pretrained(cfg.VILT_MODEL_NAME)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Text ───────────────────────────────────────────────────────────
        text = str(row["text"])
        # basic cleaning (URLs, mentions already stripped in preprocessor
        # but we do a quick safety clean here too)
        text = text.lower().strip()

        # ── Image ──────────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, str(row["image_path"]))
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # fallback: blank white image if file missing
            image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), (255, 255, 255))

        # ── ViLT encoding ──────────────────────────────────────────────────
        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_TEXT_LEN,
            return_tensors="pt",
        )
        # squeeze batch dim added by processor
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # ── Class label ────────────────────────────────────────────────────
        label_str     = str(row["label"])
        item["label"] = torch.tensor(
            cfg.LABEL2ID.get(label_str, 0), dtype=torch.long
        )

        # ── Text rationale labels (token-level binary) ─────────────────────
        # rationale column contains space-separated token indices, e.g. "3 4 7"
        num_tokens = item["input_ids"].shape[0]
        rationale_labels = torch.zeros(num_tokens, dtype=torch.float)

        if "rationales" in self.df.columns and pd.notna(row.get("rationales")):
            try:
                indices = [int(i) for i in str(row["rationales"]).split()]
                for i in indices:
                    if i < num_tokens:
                        rationale_labels[i] = 1.0
            except Exception:
                pass

        item["rationale_labels"] = rationale_labels
        return item


def get_dataloader(csv_file, split="train", shuffle=True):
    dataset = CrisisMMDDataset(csv_file, split=split)
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,       # 0 is safest on Windows
        pin_memory=(cfg.DEVICE == "cuda"),
    )


if __name__ == "__main__":
    print("Testing dataset loader...")
    loader = get_dataloader(cfg.TRAIN_FILE, split="train")
    batch  = next(iter(loader))
    print("input_ids shape      :", batch["input_ids"].shape)
    print("pixel_values shape   :", batch["pixel_values"].shape)
    print("labels               :", batch["label"])
    print("rationale_labels shape:", batch["rationale_labels"].shape)
    print("Dataset OK")