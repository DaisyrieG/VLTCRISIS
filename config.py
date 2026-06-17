import os

class Config:

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR        = "/content/drive/MyDrive/MiniDataset/csv"
    TRAIN_FILE      = os.path.join(DATA_DIR, "train.csv")
    DEV_FILE        = os.path.join(DATA_DIR, "dev.csv")
    TEST_FILE       = os.path.join(DATA_DIR, "test.csv")
    IMAGE_DIR       = "/content/drive/MyDrive/MiniDataset/images"
    CHECKPOINT_DIR  = "checkpoints"
    LOG_DIR         = "logs"

    # ── ViLT backbone ─────────────────────────────────────────────────────
    VILT_MODEL_NAME = "dandelin/vilt-b32-mlm"
    MAX_TEXT_LEN    = 40        # max tokens per tweet (paper default)
    IMAGE_SIZE      = 384       # shorter edge resized to 384
    PATCH_SIZE      = 32        # 32x32 pixel patches → 144 patches per image

    # ── Humanitarian classes ───────────────────────────────────────────────
    LABELS = [
        "infrastructure_damage",
        "affected_individuals",
        "rescue_volunteering_or_donation_effort",
        "other_relevant_information",
        "not_humanitarian",
    ]
    NUM_CLASSES = len(LABELS)
    LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL    = {i: l for i, l in enumerate(LABELS)}

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE      = 8
    NUM_EPOCHS      = 10
    LEARNING_RATE   = 0.1
    WEIGHT_DECAY    = 1e-4
    PATIENCE        = 3         # early stopping patience
    ALPHA           = 0.09      # weight of rationale loss: Loss = Loss_l + α·Loss_r

    # ── Text rationale extractor (M3) ──────────────────────────────────────
    GRU_HIDDEN_SIZE = 128
    GRU_LAYERS      = 1

    # ── Image rationale extractor (M5 - IPOT) ─────────────────────────────
    IPOT_ITERATIONS = 50        # number of Sinkhorn iterations
    IPOT_BETA       = 0.5       # regularisation coefficient
    TOP_PATCH_RATIO = 0.25      # keep top 25% patches as rationales

    # ── Device ─────────────────────────────────────────────────────────────
    # Automatically uses GPU if available, otherwise CPU
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Reproducibility ────────────────────────────────────────────────────
    SEED = 42


# Quick sanity check when run directly
if __name__ == "__main__":
    cfg = Config()
    print("Device        :", cfg.DEVICE)
    print("Num classes   :", cfg.NUM_CLASSES)
    print("Labels        :", cfg.LABELS)
    print("Batch size    :", cfg.BATCH_SIZE)
    print("Alpha (α)     :", cfg.ALPHA)
    print("Patch size    :", cfg.PATCH_SIZE)
    print("Config loaded OK")