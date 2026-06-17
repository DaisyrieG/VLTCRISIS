import torch
import numpy as np
from sklearn.metrics import f1_score
from config import Config

cfg = Config()


def macro_f1(all_preds, all_labels):
    """
    Macro-F1 across all humanitarian classes.
    Args:
        all_preds  : list of predicted class indices
        all_labels : list of ground truth class indices
    """
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


def token_f1(pred_rationales, true_rationales, attention_mask=None):
    """
    Token-level F1 for text rationale extraction.
    Args:
        pred_rationales : (B, seq_len) binary predictions
        true_rationales : (B, seq_len) binary ground truth
        attention_mask  : (B, seq_len) 1 for real tokens, 0 for padding
    """
    pred = pred_rationales.cpu().numpy().flatten()
    true = true_rationales.cpu().numpy().flatten()

    if attention_mask is not None:
        mask = attention_mask.cpu().numpy().flatten().astype(bool)
        pred = pred[mask]
        true = true[mask]

    return f1_score(true, pred, average="binary", zero_division=0)


def evaluate_model(model, dataloader, device):
    """
    Run full evaluation on a dataloader.
    Returns macro_f1 score and token_f1 score.
    """
    model.eval()
    all_preds, all_labels = [], []
    all_rat_preds, all_rat_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            pixel_mask     = batch["pixel_mask"].to(device)
            labels         = batch["label"].to(device)
            rat_labels     = batch["rationale_labels"]

            logits, rat_preds, heatmap = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                training=False,
            )

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            all_rat_preds.append(rat_preds.cpu())
            all_rat_labels.append(rat_labels)

    mf1 = macro_f1(all_preds, all_labels)

    rat_preds_cat  = torch.cat(all_rat_preds,  dim=0)
    rat_labels_cat = torch.cat(all_rat_labels, dim=0)
    tf1 = token_f1(rat_preds_cat, rat_labels_cat)

    return mf1, tf1


def comprehensiveness(model, dataloader, device):
    """
    Comprehensiveness = Performance(X) - Performance(X\R)
    X\R = non-rationale input only (rationales masked out).
    Higher is better.
    """
    from modules.masker import RationaleMasker, MASK_TOKEN_ID

    model.eval()
    all_preds_x, all_preds_xr, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            pixel_mask     = batch["pixel_mask"].to(device)
            labels         = batch["label"]

            # full input prediction
            logits_x, rat_preds, heatmap = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                training=False,
            )

            # invert rationale mask → keep only non-rationale
            inv_preds  = 1.0 - rat_preds
            inv_heat   = 1.0 - heatmap
            xr_ids, xr_pix = RationaleMasker.apply(
                input_ids, pixel_values, inv_preds, inv_heat
            )

            from modules.encoder import ViltEncoder
            # re-use encoder via model
            s2_out = model.encoder(
                input_ids=xr_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=xr_pix,
                pixel_mask=pixel_mask,
            )
            logits_xr = model.stage2_clf(s2_out["pooler_output"])

            all_preds_x.extend(logits_x.argmax(-1).cpu().tolist())
            all_preds_xr.extend(logits_xr.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())

    perf_x  = macro_f1(all_preds_x,  all_labels)
    perf_xr = macro_f1(all_preds_xr, all_labels)
    return perf_x - perf_xr


def sufficiency(model, dataloader, device):
    """
    Sufficiency = Performance(X) - Performance(R)
    R = rationale-only input.
    Lower is better (rationales alone are sufficient).
    """
    model.eval()
    all_preds_x, all_preds_r, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            pixel_mask     = batch["pixel_mask"].to(device)
            labels         = batch["label"]

            # stage2_logits already uses rationale-only (R) internally
            logits_x, rat_preds, heatmap = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                training=False,
            )

            # full-input prediction via aux encoder pass
            s1_out   = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
            )
            logits_full = model.stage2_clf(s1_out["pooler_output"])

            all_preds_x.extend(logits_full.argmax(-1).cpu().tolist())
            all_preds_r.extend(logits_x.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())

    perf_x = macro_f1(all_preds_x, all_labels)
    perf_r = macro_f1(all_preds_r, all_labels)
    return perf_x - perf_r


if __name__ == "__main__":
    print("Evaluate module loaded OK")
    preds  = [0, 1, 2, 3, 4, 0, 1]
    labels = [0, 1, 2, 2, 4, 0, 2]
    print("Macro-F1 test:", macro_f1(preds, labels))