import os
import sys
import torch
import gradio as gr
from PIL import Image

from config import Config
from modules.preprocessor import Preprocessor
from models.vltcrisis import VLTCrisis

cfg = Config()
prep = Preprocessor()

# ── Load model ──────────────────────────────────────────────────────────────
model = None

def load_model(checkpoint_path=None):
    global model
    model = VLTCrisis().to(cfg.DEVICE)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=cfg.DEVICE)
        )
        return f"Model loaded from {checkpoint_path}"
    else:
        return "Model loaded with random weights (not yet trained)"

def get_model():
    global model
    if model is None:
        load_model()
    return model


# ── Inference function ──────────────────────────────────────────────────────
def predict(text, image, checkpoint_path, sensitivity):
    if not text or text.strip() == "":
        return "Please enter a tweet.", None, "No text provided.", None, None, {}, "—"

    if image is None:
        image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                          (200, 200, 200))

    # Temporarily override the config value for this inference run
    cfg.TOP_PATCH_RATIO = sensitivity

    # load model with checkpoint if provided
    status = load_model(checkpoint_path if checkpoint_path else None)

    m = get_model()
    m.eval()

    # preprocess
    encoding, wt_map, clean_text, pil_image = prep.process(text, image)

    input_ids      = encoding["input_ids"].to(cfg.DEVICE)
    attention_mask = encoding["attention_mask"].to(cfg.DEVICE)
    token_type_ids = encoding["token_type_ids"].to(cfg.DEVICE)
    pixel_values   = encoding["pixel_values"].to(cfg.DEVICE)
    pixel_mask     = encoding["pixel_mask"].to(cfg.DEVICE)

    with torch.no_grad():
        logits, rat_preds, heatmap = m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            training=False,
        )

    # predicted class probabilities
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    class_probs = {cfg.ID2LABEL[i]: p for i, p in enumerate(probs)}

    # text rationales
    token_pred_flat = rat_preds[0]               # (seq_len,)
    word_preds      = prep.tokens_to_words(token_pred_flat, wt_map)
    highlighted     = prep.highlight_rationale_words(clean_text, word_preds)

    # heatmap visualisation on image
    heatmap_img = visualise_heatmap(pil_image, heatmap[0])

    # masked image visualisation (M6)
    masked_img = visualise_masked_image(pil_image, heatmap[0])

    return clean_text, pil_image, highlighted, heatmap_img, masked_img, class_probs, status


def visualise_heatmap(image, heatmap_tensor):
    """
    Overlay patch heatmap on the original image as a colour mask.
    """
    import numpy as np

    try:
        img_arr  = image.resize((384, 384))
        img_np   = np.array(img_arr).astype(float)

        scores   = heatmap_tensor.cpu().numpy()
        n_patches = len(scores)
        grid_size = int(n_patches ** 0.5)        # e.g. 12 for 144 patches

        patch_h  = 384 // grid_size
        patch_w  = 384 // grid_size

        overlay  = np.zeros((384, 384), dtype=float)
        idx      = 0
        for pi in range(grid_size):
            for pj in range(grid_size):
                r0, r1 = pi * patch_h, (pi + 1) * patch_h
                c0, c1 = pj * patch_w, (pj + 1) * patch_w
                if idx < len(scores):
                    overlay[r0:r1, c0:c1] = scores[idx]
                idx += 1

        # red channel highlight
        heatmap_rgb          = np.zeros((384, 384, 3), dtype=float)
        heatmap_rgb[:, :, 0] = overlay * 255   # red = importance
        blend = (0.6 * img_np + 0.4 * heatmap_rgb).clip(0, 255).astype("uint8")
        return Image.fromarray(blend)

    except Exception as e:
        print("Heatmap visualisation error:", e)
        return image


def visualise_masked_image(image, heatmap_tensor):
    """
    Masks out non-rationale patches from the image (M6).
    """
    import numpy as np

    try:
        img_arr  = image.resize((384, 384))
        img_np   = np.array(img_arr).astype(float)

        scores   = heatmap_tensor.cpu().numpy()
        k        = max(1, int(cfg.TOP_PATCH_RATIO * len(scores)))
        keep_idx = np.argsort(scores)[-k:]
        keep_set = set(keep_idx)

        n_patches = len(scores)
        grid_size = int(n_patches ** 0.5)

        patch_h  = 384 // grid_size
        patch_w  = 384 // grid_size

        masked_img = img_np.copy()
        idx = 0
        for pi in range(grid_size):
            for pj in range(grid_size):
                if idx not in keep_set:
                    r0, r1 = pi * patch_h, (pi + 1) * patch_h
                    c0, c1 = pj * patch_w, (pj + 1) * patch_w
                    masked_img[r0:r1, c0:c1] = 128.0  # Grey out non-rationale to match paper
                idx += 1
        return Image.fromarray(masked_img.astype("uint8"))

    except Exception as e:
        print("Masked image error:", e)
        return image

# ── Training launcher ───────────────────────────────────────────────────────
def launch_training(train_csv_obj, dev_csv_obj, img_zip_obj, image_dir, epochs, batch_size, alpha):
    """Run training from the UI."""
    import zipfile
    import tempfile
    import shutil

    def get_path(f):
        if f is None: return None
        if isinstance(f, str): return f
        if hasattr(f, 'name'): return f.name
        return None

    train_path = get_path(train_csv_obj)
    dev_path = get_path(dev_csv_obj)
    zip_path = get_path(img_zip_obj)

    # override config values dynamically
    cfg.TRAIN_FILE  = train_path  if train_path  else cfg.TRAIN_FILE
    cfg.DEV_FILE    = dev_path    if dev_path    else cfg.DEV_FILE
    
    if zip_path:
        # Extract uploaded zip to a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="vlt_images_")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        cfg.IMAGE_DIR = temp_dir
    else:
        cfg.IMAGE_DIR = image_dir if image_dir else cfg.IMAGE_DIR

    cfg.NUM_EPOCHS  = int(epochs)
    cfg.BATCH_SIZE  = int(batch_size)
    cfg.ALPHA       = float(alpha)

    # check files exist
    missing = []
    for f in [cfg.TRAIN_FILE, cfg.DEV_FILE]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        yield "Error: Missing dataset files!\n" + "\n".join(missing) + "\n\nDid you forget to upload your dataset?"
        return

    yield "Starting training... (check your terminal for live logs)\n"

    try:
        from train import train
        train()
        yield "Training complete! Check checkpoints/best_model.pt"
    except Exception as e:
        yield f"Training error: {e}"


# ── Data checker ────────────────────────────────────────────────────────────
def check_data(train_csv, dev_csv, test_csv, image_dir):
    import pandas as pd
    lines = []

    for name, path in [("train", train_csv), ("dev", dev_csv),
                        ("test", test_csv)]:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            lines.append(f"{name}.csv  →  {len(df)} rows")
            lines.append(f"  Columns: {list(df.columns)}")
            if "label" in df.columns:
                lines.append(f"  Class distribution:")
                for label, count in df["label"].value_counts().items():
                    lines.append(f"    {label}: {count}")
        else:
            lines.append(f"{name}.csv  →  NOT FOUND at {path}")

    if image_dir and os.path.exists(image_dir):
        imgs = [f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        lines.append(f"\nImage dir → {len(imgs)} images found")
    else:
        lines.append(f"\nImage dir NOT FOUND at {image_dir}")

    return "\n".join(lines)


def load_real_examples():
    try:
        import pandas as pd
        import config as cfg
        import os
        
        # Check default path first, then check Google Drive MiniDataset
        dev_file = cfg.DEV_FILE
        img_dir = cfg.IMAGE_DIR
        
        if not os.path.exists(dev_file):
            dev_file = "/content/drive/MyDrive/MiniDataset/csv/test.csv"
            img_dir = "/content/drive/MyDrive/MiniDataset/images"
            
        if os.path.exists(dev_file):
            df = pd.read_csv(dev_file).dropna(subset=['tweet_text', 'image_path'])
            # Grab up to 50 random rows for a good variety without crashing the UI
            n_samples = min(50, len(df))
            samples = df.sample(n=n_samples, random_state=42)
            ex_list = []
            for _, row in samples.iterrows():
                img_path = os.path.join(img_dir, row['image_path'])
                if os.path.exists(img_path):
                    ex_list.append([row['tweet_text'], img_path])
            if len(ex_list) > 0:
                return ex_list
    except:
        pass
    return [
        ["BREAKING: Massive flooding hits downtown Houston after severe storm. The water is rising fast and emergency responders are on boats! 🚨 #HoustonFlood #Emergency", "examples/flood.png"],
        ["Devastating aftermath of the 7.8 magnitude earthquake in the city center. Buildings collapsed and rescue teams are on site. 🙏 #Earthquake #Disaster", "examples/earthquake.png"],
        ["URGENT: Wildfires spreading rapidly across the hills, threatening local residential homes. Please evacuate immediately! 🔥 #WildfireAlert #Evacuate", "examples/wildfire.png"],
    ]

# ── Build UI ────────────────────────────────────────────────────────────────
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#1e40af",
    panel_background_fill="#ffffff",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}
.dark {
    background-color: #f8fafc !important;
}

/* Style tabs to look like a modern dashboard navigation */
div.tabs > div.tab-nav {
    border-bottom: 1px solid #e2e8f0 !important;
    padding-left: 20px !important;
    background-color: #ffffff !important;
    padding-top: 15px !important;
}
div.tabs > div.tab-nav > button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #64748b !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease;
}
div.tabs > div.tab-nav > button.selected {
    color: #1d4ed8 !important;
    border-bottom: 3px solid #1d4ed8 !important;
    background: #eff6ff !important;
}
div.tabs > div.tab-nav > button:hover:not(.selected) {
    background: #f1f5f9 !important;
}

.figma-card {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
    padding: 24px !important;
    margin-bottom: 16px !important;
}
.figma-header {
    background-color: transparent !important;
    border-radius: 0px !important;
    padding: 8px 0px 16px 0px !important;
    margin-bottom: 8px !important;
    border: none !important;
    box-shadow: none !important;
}
h3 {
    color: #475569 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 12px !important;
}
.gr-button-primary {
    border-radius: 6px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
"""

def build_ui():
    with gr.Blocks(title="VLTCrisis — AI System", theme=custom_theme, css=custom_css) as demo:
        
        # This forces the browser to drop dark mode classes if Gradio applies them
        demo.load(None, js="""
            function() {
                document.body.classList.remove('dark');
            }
        """)

        with gr.Tabs():
            with gr.Tab("Interactive Analysis"):
                with gr.Row():
                    # ── LEFT COLUMN (INPUTS) ──
                    with gr.Column(scale=1):
                        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("""
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="background: #1d4ed8; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; font-weight: bold; letter-spacing: -1px;">V</div>
                                <div>
                                    <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #111827;">VLTCrisis AI System</h2>
                                    <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">Cross-Modal Rationale Transfer · AAAI 2024</p>
                                </div>
                            </div>
                            """)
        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("### TWEET INPUT")
                            tweet_input = gr.Textbox(
                                label="",
                                placeholder="Paste a crisis-related tweet here... e.g. 'Flash floods hit downtown Houston...'",
                                lines=4,
                                show_label=False
                            )
                        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("### UPLOAD CRISIS IMAGE")
                            image_input = gr.Image(
                                label="",
                                type="pil",
                                show_label=False
                            )
                        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("### EXPLAINABLE AI (XAI) SETTINGS")
                            gr.Markdown("<p style='font-size: 0.8rem; color: #6b7280; margin-bottom: 5px;'>Lower value = highlights only the most critical damage. Higher value = shows broader context.</p>")
                            sensitivity_slider = gr.Slider(
                                minimum=0.05, maximum=1.0, value=0.25, step=0.05,
                                label="Rationale Sensitivity (Top % of patches to keep)"
                            )
                        
                        ckpt_input  = gr.Textbox(
                            label="Checkpoint Path",
                            value="checkpoints/best_model.pt",
                            visible=False 
                        )
                        
                        predict_btn = gr.Button("▶ Run Pipeline Analysis", variant="primary", size="lg")
        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("### EXAMPLE SCENARIOS")
                            gr.Examples(
                                examples=load_real_examples(),
                                inputs=[tweet_input, image_input],
                                examples_per_page=5
                            )
        
                    # ── RIGHT COLUMN (PIPELINE ARCHITECTURE) ──
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes="figma-header"):
                            gr.Markdown("""
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #111827;">Final Prediction & Explanation</h2>
                                    <p style="margin: 0; font-size: 0.85rem; color: #6b7280;">See what the AI thinks and why</p>
                                </div>
                            </div>
                            """)
                        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                <div style="background: #dbeafe; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #1e40af; font-weight: bold;">1</div>
                                <div>
                                    <h2 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #111827;">Class Confidence <span style="background: #e2e8f0; color: #475569; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">FINAL</span></h2>
                                    <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">What type of disaster is this?</p>
                                </div>
                            </div>
                            """)
                            out_class = gr.Label(label="", num_top_classes=3, show_label=False)
                            out_status = gr.Textbox(label="System Log", lines=1, visible=False)

                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                <div style="background: #1d4ed8; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">✓</div>
                                <div>
                                    <h2 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #111827;">Text Analysis <span style="background: #e0e7ff; color: #3730a3; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">WORDS</span></h2>
                                    <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">Which words mattered most for this prediction?</p>
                                </div>
                            </div>
                            """)
                            gr.Markdown("### IMPORTANT WORDS")
                            out_text_rat = gr.Textbox(label="", lines=3, show_label=False)
        
                        with gr.Column(elem_classes="figma-card"):
                            gr.Markdown("""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                <div style="background: #1d4ed8; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">✓</div>
                                <div>
                                    <h2 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #111827;">Image Analysis <span style="background: #e0e7ff; color: #3730a3; font-size: 0.6rem; padding: 2px 6px; border-radius: 4px; margin-left: 8px;">VISUAL</span></h2>
                                    <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">Where is the structural damage?</p>
                                </div>
                            </div>
                            """)
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### DAMAGE HEATMAP")
                                    out_img_rat = gr.Image(label="", height=250, show_label=False)
                                with gr.Column():
                                    gr.Markdown("### MASKED IMAGE (STRICT)")
                                    out_masked_img = gr.Image(label="", height=250, show_label=False)

                        with gr.Accordion("Advanced: Show Technical Preprocessing (M1)", open=False):
                            with gr.Column(elem_classes="figma-card"):
                                gr.Markdown("""
                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #111827;">M1 · Multimodal Preprocessor</h2>
                                        <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">Tokenizes text & normalizes image tensors</p>
                                    </div>
                                </div>
                                """)
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("### TEXT TOKENS")
                                        out_clean_text = gr.Textbox(label="", lines=4, show_label=False)
                                    with gr.Column():
                                        gr.Markdown("### NORMALIZED IMAGE")
                                        out_clean_img = gr.Image(label="", height=200, show_label=False)
        
                predict_btn.click(
                    fn=predict,
                    inputs=[tweet_input, image_input, ckpt_input, sensitivity_slider],
                    outputs=[out_clean_text, out_clean_img, out_text_rat, out_img_rat, out_masked_img, out_class, out_status],
                )
        
        # ── ADDITIONAL TABS FOR DEV/TRAINING ──
            with gr.Tab("Dataset & Pipeline Training"):
                with gr.Tabs():
                    with gr.Tab("Check Dataset"):
                        gr.Markdown("Verify your CrisisMMD data files are correctly formatted.")

                        with gr.Row():
                            dc_train = gr.Textbox(label="Train CSV path",
                                                  value=cfg.TRAIN_FILE)
                            dc_dev   = gr.Textbox(label="Dev CSV path",
                                                  value=cfg.DEV_FILE)
                            dc_test  = gr.Textbox(label="Test CSV path",
                                                  value=cfg.TEST_FILE)
                            dc_imgs  = gr.Textbox(label="Image directory",
                                                  value=cfg.IMAGE_DIR)
    
                        dc_btn    = gr.Button("Check Data", variant="primary")
                        dc_output = gr.Textbox(label="Data summary", lines=20,
                                               interactive=False)
    
                        dc_btn.click(
                            fn=check_data,
                            inputs=[dc_train, dc_dev, dc_test, dc_imgs],
                            outputs=dc_output,
                        )
    
                    # ── Tab 3: Training ────────────────────────────────────────────
                    with gr.Tab("Train Model"):
                        gr.Markdown("Configure and launch training. Logs appear in your terminal.")
    
                        with gr.Row():
                            with gr.Column():
                                tr_train  = gr.File(label="Upload Train CSV", file_types=[".csv"])
                                tr_dev    = gr.File(label="Upload Dev CSV", file_types=[".csv"])
                                tr_zip    = gr.File(label="Upload Images (ZIP) [Optional]", file_types=[".zip"])
                                tr_imgs   = gr.Textbox(label="Or provide Image directory path",
                                                       value=cfg.IMAGE_DIR)
                            with gr.Column():
                                tr_epochs = gr.Slider(1, 20, value=10, step=1,
                                                      label="Epochs")
                                tr_batch  = gr.Slider(2, 32, value=8,  step=2,
                                                      label="Batch size")
                                tr_alpha  = gr.Slider(0.01, 1.0, value=0.09,
                                                      step=0.01,
                                                      label="Alpha (rationale loss weight)")
    
                        tr_btn    = gr.Button("Start Training", variant="primary")
                        tr_output = gr.Textbox(label="Training status", lines=5,
                                               interactive=False)
    
                        tr_btn.click(
                            fn=launch_training,
                            inputs=[tr_train, tr_dev, tr_zip, tr_imgs,
                                     tr_epochs, tr_batch, tr_alpha],
                            outputs=tr_output,
                        )
    
                    # ── Tab 4: Evaluation ───────────────────────────────────────────
                    with gr.Tab("Evaluate Dataset"):
                        gr.Markdown("Run the trained pipeline across the full dataset to get F1 metrics automatically.")
                        
                        with gr.Row():
                            ev_csv  = gr.Textbox(label="Evaluation CSV", value=cfg.TEST_FILE)
                            ev_imgs = gr.Textbox(label="Image directory", value=cfg.IMAGE_DIR)
                            
                        ev_btn    = gr.Button("▶ Run Automatic Evaluation", variant="primary")
                        ev_output = gr.Textbox(label="Results", lines=5, interactive=False)
    
                        def run_eval_ui(csv_path, img_dir, ckpt_path):
                            yield "Loading model and dataset..."
                            try:
                                from dataset import get_dataloader
                                from evaluate import evaluate_model
                                
                                # Temporarily override config image dir for evaluation
                                cfg.IMAGE_DIR = img_dir
                                
                                status = load_model(ckpt_path)
                                m = get_model()
                                
                                # get_dataloader only takes csv_file, split, shuffle
                                loader = get_dataloader(csv_path, split="test", shuffle=False)
                                yield f"Running evaluation on {len(loader.dataset)} samples. Please wait..."
                                
                                mf1, tf1 = evaluate_model(m, loader, cfg.DEVICE)
                                yield f"Evaluation Complete!\nStatus: {status}\nClassification Macro-F1: {mf1:.4f}\nRationale Token-F1: {tf1:.4f}"
                            except Exception as e:
                                yield f"Error during evaluation: {e}"
    
                        ev_btn.click(
                            fn=run_eval_ui,
                            inputs=[ev_csv, ev_imgs, ckpt_input],
                            outputs=ev_output,
                        )
    
                    # ── Tab 5: About ───────────────────────────────────────────────
                    with gr.Tab("About"):
                        gr.Markdown(f"""
                        ## VLTCrisis System
    
                        **Architecture:** 7-module pipeline based on ViLT (Vision-and-Language Transformer)
    
                        | Module | Role |
                        |--------|------|
                        | M1 Preprocessor | Clean text, resize image, tokenize, patch |
                        | M2 ViLT Encoder | Joint text+image representation |
                        | M3 Text Rationale | GRU + Sigmoid — which tokens matter |
                        | M4 Aux Classifier | Stage 1 class head (training only) |
                        | M5 Image Rationale | IPOT cross-modal alignment → patch heatmap |
                        | M6 Masker | Zero out non-rationale tokens and patches |
                        | M7 Stage 2 Classifier | Final prediction on rationale-only input |
    
                        **Classes:**
                        {chr(10).join(f'- {l}' for l in cfg.LABELS)}
    
                        **Device:** {cfg.DEVICE}
    
                        **Paper:** Nguyen et al., WWW 2026
                        """)
    
        return demo
    
    
# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting VLTCrisis UI...")
    print(f"Device: {cfg.DEVICE}")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,         # set True to get a public Gradio link
        inbrowser=True,      # auto-opens browser
    )