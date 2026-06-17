import re
import os
import torch
from typing import List, Dict, Tuple, Any, Union
from PIL import Image
from transformers import ViltProcessor
from config import Config

cfg = Config()


class Preprocessor:
    """
    M1 — Data Preprocessor.

    Handles ALL cleaning and formatting before anything enters ViLT.
    """

    def __init__(self):
        self.processor = ViltProcessor.from_pretrained(cfg.VILT_MODEL_NAME)

    # ── Text cleaning ───────────────────────────────────────────────────────
    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        # remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # remove user mentions (@username)
        text = re.sub(r"@\w+", "", text)
        # remove RT prefix
        text = re.sub(r"^rt\s+", "", text)
        # remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Image loading & validation ──────────────────────────────────────────
    @staticmethod
    def load_image(image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Loads and validates an image from a filepath or a PIL Image.
        Returns a normalized RGB image, falling back to a blank white image if not found.
        """
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                image = Image.open(image_input).convert("RGB")
            else:
                print(f"Warning: image not found at {image_input}, using blank.")
                image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                  (255, 255, 255))
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                              (255, 255, 255))

        # resize shorter edge to IMAGE_SIZE, keep aspect ratio
        w, h   = image.size
        scale  = cfg.IMAGE_SIZE / min(w, h)
        new_w  = min(int(w * scale), 640)
        new_h  = min(int(h * scale), 640)
        # Use modern Pillow resampling to prevent deprecation warnings
        resample_filter = getattr(Image, "Resampling", Image).BILINEAR
        image  = image.resize((new_w, new_h), resample_filter)
        return image

    # ── Word → token mapping ────────────────────────────────────────────────
    def build_word_token_map(self, text: str) -> List[List[int]]:
        """
        Returns a list where index i contains the list of token positions
        that correspond to word i in the original text.
        This is needed to recover word-level rationales from token-level preds.
        """
        tokenizer   = self.processor.tokenizer
        words       = text.split()
        word_token_map = []
        token_offset   = 1   # skip [CLS] token at position 0

        for word in words:
            tokens   = tokenizer.tokenize(word)
            n_tokens = len(tokens)
            indices  = list(range(token_offset, token_offset + n_tokens))
            word_token_map.append(indices)
            token_offset += n_tokens

        return word_token_map

    # ── Main process method ─────────────────────────────────────────────────
    def process(self, text: str, image_input: Union[str, Image.Image]) -> Tuple[Dict[str, torch.Tensor], List[List[int]], str, Image.Image]:
        """
        Full preprocessing pipeline for a single tweet.

        Returns:
            encoding       : dict with input_ids, attention_mask,
                             token_type_ids, pixel_values, pixel_mask
                             all as (1, ...) tensors
            word_token_map : list of lists (word → token indices)
            clean_text     : cleaned text string
            image          : loaded PIL image
        """
        clean  = self.clean_text(text)
        image  = self.load_image(image_input)
        wt_map = self.build_word_token_map(clean)

        encoding = self.processor(
            images=image,
            text=clean,
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_TEXT_LEN,
            return_tensors="pt",
        )

        return encoding, wt_map, clean, image

    def process_batch(self, texts: List[str], image_inputs: List[Union[str, Image.Image]]) -> Tuple[Dict[str, torch.Tensor], List[List[List[int]]], List[str], List[Image.Image]]:
        """
        Process a list of tweets and images into batched tensors.
        """
        clean_texts = [self.clean_text(t) for t in texts]
        images      = [self.load_image(img) for img in image_inputs]

        encoding = self.processor(
            images=images,
            text=clean_texts,
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_TEXT_LEN,
            return_tensors="pt",
        )

        wt_maps = [self.build_word_token_map(t) for t in clean_texts]
        return encoding, wt_maps, clean_texts, images

    # ── Token → word rationale recovery ────────────────────────────────────
    @staticmethod
    def tokens_to_words(token_preds: Union[List[int], torch.Tensor], word_token_map: List[List[int]]) -> List[int]:
        """
        Convert token-level binary predictions back to word-level using
        max-pooling (a word is a rationale if ANY of its tokens are predicted
        as rationale).

        Args:
            token_preds    : (seq_len,) binary tensor or list
            word_token_map : list of lists mapping words to token indices

        Returns:
            word_preds : list of 0/1 per word indicating rationale presence
        """
        word_preds = []
        for token_indices in word_token_map:
            vals = [token_preds[i].item() for i in token_indices
                    if i < len(token_preds)]
            word_preds.append(1 if any(v == 1 for v in vals) else 0)
        return word_preds

    @staticmethod
    def highlight_rationale_words(text: str, word_preds: List[int]) -> str:
        """
        Returns a string with rationale words wrapped in [[ ]].
        Useful for display/debugging.
        """
        words  = text.split()
        result = []
        for i, word in enumerate(words):
            if i < len(word_preds) and word_preds[i] == 1:
                result.append(f"[[{word}]]")
            else:
                result.append(word)
        return " ".join(result)


if __name__ == "__main__":
    print("Testing Preprocessor...")
    prep = Preprocessor()

    sample_text  = "RT @USER: Hurricane Maria ripped the roofs off these homes in San Juan http://t.co/example"
    sample_image = Image.new("RGB", (400, 300), color=(100, 150, 200))

    encoding, wt_map, clean, image = prep.process(sample_text, sample_image)

    print("Clean text     :", clean)
    print("Word-token map :", wt_map[:5], "...")
    print("input_ids shape:", encoding["input_ids"].shape)
    print("pixel_values   :", encoding["pixel_values"].shape)

    # test highlight
    fake_word_preds = [0] * len(clean.split())
    fake_word_preds[2] = 1
    fake_word_preds[3] = 1
    highlighted = prep.highlight_rationale_words(clean, fake_word_preds)
    print("Highlighted    :", highlighted)
    print("Preprocessor OK")