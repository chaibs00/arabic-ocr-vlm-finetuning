"""
Step 2: Evaluate the base model.

Run the base student model (google/gemma-3-4b-it) on one sample page
and save its Markdown output.
"""

import logging
import os

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import BASE_STUDENT_MODEL, IMAGES_DIR, OUTPUTS_DIR
from prompts import BASELINE_OCR_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Load the Hugging Face token from .env (Gemma is a gated model)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

IMAGE_PATH = IMAGES_DIR / "file_5" / "page_002.jpg"
MAX_NEW_TOKENS = 1024


def load_model():
    """Load the model and its processor."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_STUDENT_MODEL, dtype="auto", device_map="auto", token=HF_TOKEN
    ).eval()
    processor = AutoProcessor.from_pretrained(BASE_STUDENT_MODEL, token=HF_TOKEN)

    return model, processor


def run_ocr(model, processor, image_path):
    """Run OCR on one image and return the model output."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path)},
                {"type": "text", "text": BASELINE_OCR_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    # Keep only the new tokens, not the prompt
    return processor.decode(output[0][input_len:], skip_special_tokens=True)


def save_result(text, image_path):
    """Save the output as a Markdown file."""
    out_dir = OUTPUTS_DIR / "ocr_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # e.g. file_5_page_002_gemma-3-4b-it.md
    model_name = BASE_STUDENT_MODEL.split("/")[-1]
    out_path = out_dir / f"{image_path.parent.name}_{image_path.stem}_{model_name}.md"

    # The prompt opens a ```markdown block, so remove the closing fence
    clean_text = text.strip().removesuffix("```").strip()
    out_path.write_text(clean_text, encoding="utf-8")

    return out_path


def main():
    logger.info("Loading model %s", BASE_STUDENT_MODEL)
    model, processor = load_model()

    logger.info("Running OCR on %s", IMAGE_PATH)
    result = run_ocr(model, processor, IMAGE_PATH)

    out_path = save_result(result, IMAGE_PATH)
    logger.info("Saved result to %s", out_path)

    print(result)


if __name__ == "__main__":
    main()