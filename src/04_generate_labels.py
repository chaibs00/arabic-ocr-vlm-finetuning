"""
Step 4: Generate labels (knowledge distillation).

Send every page image to the teacher model and save each output
as one line in a JSONL file. Stop when the cost reaches the budget.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from config import IMAGES_DIR, LABELS_FILE, MAX_BUDGET, PRICE_INPUT_1M, PRICE_OUTPUT_1M, TEACHER_MODEL
from prompts import TEACHER_PROMPT
from utils import encode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Load the API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_labeled_images():
    """Return the images already labeled, so a rerun does not pay for them again."""
    if not LABELS_FILE.exists():
        return set()

    with open(LABELS_FILE, encoding="utf-8") as f:
        return {json.loads(line)["image_path"] for line in f if line.strip()}


def label_image(client, image_path):
    """Send one image to the teacher model and return the response."""
    base64_image = encode_image(image_path)

    return client.responses.create(
        model=TEACHER_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": TEACHER_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
    )


def compute_cost(input_tokens, output_tokens):
    """Return the cost in dollars."""
    return (input_tokens / 1_000_000) * PRICE_INPUT_1M + (output_tokens / 1_000_000) * PRICE_OUTPUT_1M


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Absolute paths, so LlamaFactory can find the images later
    all_images = sorted(p.resolve() for p in IMAGES_DIR.glob("*/*.jpg"))
    labeled = get_labeled_images()
    todo = [img for img in all_images if str(img) not in labeled]

    logger.info("%d images found, %d already labeled, %d to do", len(all_images), len(labeled), len(todo))

    total_cost = 0.0  # cost of this run only

    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        for i, image_path in enumerate(todo, start=1):
            response = label_image(client, image_path)

            record = {
                "pdf_name": image_path.parent.name,
                "image_path": str(image_path),
                "model_id": TEACHER_MODEL,
                "output": response.output_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_cost += compute_cost(response.usage.input_tokens, response.usage.output_tokens)
            logger.info("[%d/%d] %s/%s | cost: $%.4f", i, len(todo), record["pdf_name"], image_path.name, total_cost)

            if total_cost >= MAX_BUDGET:
                logger.warning("Budget of $%.2f reached. Stopping.", MAX_BUDGET)
                break

    logger.info("Labels saved to %s", LABELS_FILE)


if __name__ == "__main__":
    main()