"""
Step 3: Evaluate the teacher model.

Run the cloud teacher model on one sample page and save its JSON output.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from config import IMAGES_DIR, OUTPUTS_DIR, TEACHER_MODEL
from prompts import TEACHER_PROMPT
from utils import encode_image, parse_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Load the API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMAGE_PATH = IMAGES_DIR / "file_5" / "page_002.jpg"


def run_teacher(client, image_path):
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


def save_result(text, image_path):
    """Save the output as a JSON file."""
    out_dir = OUTPUTS_DIR / "ocr_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # e.g. file_5_page_002_gpt-5.5.json
    out_path = out_dir / f"{image_path.parent.name}_{image_path.stem}_{TEACHER_MODEL}.json"

    data = parse_json(text)

    # ensure_ascii=False keeps the Arabic text readable
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    logger.info("Sending %s to %s", IMAGE_PATH, TEACHER_MODEL)
    response = run_teacher(client, IMAGE_PATH)

    out_path = save_result(response.output_text, IMAGE_PATH)
    logger.info("Saved result to %s", out_path)

    print(response.output_text)


if __name__ == "__main__":
    main()