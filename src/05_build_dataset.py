"""
Step 5: Build the fine-tuning dataset.

Split each teacher label into two tasks (content and metadata),
divide the data into train/val sets by PDF, and save it in ShareGPT format.
"""

import json
import logging
import random

from config import DATASET_DIR, LABELS_FILE, VAL_PDFS
from prompts import TASK1_PROMPT, TASK2_PROMPT
from utils import parse_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def split_tasks(label):
    """Split one teacher label into task 1 (content) and task 2 (metadata)."""
    task1 = {
        "content": label.pop("content"),
        "structural_elements": label.pop("structural_elements", ""),
    }
    task2 = label  # all the remaining fields
    return task1, task2


def make_record(prompt, output, image_path):
    """Make one sample in ShareGPT format for LlamaFactory."""
    return {
        "conversations": [
            {"from": "human", "value": "<image>" + prompt},
            {"from": "gpt", "value": json.dumps(output, ensure_ascii=False, default=str)},
        ],
        "images": [image_path],
    }


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def main():
    train_ds, val_ds = [], []
    seen_images = set()
    skipped = 0

    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)
            label = parse_json(rec["output"])

            # Skip broken outputs and duplicate images
            if not label or "content" not in label or rec["image_path"] in seen_images:
                skipped += 1
                continue
            seen_images.add(rec["image_path"])

            task1, task2 = split_tasks(label)
            records = [
                make_record(TASK1_PROMPT, task1, rec["image_path"]),
                make_record(TASK2_PROMPT, task2, rec["image_path"]),
            ]

            # Split by PDF, so pages of one document are never in both sets
            if rec["pdf_name"] in VAL_PDFS:
                val_ds.extend(records)
            else:
                train_ds.extend(records)

    random.Random(101).shuffle(train_ds)
    random.Random(101).shuffle(val_ds)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    save_json(train_ds, DATASET_DIR / "train.json")
    save_json(val_ds, DATASET_DIR / "val.json")

    logger.info("Train: %d samples | Val: %d samples | Skipped: %d", len(train_ds), len(val_ds), skipped)
    logger.info("Saved to %s", DATASET_DIR)


if __name__ == "__main__":
    main()