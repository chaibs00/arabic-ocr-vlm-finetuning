from pathlib import Path

# Project
PROJECT_ROOT = Path(__file__).resolve().parent


# Data
DATA_DIR = PROJECT_ROOT / "data"
PDFS_DIR = DATA_DIR / "pdfs"
IMAGES_DIR = DATA_DIR / "images"
OUTPUTS_DIR = DATA_DIR / "outputs"


# Step 4: label generation
LABELS_FILE = DATA_DIR / "labels" / "teacher_labels.jsonl"
MAX_BUDGET = 3.00          # dollars
PRICE_INPUT_1M = 5.00      # $ per 1M input tokens
PRICE_OUTPUT_1M = 30.00    # $ per 1M output tokens


# Step 5: dataset
DATASET_DIR = DATA_DIR / "datasets"
VAL_PDFS = ["file_1"]      # PDFs kept for validation



# Models
# Base model that will be evaluted first then improved with the teacher model
BASE_STUDENT_MODEL = "google/gemma-3-4b-it"  
TEACHER_MODEL = "gpt-5.5"


# Generation
TEACHER_BUDGET_USD = 5.0


# Validation
VALIDATION_PDFS = [
    "pdf_01.pdf",
    "pdf_02.pdf",
]