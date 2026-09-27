from pathlib import Path

# Project
PROJECT_ROOT = Path(__file__).resolve().parent


# Data
DATA_DIR = PROJECT_ROOT / "data"

PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"


# Models
# Base model that will be evaluted first then improved with the teacher model
STUDENT_MODEL = "google/gemma-3-4b-it"  
TEACHER_MODEL = "..."


# Generation
TEACHER_BUDGET_USD = 5.0


# Validation
VALIDATION_PDFS = [
    "pdf_01.pdf",
    "pdf_02.pdf",
]