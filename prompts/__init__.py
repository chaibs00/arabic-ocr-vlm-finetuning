from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name):
    """Read a prompt file from the prompts/ folder."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


BASELINE_OCR_PROMPT = load_prompt("baseline_ocr")
TEACHER_PROMPT = load_prompt("teacher_extraction")
TASK1_PROMPT = load_prompt("task1_content")
TASK2_PROMPT = load_prompt("task2_metadata")