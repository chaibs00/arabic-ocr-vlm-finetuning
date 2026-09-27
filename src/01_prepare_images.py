import logging

from utils import convert_pdf_to_images

# Directories saved in config.py :
from config import PDF_DIR, IMAGE_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    # Log the number of PDFs found
    logger.info("Found %d PDFs in %s", len(pdf_files), PDF_DIR)

    for pdf_path in pdf_files:

        # Log the PDF being processed
        logger.info("Processing %s", pdf_path.name)

        convert_pdf_to_images(
            pdf_path=pdf_path,
            output_base_dir=IMAGE_DIR,
        )

    # Log completion message 
    logger.info("Finished converting all PDFs to images.")


if __name__ == "__main__":
    main()