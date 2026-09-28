
import os
from os.path import join
from glob import glob
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import base64
import json_repair




def preprocess_image(image, max_width=600):
    """
    Preprocess image to reduce size and prepare for OCR.

    Args:
        image: PIL Image object
        max_width: Maximum width in pixels (height auto-calculated to maintain ratio)

    Steps:
    1. Convert to grayscale (reduces size and improves OCR)
    2. Resize to reasonable width while maintaining aspect ratio
    3. Increase contrast
    """

    # Convert to grayscale
    gray_image = image.convert('L')

    # Resize if image is too large
    if gray_image.width > max_width:
        # Calculate new height to maintain aspect ratio
        ratio = max_width / gray_image.width
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((max_width, new_height), Image.LANCZOS)

    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(1.5)

    return enhanced_image




def convert_pdf_to_images(pdf_path, output_base_dir, max_width=600):
    """
    Convert a PDF file to a set of preprocessed images.

    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory to save images
        max_width: Maximum width for output images (default: 600px)
    """

    # Get PDF filename without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create output directory for this PDF
    output_dir = join(output_base_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {pdf_name}.pdf...")

    images = convert_from_path(pdf_path, dpi=200)
    generated_paths = []

    # Process and save each page
    for page_num, image in enumerate(images, start=1):

        # Preprocess the image
        processed_image = preprocess_image(image, max_width)

        # Save as JPEG with optimization
        output_path = join(output_dir, f"page_{page_num:03d}.jpg")
        processed_image.save(output_path, 'JPEG', quality=85, optimize=True)

        print(f"  Saved page {page_num} -> {output_path}")
        generated_paths.append(output_path)

    return generated_paths




def encode_image(image_path):
    """Read an image and return it as a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")




def parse_json(text):
    """Parse JSON text, repairing it if needed. Return None if it fails."""
    try:
        return json_repair.loads(text)
    except Exception:
        return None