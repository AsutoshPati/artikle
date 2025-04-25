import subprocess

import numpy
import pytesseract

custom_config = r"--oem 3 --psm 1"
# OCR Engine Mode (OEM) 3: Use the default OCR engine
# Page Segmentation Mode (PSM) 1:
#   Assumes a complex document layout (multiple columns)


def extract_text_content(processed_img: numpy.ndarray) -> str | None:
    """
    Extracts text content from a pre-processed image using Tesseract OCR; with a
    custom OCR configuration optimized for complex, multi-column documents.

    Args:
        processed_img (numpy.ndarray): A pre-processed (grayscale and
            thresholded) image.

    Returns:
        str or None: Extracted text as a string if successful, otherwise None.
    """
    text = None

    try:
        # Run Tesseract OCR with custom config (OEM 3, PSM 1)
        text = pytesseract.image_to_string(processed_img, config=custom_config)
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
    finally:
        return text


def is_tesseract_installed() -> bool:
    """
    Checks whether Tesseract OCR is installed and accessible from the command
    line.

    Runs the `tesseract --version` command to determine if Tesseract is
    installed.

    Returns:
        bool: True if Tesseract is installed and callable, False if not
            installed or not in PATH.
    """
    try:
        # Try running the tesseract version command
        subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
