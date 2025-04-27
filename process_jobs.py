import cv2
import numpy
import os

from img_gen_jobs import generate_ai_img
from llm_jobs import analyse_content, generate_image_prompt
from ocr_jobs import extract_text_content


def read_image(img_path: str) -> numpy.ndarray | None:
    """
    Reads an image from the given file path

    Args:
        img_path (str): The file path to the image.

    Returns:
        numpy.ndarray or None: The image as a NumPy array if successful;
            otherwise, None.

    """
    img = None

    try:
        img_path = "." + img_path
        print("\n\nImage Path:")
        print(img_path)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            print("Image doesn't exist")
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
    finally:
        return img


def pre_process_img(img: numpy.ndarray) -> numpy.ndarray | None:
    """
    Applies preprocessing steps to an input image to enhance it for text or edge
    detection.
    The image is converted to grayscale, denoised using a bilateral filter, and
    then binarized using adaptive thresholding.

    Args:
        img (numpy.ndarray): The input image in BGR format (as read by OpenCV).

    Returns:
        numpy.ndarray or None: The preprocessed binary image, or None if an
            error occurs.

    """
    processed_img = None

    try:
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        processed_img = cv2.bilateralFilter(processed_img, 11, 17, 17)

        # Apply adaptive thresholding to highlight important regions
        processed_img = cv2.adaptiveThreshold(
            processed_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            10,
        )
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
    finally:
        return processed_img


def generate_article_visual(
    img_path: str, image_style: str, image_storage: str
) -> dict:
    """
    Generates a visual representation of text content extracted from an image.
    This function pre-processes  fthe uploaded image for better text extraction,
    extracts the textual content, summarizes it, generates an image prompt based
    on the summary, and finally creates a styled AI-generated image to return it
    back.

    Args:
        img_path (str): Path to the uploaded image file.
        image_style (str): Visualization style to apply (e.g., cartoon, comic,
            realistic).
        image_storage (str): Destination path or identifier to store the
            generated image.

    Returns:
        dict: A JSON-like dictionary containing:
            - "summary" (str): A concise summary of the extracted content.
            - "image prompt" (str): The prompt generated for AI-based image
                creation.
            - "generated_image" (str): Path or URL to the generated image.
        If any step fails, returns:
            - "status" (int): HTTP error code (e.g., 500).
            - "payload" (dict): Contains "msg" with error details.
    """
    content_path = None

    # Read the uploaded image
    img = read_image(img_path)
    if img is None:
        return {
            "status": 500,
            "payload": {"msg": "Unable to access uploaded image"},
        }

    # Preprocess the image for better text extraction
    processed_img = pre_process_img(img)
    if processed_img is None:
        return {
            "status": 500,
            "payload": {"msg": "Unable to pre-process image"},
        }

    # Extract text from the preprocessed image
    text = extract_text_content(processed_img)
    print("\n\nExtracted Text:")
    print(text)
    if not text:
        return {
            "status": 500,
            "payload": {"msg": "Unable to extract any text from image"},
        }

    # Analyze and summarize the extracted text
    try:
        summary = analyse_content(text)
        summary = summary["summary"]
        print("\n\nSummary:")
        print(summary)
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
        return {"status": 500, "payload": {"msg": "Unable to analyse content"}}

    # Generate a descriptive prompt for the image generator
    try:
        img_prompt = generate_image_prompt(summary, image_style=image_style)
        img_prompt = img_prompt["image_prompt"]
        print("\n\nImage Prompt:")
        print(img_prompt)
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
        return {
            "status": 500,
            "payload": {"msg": "Unable to generate content description"},
        }

    # Generate the final AI image using the prompt and store it
    try:
        content_path = None
        content_path = generate_ai_img(img_prompt, image_storage)
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")
        return {"status": 500, "payload": {"msg": "Unable to generate content"}}

    # If unable to complete any process
    if content_path is None:
        return {"status": 500, "payload": {"msg": "Unable to generate content"}}

    # Final result
    return {
        "summary": summary,
        "image_prompt": img_prompt,
        "generated_image": content_path,
    }
