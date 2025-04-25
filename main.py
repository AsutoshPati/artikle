import os
import uuid
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ocr_jobs import is_tesseract_installed
from process_jobs import generate_article_visual

# Constants [App Config]
HOST = os.getenv("HOST", None)
PORT = int(os.getenv("PORT", None))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
STATIC_DIR = "static"
UPLOAD_DIR = "uploads"
image_dir = os.path.join(STATIC_DIR, "generated_images")

# Create the API app
app = FastAPI(
    title="Artikle",
    description="Transform any text-based content—be it a news article, "
    "subject matter, or more—into a visual experience. Simply upload an "
    "image containing text to get started.",
    version="1.0.0",
)


# Allow uploads to be served
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/" + UPLOAD_DIR, StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Allow generated images to be served as static files
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
app.mount("/" + STATIC_DIR, StaticFiles(directory="static"), name="static")

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enum for allowed image styles
class ImageStyle(str, Enum):
    cartoon = "cartoon"
    realistic = "realistic"
    comic = "comic"


@app.post(
    "/upload/",
    summary="Upload an image containing text content along with your "
    "preferred visualization style.",
    tags=["Upload"],
)
async def upload_image(
    style: ImageStyle = Form(
        ..., description="Choose one: cartoon, comic, or realistic"
    ),
    file: UploadFile = File(
        ..., description="Upload an image file (jpg, png, etc.)"
    ),
):
    """
    Uploads an image file with text content and a chosen visualization style,
    returns a visual summary based on the extracted text and also generates
    a visual representation of the text using the specified style.

    Args:
        style (ImageStyle): The visualization style to apply (e.g., cartoon,
            comic, realistic).
        file (UploadFile): The image file containing text content to be
            visualized.

    Raises:
        HTTPException: If the uploaded file is not a supported image type.
        HTTPException: If the backend returns an error status in its JSON
            response.

    Returns:
        JSONResponse: A JSON object containing:
            - `summary` (str): A brief summary of the extracted text.
            - `prompt` (str): The generated image prompt used for visualization.
            - `url` (str): The URL to the generated image.

    """
    global UPLOAD_DIR, image_dir

    # Validate MIME type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and "
            "PNG are allowed.",
        )

    # Generate a unique filename
    file_ext = file.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    # Save the image
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # File path to access image
    file_url = "/" + os.path.join(UPLOAD_DIR, file_name)

    # Process & Generate Image
    json_response = generate_article_visual(file_url, style, image_dir)
    print("\n\nJSON Response:")
    print(json_response)
    print("\n\n")

    # Raise error, if any process failed
    if "status" in json_response:
        raise HTTPException(
            status_code=json_response["status"],
            detail=json_response["payload"]["msg"],
        )

    # Return back the response; with summary, image prompt & generated image
    return JSONResponse(content=json_response)


if __name__ == "__main__":
    # Check if tesseract is available
    if not is_tesseract_installed():
        print("To proceed with the app; Tesseract is required")

    # Auto-run the app
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
