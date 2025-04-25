import os
from io import BytesIO
from uuid import uuid4

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
# Load OpenAI API key and model from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_IMG_MODEL", None)

# Ensure the OpenAI API key and model are set
if OPENAI_API_KEY is None or OPENAI_MODEL is None:
    raise ValueError(
        "OPENAI_API_KEY or OPENAI_IMG_MODEL " "environment variable is not set."
    )

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_image(
    prompt, size="1024x1024", quality="standard", n=1
) -> list | None:
    """
    Generate an image using OpenAI's image generation model.

    Args:
        prompt (str): Description of the image to generate.
        size (str): Size of the image. Options include:
            - "256x256"
            - "512x512"
            - "1024x1024" (default)
        quality (str): Quality of the image. Options are:
            - "standard" (default)
            - "hd"
        n (int): Number of images to generate (default is 1).

    Returns:
        list | None: A list of URLs pointing to the generated images.
    """
    try:
        # Call the OpenAI API to generate an image
        response = client.images.generate(
            model=OPENAI_MODEL, prompt=prompt, size=size, quality=quality, n=n
        )

        # Extract and return the image URLs
        image_urls = [item.url for item in response.data]
        return image_urls

    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def generate_and_save_image(prompt: str, storage_dir: str) -> str | None:
    """
    Generate an image from a given prompt and save it to the specified
    directory.

    Args:
        prompt (str): The description of the image to generate.
        storage_dir (str): The directory where the generated image will be
            saved.

    Returns:
        str or None: The path where the image was saved, or None if the image
            couldn't be saved.
    """
    img_path = None

    # Generate the image
    image_urls = generate_image(prompt)

    if image_urls:
        try:
            # Save the first generated image
            img_path = os.path.join(storage_dir, f"{uuid4()}.png")
            response = requests.get(image_urls[0])
            img = Image.open(BytesIO(response.content))
            img.save(img_path)
            print(f"Image saved as {img_path}")
        except Exception as err:
            print(f"\n\nERROR: {type(err)}: {err}")

    return img_path
