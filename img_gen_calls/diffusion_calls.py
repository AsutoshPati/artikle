import os
from uuid import uuid4

import torch
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import CLIPTokenizer

load_dotenv()

# Get the diffusion model option from environment variables
# Default to 0 if not set or if conversion to int fails
DIFFUSION_MODEL_OPT = int(os.getenv("DIFFUSION_MODEL_OPT", 0))

# Login to Hugging Face Hub using the HF_TOKEN environment variable.
# This is required for accessing certain models that need authentication to
# download.
# Make sure you've also accepted the model license/terms on the
# Hugging Face Hub, if required.
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print(
        "HF_TOKEN not found in environment. "
        "Some models may require authentication and "
        "prior consent to download."
    )


def select_pipeline() -> tuple:
    """
    Selects and initializes a diffusion model pipeline based on the global
    DIFFUSION_MODEL_OPT value.

    Returns:
        tuple: (pipeline object, number of inference steps, guidance scale)

    Raises:
        ValueError: If an unsupported DIFFUSION_MODEL_OPT value is provided.

    Notes:
        - Uses GPU (CUDA) for acceleration.
        - Requires models to be downloaded via Hugging Face (authentication
            may be needed).
    """
    global DIFFUSION_MODEL_OPT

    # Inference configuration:
    # - num_inference_steps: number of denoising steps
    # (higher = better quality, slower)
    # - guidance_scale: how strongly the image adheres to the prompt
    # (higher = stricter)

    if DIFFUSION_MODEL_OPT == 1:
        num_inference_steps = 50
        guidance_scale = 7.5

        # Standard Stable Diffusion v1.5
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    elif DIFFUSION_MODEL_OPT == 2:
        num_inference_steps = 30
        guidance_scale = 7.5

        # Stable Diffusion XL base model (more advanced than v1.5)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        tokenizer_one = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
        )
    # elif DIFFUSION_MODEL_OPT == 3:
    #     num_inference_steps=30
    #     guidance_scale=3.5

    #     # Experimental FLUX.1-dev model
    #     model_id = "black-forest-labs/FLUX.1-dev"
    #     pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    elif DIFFUSION_MODEL_OPT == 4:
        num_inference_steps = 50
        guidance_scale = 7.0

        # Anime-style diffusion model [light-weight]
        model_id = "hakurei/waifu-diffusion"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    # elif DIFFUSION_MODEL_OPT == 5:
    #     num_inference_steps = 30
    #     guidance_scale = 7.5

    #     # DeepFloyd IF-I XL model
    #     model_id = "DeepFloyd/IF-I-XL-v1.0"
    #     pipe = DiffusionPipeline.from_pretrained(
    #         model_id, torch_dtype=torch.float16, variant="fp16"
    #     )
    else:
        raise ValueError("Invalid diffusion model selected")

    print(f"\n\nUsing Diffusion Model: {model_id}\n\n")

    # Move pipeline to GPU for faster inference
    pipe = pipe.to("cuda")

    return pipe, num_inference_steps, guidance_scale


def generate_and_save_image(prompt: str, storage_dir: str) -> str | None:
    """
    Generates an image based on the provided prompt using a selected diffusion
    model, and saves the generated image to a specified directory.

    Args:
        prompt (str): The textual prompt to generate the image based on.
        storage_dir (str): The directory where the generated image will be
            saved.

    Returns:
        str or None: The file path of the saved image, or None if there was an
            error.

    Notes:
        - The function selects the appropriate diffusion pipeline and
            configuration based on the global `DIFFUSION_MODEL_OPT`.
        - The generated image is saved as a PNG file with a unique filename.
        - The image is processed using the specified number of inference steps
            and guidance scale for image generation.
    """
    img_path = None

    try:
        # Select the diffusion pipeline and
        # configuration based on the model option
        pipeline, num_inference_steps, guidance_scale = select_pipeline()

        # Generate the image using the selected pipeline and provided prompt
        image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[
            0
        ]  # Take the first generated image

        # Create a unique file name for the image and save it
        img_path = os.path.join(storage_dir, f"{uuid4()}.png")
        image.save(img_path)  # Save the image in PNG format
        print(f"Image saved as {img_path}")
    except Exception as err:
        print(f"\n\nERROR: {type(err)}: {err}")

    return img_path
