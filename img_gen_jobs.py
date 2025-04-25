import os

from dotenv import load_dotenv

load_dotenv()

# Determine which image generation service to use
IMG_GEN_SERVICE = os.getenv("IMG_GEN_SERVICE", None)

if IMG_GEN_SERVICE == "DIFFUSION":
    # Use local or hosted diffusion-based image generation
    from img_gen_calls.diffusion_calls import generate_and_save_image
elif IMG_GEN_SERVICE == "OPENAI":
    # Use OpenAI's image generation
    print(f"\n\nUsing Image Gen Service:  {IMG_GEN_SERVICE}\n\n")
    from img_gen_calls.openai_img_calls import generate_and_save_image


def generate_ai_img(prompt: str, img_dir: str) -> str:
    """
    Generate and save an image based on a text prompt using the configured image
    generation service.

    Args:
        prompt (str): The text prompt describing the image to be generated.
        img_dir (str): The directory where the generated image should be saved.

    Returns:
        str: The path to the generated image file.
    """
    generated_img = None
    generated_img = generate_and_save_image(prompt, img_dir)
    return generated_img
