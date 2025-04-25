import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from .common_helper import extract_json

load_dotenv()

# Retrieve the Gemini API key and model identifier from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", None)

# Check if the required environment variables are available
if GEMINI_API_KEY is None or GEMINI_MODEL is None:
    raise ValueError(
        "Missing required environment variables: "
        "GEMINI_API_KEY or GEMINI_MODEL"
    )

# Initialize the Gemini API client with the API key
genai.configure(api_key=GEMINI_API_KEY)
# Create a Generative Model instance for the specified model
llm = genai.GenerativeModel(GEMINI_MODEL)


def get_llm_response(gpt_prompt: str) -> dict | list | None:
    """
    Generates a response from the LLM and extracts a valid JSON object from it.

    Args:
        gpt_prompt (str): The prompt string to send to the language model.

    Returns:
        dict or list or None: A parsed JSON object (dict or list) if extraction
            and decoding succeed; otherwise, None.
    """
    global llm

    response = None
    clean_json = None

    try:
        # Generate a response from the LLM using the prompt
        response = llm.generate_content(gpt_prompt)
        # print("\n\n=====Response:")
        # print(response)
        response = response.text.strip()

        # Extract JSON string & Attempt to parse it from the response
        clean_json = extract_json(response)
        clean_json = json.loads(clean_json)
    except json.JSONDecodeError:
        print("JSONDecodeError:")
        print("Response:", response)
    except Exception as err:
        print(f"{type(err)} in function get_llm_response")
        print(err)

    return clean_json


def analyse_content(text_content: str) -> dict | None:
    """
    Analyzes the given text content using a GPT prompt template and returns
    structured insights.

    Loads a predefined GPT prompt from a text file, inserts the text content
    into it, and sends the prompt to a language model to extract structured
    analysis as a JSON object.

    Args:
        text_content (str): The text content (e.g., an article or paragraph) to
            be analyzed.

    Returns:
        dict or None: A JSON object (probably a dict) containing the structured
            analysis if successful, otherwise None (e.g., if the prompt file is
            missing or LLM call fails).
    """
    analysis_template_path = "./llm_calls/gpt_prompts/content_analyser.txt"
    result = None

    try:
        # Load the GPT prompt template from file
        with open(analysis_template_path) as f:
            analysis_template = f.read()

        # Format the prompt with the input content
        gpt_prompt = analysis_template.format(article=text_content)

        # Send the formatted prompt to the LLM and retrieve the response
        result = get_llm_response(gpt_prompt)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {analysis_template_path}")

    return result


def generate_image_prompt(
    topic: str, image_style: str = "cartoon"
) -> dict | None:
    """
    Generates an image description prompt using a topic and a visual style via a
    GPT model.

    Loads a prompt template from file, fills in the topic and image style, then
    sends it to a language model to generate a structured image prompt.

    Args:
        topic (str): The main subject or theme for the image.
        image_style (str, optional): The visual style of the image (e.g.,
            "cartoon", "comic", "realistic"). Defaults to "cartoon".

    Returns:
        dict or None: A dictionary containing the generated image prompt if
            successful, otherwise None.
    """
    img_prompt_template_path = "./llm_calls/gpt_prompts/image_gen_prompt.txt"
    result = None

    try:
        # Load the image generation prompt template
        with open(img_prompt_template_path) as f:
            img_prompt_template = f.read()

        # Format the template with the given style and topic
        gpt_prompt = img_prompt_template.format(
            image_style=image_style, topic=topic
        )

        # Send the formatted prompt to the LLM and retrieve the response
        result = get_llm_response(gpt_prompt)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {img_prompt_template_path}")

    return result
