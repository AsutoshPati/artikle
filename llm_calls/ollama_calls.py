import json
import os
import subprocess

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

from .common_helper import extract_json, repiar_json_str

load_dotenv()

# Retrieve the Ollama model identifier from environment variables
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", None)

# Check if the required environment variable is available
if OLLAMA_MODEL is None:
    raise ValueError("Missing required environment variable: OLLAMA_MODEL")


def is_ollama_installed() -> bool:
    """
    Check if Ollama is installed by running `ollama --version`.

    Returns:
        bool: True if Ollama is installed and callable, False if not installed
            or not in PATH.
    """
    try:
        # Try running the Ollama version command
        subprocess.run(
            ["ollama", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Ensure Ollama is installed and running
if not is_ollama_installed():
    raise EnvironmentError(
        "Ollama is not installed or not accessible in " "the system PATH."
    )

# Initialize the Ollama model with the specified model identifier
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.7)


def get_llm_response(gpt_prompt: str) -> dict | list | None:
    """
    Sends a prompt to the LLM and attempts to extract a clean JSON response.

    Args:
        gpt_prompt (str): The prompt string to send to the language model.

    Returns:
        dict or list or None: A parsed JSON object (dict or list) if extraction
            and decoding succeed; otherwise, None (e.g., if LLM invocation or
            JSON extraction fails).
    """
    global llm

    response = None
    clean_json = None

    try:
        # Generate the response using the LLM
        response = llm.invoke(gpt_prompt)
        # print("\n\n=====Response:")
        # print(response)
    except Exception as err:
        print(f"{type(err)} in function get_llm_response")
        print(err)
        return response

    try:
        # Attempt to repair malformed JSON
        # (lower-tier models might produce unstructured JSON)
        response = repiar_json_str(response)

        # Extract and clean the JSON from the response
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
    Analyzes the content of a given article using a GPT-based language model.

    This function loads a content analysis template, formats it with the
    provided article text, and sends the prompt to the language model for
    analysis.

    Args:
        text_content (str): The article or text content to be analyzed.

    Returns:
        dict or None: A dictionary containing the result of the content
            analysis, or None if an error occurs.
    """
    analysis_template_path = "./llm_calls/gpt_prompts/content_analyser.txt"
    result = None

    try:
        # Load the content analysis template
        with open(analysis_template_path) as f:
            analysis_template = f.read()

        # Format the prompt with the provided text content
        gpt_prompt = analysis_template.format(article=text_content)

        # Get the response from the language model
        result = get_llm_response(gpt_prompt)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {analysis_template_path}")

    return result


def generate_image_prompt(
    topic: str, image_style: str = "cartoon"
) -> dict | None:
    """
    Generates an image prompt for a given topic and image style using a
    GPT-based language model.

    This function loads an image generation prompt template, formats it with
    the provided topic and image style, and then sends the prompt to the
    language model to generate a detailed image prompt.

    Args:
        topic (str): The topic or subject of the image prompt.
        image_style (str, optional): The desired style for the image (e.g.,
            "cartoon", "realistic"). Defaults to "cartoon".

    Returns:
        dict or None: A dictionary containing the generated image prompt, or
            None if an error occurs.
    """
    img_prompt_template_path = "./llm_calls/gpt_prompts/image_gen_prompt.txt"
    result = None

    try:
        # Load the image generation prompt template from the file
        with open(img_prompt_template_path) as f:
            img_prompt_template = f.read()

        # Format the template with the provided topic and image style
        gpt_prompt = img_prompt_template.format(
            image_style=image_style, topic=topic
        )

        # Get the response (image prompt) from the language model
        result = get_llm_response(gpt_prompt)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {img_prompt_template_path}")

    return result
