import json
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .common_helper import extract_json

load_dotenv()

# Retrieve the OpenAI API key and model name from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_LLM_MODEL", None)

# Initialize the OpenAI chat model with the retrieved API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.7)


def get_llm_response(gpt_prompt, invoke_data: dict) -> dict | None:
    """
    Generates a response from the LLM based on the provided PromptTemplate and
    invocation data.

    This function constructs a chain using the provided `PromptTemplate` and
    LLM, invokes the model with the given data, and extracts the response. It
    then attempts to parse the response into clean JSON.

    Args:
        gpt_prompt (PromptTemplate): The `PromptTemplate` to send to the
            language model. This template will be used to format the prompt
            dynamically based on the input data.
        invoke_data (dict): The data to invoke the model with. This data will be
            used to fill in the placeholders in the `gpt_prompt` template during
            invocation.

    Returns:
        dict or None:
            - A dictionary containing the parsed and cleaned JSON response from
                the LLM, or
            - None if there was an error during processing or response parsing.
    """
    global llm

    response = None
    clean_json = None

    try:
        # Create the chain using pipe syntax to pass the prompt to the LLM
        chain: Runnable = gpt_prompt | llm

        # Generate the response using the LLMChain
        raw_response = chain.invoke(invoke_data)
        # print("\n\n=====Response:")
        # print(raw_response)

        # Extract the response content if available
        if hasattr(raw_response, "content"):
            response = raw_response.content
        else:
            response = str(raw_response)

        # Extract and clean JSON from the response
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
    Analyzes the provided text content using a pre-defined GPT prompt template.

    This function reads a content analysis template, formats it using the
    provided text content, and invokes the LLM to generate an analysis of the
    article.

    Args:
        text_content (str): The content (e.g., an article) to be analyzed.

    Returns:
        dict or None:
            - A dictionary containing the analysis result from the LLM if
                successful, or
            - None if there was an error during processing, template loading,
                or invocation.

    Raises:
        FileNotFoundError: If the content analysis template file is not found.
        Exception: If any other error occurs during reading the template,
            generating the prompt, or invoking the LLM.
    """
    analysis_template_path = "./llm_calls/gpt_prompts/content_analyser.txt"
    result = None

    try:
        # Read the analysis template from the specified path
        with open(analysis_template_path) as f:
            analysis_template = f.read()

        # Create the PromptTemplate using the loaded template
        gpt_prompt = PromptTemplate(
            input_variables=["article"],
            template=analysis_template,
        )
        invoke_data = {
            "article": text_content,
        }  # Prepare the data to be invoked by the model

        # Get the LLM response for the formatted prompt
        result = get_llm_response(gpt_prompt, invoke_data)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {analysis_template_path}")
    except Exception as err:
        print(f"{type(err)}: {err}")

    return result


def generate_image_prompt(
    topic: str, image_style: str = "cartoon"
) -> dict | None:
    """
    Generates a visual prompt based on a topic and desired image style using a
    language model.

    This function loads an image generation prompt template, fills in the
    placeholders using the provided `topic` and `image_style`, and invokes a
    language model to produce a descriptive image prompt.

    Args:
        topic (str): The subject or theme of the image to be generated.
        image_style (str, optional): The visual style of the image prompt.
            Defaults to "cartoon".

    Returns:
        dict or None: A dictionary containing the generated image prompt or
            `None` if generation fails.

    Raises:
        FileNotFoundError: If the image generation prompt template file is not
            found.
        Exception: For any other runtime issues during prompt processing or
            LLM invocation.
    """
    img_prompt_template_path = "./llm_calls/gpt_prompts/image_gen_prompt.txt"
    result = None

    try:
        # Read the image prompt template from the file
        with open(img_prompt_template_path) as f:
            img_prompt_template = f.read()

        # Create a PromptTemplate by defining input variables
        gpt_prompt = PromptTemplate(
            input_variables=["image_style", "topic"],
            template=img_prompt_template,
        )
        invoke_data = {
            "image_style": image_style,
            "topic": topic,
        }  # Prepare the actual data to be injected into the prompt

        # Invoke the language model to generate the image prompt
        result = get_llm_response(gpt_prompt, invoke_data)
    except FileNotFoundError as err:
        print(f"{type(err)}: {err}")
        print(f"No file available at {img_prompt_template_path}")
    except Exception as err:
        print(f"{type(err)}: {err}")

    return result
