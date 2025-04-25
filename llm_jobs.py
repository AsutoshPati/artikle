import os

from dotenv import load_dotenv

load_dotenv()

# Dynamically import LLM-based analysis and image generation functions
# depending on the environment variable `USE_MODEL`.
USE_MODEL = os.getenv("USE_MODEL", None)

if USE_MODEL == "GEMINI":
    # Use Google's Gemini model
    from llm_calls.gemini_calls import analyse_content, generate_image_prompt
elif USE_MODEL == "OLLAMA":
    # Use a local Ollama LLM model
    from llm_calls.ollama_calls import analyse_content, generate_image_prompt
elif USE_MODEL == "OPENAI":
    # Use OpenAI's LLM model
    from llm_calls.openai_calls import analyse_content, generate_image_prompt
else:
    raise ValueError(f"Unsupported or undefined USE_MODEL: {USE_MODEL}")
