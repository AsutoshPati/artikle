import re

import demjson3
from json_repair import repair_json


def extract_json(text: str) -> str | None:
    """
    Extracts the first JSON object or array found in a string.

    Uses a regular expression to search for a JSON-like pattern (object `{}` or
    array `[]`) within the input text.

    Args:
        text (str): The input text potentially containing a JSON structure.

    Returns:
        str or None: The extracted JSON string if a match is found, else None.
    """
    json_pattern = r"(\{.*\}|\[.*\])"  # Match a JSON object or array

    # DOTALL allows matching across newlines
    match = re.search(json_pattern, text, re.DOTALL)

    # Return the match or None
    return match.group(0) if match else None


def repiar_json_str(text: str) -> dict | list | None:
    """
    Attempts to repair and decode a malformed JSON string using multiple
    methods.

    Tries a sequence of repair functions (e.g., custom repair logic or
    third-party decoders like demjson3) to parse the input text into a valid
    JSON object.

    Args:
        text (str): The potentially malformed JSON string.

    Returns:
        dict or list or None: A valid JSON object (dictionary or list) if
            successfully repaired and parsed; otherwise, None.
    """
    # List of functions to try for repairing JSON
    repair_methods = [repair_json, demjson3.decode]

    for method in repair_methods:
        try:
            return method(text)  # Try parsing with each method
        except Exception:
            continue  # Silently ignore and move to the next

    return None
