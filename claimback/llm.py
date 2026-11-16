
"""Gemini API wrapper so the rest of Claimback stays unchanged."""

import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_client = None


def available():
    """Check whether a Gemini API key is configured."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def _get_client():
    global _client

    if _client is None:
        from google import genai
        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

    return _client


def ask(prompt, system=None, json_schema=None, max_tokens=4000):
    """
    Send a prompt to Gemini.

    Returns text, or a Python dictionary when json_schema is supplied.
    Returns None if the API call fails.
    """
    if not available():
        return None

    try:
        config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.2,
        }

        if system:
            config["system_instruction"] = system

        if json_schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = json_schema

        response = _get_client().models.generate_content(
            model=os.environ.get(
                "GEMINI_MODEL", "gemini-2.5-flash"
            ),
            contents=prompt,
            config=config,
        )

        result = response.text

        if not result:
            return None

        if json_schema:
            return json.loads(result)

        return result.strip()

    except Exception as e:
        print("Gemini API call failed:", e)
        return None
