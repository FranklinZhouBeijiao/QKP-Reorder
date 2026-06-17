from __future__ import annotations

import os
import time
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qkp_reorder.prompt_builder import build_system_prompt


ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load project-level environment variables if .env exists."""
    load_dotenv(dotenv_path=ENV_PATH)


def get_deepseek_client() -> OpenAI:
    """Create a DeepSeek client using the OpenAI-compatible SDK."""
    load_project_env()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Please add it to the project .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
    )


def get_deepseek_model(default: str = "deepseek-v4-flash") -> str:
    """Read the configured DeepSeek model name."""
    load_project_env()
    return os.getenv("DEEPSEEK_MODEL", default)


def generate_answer(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Generate an answer using DeepSeek's OpenAI-compatible chat API."""
    client = get_deepseek_client()
    model = model or get_deepseek_model()

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    end_time = time.time()

    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    return {
        "answer": answer,
        "model": model,
        "answer_time": end_time - start_time,
        "usage": usage.model_dump() if usage is not None else None,
    }


def test_deepseek_connection() -> None:
    """Run a minimal DeepSeek connection test."""
    result = generate_answer(
        prompt="Say hello in one short sentence.",
        max_tokens=32,
    )
    print(result["answer"])
    print("model:", result["model"])
    print("answer_time:", round(result["answer_time"], 3))
    print("usage:", result["usage"])


if __name__ == "__main__":
    test_deepseek_connection()

