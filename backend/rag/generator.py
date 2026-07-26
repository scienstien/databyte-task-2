"""
generator.py
Builds the augmented prompt and calls the locally hosted Ollama LLM.
No API keys required — Ollama runs models on your own machine.
"""

import os
import sys
from typing import List, Dict

import httpx

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, SYSTEM_PROMPT
)


def _build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] Source: {chunk['source']} (chunk {chunk['chunk_index']})")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build an augmented prompt from retrieved context and call Ollama.
    Returns the model's response text.
    """
    if not retrieved_chunks:
        return (
            "I was unable to find relevant information in the uploaded documents "
            "to answer your question. Please upload relevant medical documents first."
        )

    context = _build_context(retrieved_chunks)
    system_message = SYSTEM_PROMPT.format(context=context)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": f"{system_message}\n\nQuestion: {question}\n\nAnswer:",
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        }
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120.0          # local inference can be slow on CPU
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except httpx.ConnectError:
        return (
            "Error: Cannot connect to Ollama. "
            "Please ensure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        )
    except httpx.HTTPStatusError as e:
        return f"Error from Ollama: {e.response.status_code} — {e.response.text}"
    except Exception as e:
        return f"Unexpected error during generation: {str(e)}"


def check_ollama_health() -> bool:
    """Returns True if Ollama is reachable."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False
