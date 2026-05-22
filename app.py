"""Gradio chat demo backed by an Auxen dedicated LLM endpoint."""

from __future__ import annotations

import os

import gradio as gr
from openai import OpenAI

AUXEN_API_BASE = os.environ.get("AUXEN_API_BASE")
AUXEN_API_KEY = os.environ.get("AUXEN_API_KEY")
AUXEN_MODEL = os.environ.get("AUXEN_MODEL", "llama-3.1-8b")


def _build_client() -> OpenAI | None:
    if not AUXEN_API_BASE or not AUXEN_API_KEY:
        return None
    return OpenAI(base_url=AUXEN_API_BASE, api_key=AUXEN_API_KEY)


_client = _build_client()


def respond(message: str, history: list[dict]):
    if _client is None:
        yield (
            "⚠️ Auxen credentials are not configured. The Space owner needs to set "
            "`AUXEN_API_BASE` and `AUXEN_API_KEY` as Space secrets (get them from "
            "https://auxen.ai)."
        )
        return

    messages = [*history, {"role": "user", "content": message}]
    stream = _client.chat.completions.create(
        model=AUXEN_MODEL,
        messages=messages,
        stream=True,
    )

    text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            yield text


with gr.Blocks(
    title="Auxen Chat Demo",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        "## Auxen Chat Demo\n"
        "Backed by a [dedicated LLM endpoint on Auxen](https://auxen.ai). "
        "Per-minute GPU billing, no per-token charges. "
        f"Model: `{AUXEN_MODEL}`."
    )
    gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            "What's your name?",
            "Explain dedicated GPU inference in two sentences.",
            "Write a Python one-liner that prints fizzbuzz for 1 to 15.",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
