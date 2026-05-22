"""Auxen Space — UI demo for the Auxen dedicated-LLM platform.

This Space is intentionally a marketing/UI demo, not a live-inference demo.
Auxen instances are per-customer dedicated GPUs (per-minute billing, not
per-token), so there's no shared backend to point this Space at. Visitors
see the chat UI and a friendly canned response explaining how to provision
their own instance at auxen.ai.

If you want a live-chat demo: provision an Auxen instance at auxen.ai,
fork this Space, and set AUXEN_API_BASE + AUXEN_API_KEY as Space secrets
— then the chat actually streams from your dedicated model. The
implementation is in the `_live_respond` function below.
"""

from __future__ import annotations

import os
from typing import Iterator

import gradio as gr

AUXEN_API_BASE = os.environ.get("AUXEN_API_BASE")
AUXEN_API_KEY = os.environ.get("AUXEN_API_KEY")
AUXEN_MODEL = os.environ.get("AUXEN_MODEL", "llama-3.1-8b")

LIVE_MODE = bool(AUXEN_API_BASE and AUXEN_API_KEY)


CANNED_REPLY = """\
👋 This Space is a UI preview, not a live model.

**To actually chat with a private LLM**, provision an Auxen instance:

1. Sign up at **[auxen.ai](https://auxen.ai)** (~30 sec).
2. Pick a model (Llama 3.1/3.2, Qwen 2.5, Mistral, Gemma 2, Mixtral, Phi-3, Command R).
3. Your instance comes online in ~3 minutes with a stable HTTPS endpoint, OpenAI-compatible API, and pay-per-minute pricing — no per-token bills.

You can fork this Space and set `AUXEN_API_BASE` + `AUXEN_API_KEY` as secrets to wire your instance into this same UI.

**Why not run a live demo here?** Auxen is per-customer dedicated GPU (the value prop is "your own instance, not a shared fleet"). A shared demo instance would contradict that. So this Space exists as a marketing surface; the chat happens on your own provisioned instance.

---

Try the [Next.js starter](https://github.com/auxen-ai/auxen-nextjs-starter) or the [LangChain starter](https://github.com/auxen-ai/auxen-langchain-starter) for a working dev-environment template.
"""


def _live_respond(message: str, history: list[dict]) -> Iterator[str]:
    """Streaming chat against an Auxen instance — only runs when LIVE_MODE."""
    try:
        from openai import OpenAI
    except ImportError:
        yield "Live mode requires the `openai` package. See requirements.txt."
        return

    client = OpenAI(base_url=AUXEN_API_BASE, api_key=AUXEN_API_KEY)
    messages = [*history, {"role": "user", "content": message}]
    stream = client.chat.completions.create(
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


def respond(message: str, history: list[dict]) -> Iterator[str]:
    if LIVE_MODE:
        yield from _live_respond(message, history)
    else:
        yield CANNED_REPLY


with gr.Blocks(
    title="Auxen — Private Dedicated LLM Endpoints",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        """
        ## Auxen — Private Dedicated LLM Endpoints

        Your own GPU. Your own model. **Per-minute pricing, not per-token.**

        Auxen provisions a dedicated GPU instance running an open-source model behind a stable HTTPS endpoint with an OpenAI-compatible API. Billing is by the minute the GPU is running — no token meter, no monthly minimums.

        - **Catalog**: Llama 3.1/3.2, Qwen 2.5, Mistral, Gemma 2, Mixtral, Phi-3, Command R
        - **Pricing**: $0.10/hr (3-7B) up to $1.50/hr (70B+) — see [auxen.ai/pricing](https://auxen.ai/pricing)
        - **MCP server** with OAuth 2.1 + PKCE at `api.auxen.ai/mcp` for agent workloads

        **[Provision an instance →](https://auxen.ai)**

        ---
        """
    )
    gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            "What is Auxen?",
            "How does per-minute pricing compare to per-token?",
            "Which models can I deploy?",
            "Show me the API",
        ],
        cache_examples=False,
        title="Chat preview",
        description=(
            "Type a message to see what the chat UI looks like. "
            "(Real chat happens on your own provisioned Auxen instance — see the response.)"
            if not LIVE_MODE
            else f"Live chat against your Auxen instance (model: {AUXEN_MODEL})."
        ),
    )
    gr.Markdown(
        """
        ---

        ### Source

        - **Auxen platform**: [auxen.ai](https://auxen.ai) · [docs](https://auxen.ai/docs)
        - **This Space**: [github.com/auxen-ai/auxen-hf-space](https://github.com/auxen-ai/auxen-hf-space)
        - **npm provider**: [`@auxen/ai-sdk-provider`](https://www.npmjs.com/package/@auxen/ai-sdk-provider)
        - **Starter templates**: [Next.js](https://github.com/auxen-ai/auxen-nextjs-starter) · [LangChain](https://github.com/auxen-ai/auxen-langchain-starter) · [LangGraph](https://github.com/auxen-ai/auxen-langgraph-starter)
        """
    )

if __name__ == "__main__":
    demo.launch()
