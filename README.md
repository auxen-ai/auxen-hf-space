---
title: Auxen Chat Demo
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Chat demo for Auxen — private dedicated LLM endpoints
---

# Auxen Chat Demo

A live Gradio chat demo backed by an [Auxen](https://auxen.ai) dedicated LLM endpoint. Auxen instances run open-source models (Llama, Qwen, Mistral, Gemma, Mixtral, Phi, Command R) on a per-customer dedicated GPU with an OpenAI-compatible API.

## What you're seeing

This Space connects to a single Auxen instance configured by the Space owner. Each Auxen instance is dedicated to one customer (no shared inference fleet), and pricing is per-minute of GPU runtime, not per token.

## Run your own

Provision your own Auxen instance at [auxen.ai](https://auxen.ai), then duplicate this Space and set two secrets:

| Secret           | Value                                                                |
| ---------------- | -------------------------------------------------------------------- |
| `AUXEN_API_BASE` | Your per-instance base URL, e.g. `https://api.auxen.ai/v1/inst_xxx/v1` |
| `AUXEN_API_KEY`  | Your `auxk_*` API key                                                |
| `AUXEN_MODEL`    | Optional. Defaults to `llama-3.1-8b`.                                |

## How it works

```python
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["AUXEN_API_BASE"],
    api_key=os.environ["AUXEN_API_KEY"],
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
```

Auxen exposes the OpenAI Chat Completions wire format on every instance, so the integration is just an OpenAI client pointed at your instance URL.

## Source

Source for this Space: [github.com/auxen-ai/auxen-hf-space](https://github.com/auxen-ai/auxen-hf-space).

For a Next.js version: [auxen-ai/auxen-nextjs-starter](https://github.com/auxen-ai/auxen-nextjs-starter).
