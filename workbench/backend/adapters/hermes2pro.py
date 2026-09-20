"""Hermes-2-Pro-Llama-3-8B adapter (first supported model).

Templates are byte-exact copies extracted from the model repo's
tokenizer_config.json (chat_template list: "default" and "tool_use");
the tool_use copy was cross-checked against llama.cpp's vendored extraction.
"""

from __future__ import annotations

from backend.adapters.base import ModelAdapter


class Hermes2ProLlama3Adapter(ModelAdapter):
    name = "hermes2pro-llama3-8b"
    display_name = "Hermes 2 Pro Llama-3 8B"
    hf_repo = "NousResearch/Hermes-2-Pro-Llama-3-8B"

    bos_token = "<|begin_of_text|>"
    eos_token = "<|im_end|>"

    default_template_file = "hermes2pro_default.jinja"
    tool_use_template_file = "hermes2pro_tool_use.jinja"
