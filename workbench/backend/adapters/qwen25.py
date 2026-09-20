"""Qwen2.5-7B-Instruct adapter (second supported model).

Template is the tool-call variant vendored by llama.cpp, extracted from
Qwen/Qwen2.5-7B-Instruct's tokenizer_config.json.
"""

from __future__ import annotations

from backend.adapters.base import ModelAdapter


class Qwen25Adapter(ModelAdapter):
    name = "qwen2.5-7b-instruct"
    display_name = "Qwen2.5 7B Instruct"
    hf_repo = "Qwen/Qwen2.5-7B-Instruct"

    bos_token = ""
    eos_token = "<|im_end|>"

    default_template_file = "qwen25_tool_call.jinja"
    tool_use_template_file = "qwen25_tool_call.jinja"
