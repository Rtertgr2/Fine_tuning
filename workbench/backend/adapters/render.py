"""Local jinja rendering of vendored chat templates.

Uses transformers' own template compiler (pure jinja2, no model download), so
the `tojson` filter and sandbox behaviour are byte-identical to training-time
`tokenizer.apply_chat_template`.
"""

from __future__ import annotations

from typing import Any, Sequence


def render_jinja(
    template_text: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    bos_token: str = "",
    eos_token: str = "",
    add_generation_prompt: bool = False,
    **extra: Any,
) -> str:
    from transformers.utils import chat_template_utils as ctu

    # Public API (transformers >= 5.x): same compiler + tojson filter as
    # training-time apply_chat_template, no private access. Returns
    # (rendered_texts, generation_indices); one conversation -> one string.
    rendered, _generation_indices = ctu.render_jinja_template(
        conversations=[list(messages)],
        tools=list(tools) if tools else None,
        chat_template=template_text,
        bos_token=bos_token,
        eos_token=eos_token,
        add_generation_prompt=add_generation_prompt,
        **extra,
    )
    return rendered[0]
