"""General rules C1–C6 (plan 01 §5).

C1  roles valid and ordered            err
C2  every message has content          err
C3  at least one assistant message     err
C4  rendered tokens <= max length      err
C5  no secrets                         err
C6  no exact / near duplicates         err / warn
"""

from __future__ import annotations

import re

from backend.adapters.base import content_hash, jaccard, shingles
from backend.validators.base import Example, ValidationContext, Violation, v

VALID_ROLES = {"system", "user", "assistant", "tool"}


class C1_Roles:
    code = "C1"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out: list[Violation] = []
        msgs = ex.messages
        if not msgs:
            return [v(self.code, self.level, "example has no messages")]

        for i, m in enumerate(msgs):
            role = m.get("role")
            if role not in VALID_ROLES:
                out.append(v(self.code, self.level, f"invalid role {role!r}", f"messages[{i}]"))

        if msgs and msgs[0].get("role") == "system":
            for i, m in enumerate(msgs[1:], start=1):
                if m.get("role") == "system":
                    out.append(
                        v(self.code, self.level, "system message must be first", f"messages[{i}]")
                    )

        # role transitions
        for i in range(1, len(msgs)):
            prev, cur = msgs[i - 1].get("role"), msgs[i].get("role")
            if cur == "user" and prev == "user":
                out.append(v(self.code, self.level, "two user messages in a row", f"messages[{i}]"))
            if cur == "assistant" and prev == "assistant":
                out.append(
                    v(self.code, self.level, "two assistant messages in a row", f"messages[{i}]")
                )
            if cur == "tool" and prev not in ("assistant", "tool"):
                out.append(
                    v(
                        self.code,
                        self.level,
                        "tool result must directly follow an assistant message or another tool result",
                        f"messages[{i}]",
                    )
                )

        last_role = msgs[-1].get("role")
        if last_role != "assistant":
            out.append(
                v(
                    self.code,
                    self.level,
                    f"conversation must end with an assistant message (ends with {last_role!r})",
                    f"messages[{len(msgs) - 1}]",
                )
            )
        return out


class C2_NonEmpty:
    code = "C2"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out = []
        for i, m in enumerate(ex.messages):
            content = m.get("content")
            if content is None or not str(content).strip():
                out.append(v(self.code, self.level, "empty message content", f"messages[{i}]"))
        return out


class C3_HasAssistant:
    code = "C3"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        if not any(m.get("role") == "assistant" for m in ex.messages):
            return [v(self.code, self.level, "no assistant message in the example")]
        return []


class C4_MaxTokens:
    code = "C4"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        if not ctx.tokenizer_available:
            return []  # skipped; reported separately by the validator runner
        text = ctx.adapter.render_conversation(
            ex.messages, tools=ex.tools, add_generation_prompt=False
        )
        count = ctx.adapter.count_tokens(text)
        if count is None:
            return []
        if count > ctx.max_seq_len:
            return [
                v(
                    self.code,
                    self.level,
                    f"rendered conversation is {count} tokens > max {ctx.max_seq_len}",
                )
            ]
        return []


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("password assignment", re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"]?[^\s'\"]{6,}")),
]


class C5_NoSecrets:
    code = "C5"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out = []
        for i, m in enumerate(ex.messages):
            text = str(m.get("content") or "")
            for label, pattern in _SECRET_PATTERNS:
                match = pattern.search(text)
                if match:
                    snippet = match.group(0)
                    shown = snippet if len(snippet) <= 24 else snippet[:24] + "…"
                    out.append(
                        v(
                            self.code,
                            self.level,
                            f"possible secret ({label}) near {shown!r}",
                            f"messages[{i}]",
                        )
                    )
        return out


class C6_Duplicates:
    code = "C6"
    level = "err"
    categories = None
    scope = "example"

    def check(self, ex: Example, ctx: ValidationContext) -> list[Violation]:
        out: list[Violation] = []
        my_hash = ex.content_hash or content_hash(ex.messages, ex.tools)
        mine_text = _all_text(ex)
        mine_shingles = None

        for peer in ctx.peers:
            if peer.id == ex.id:
                continue
            peer_hash = peer.content_hash or content_hash(peer.messages, peer.tools)
            if my_hash == peer_hash:
                out.append(
                    v(self.code, "err", f"exact duplicate of {peer.id} (same content hash)")
                )
                continue
            if mine_shingles is None:
                mine_shingles = shingles(mine_text)
            score = jaccard(mine_shingles, shingles(_all_text(peer)))
            if score >= ctx.near_dup_threshold:
                out.append(
                    v(
                        self.code,
                        "warn",
                        f"near-duplicate of {peer.id} (similarity {score:.2f})",
                    )
                )
        return out


def _all_text(ex: Example) -> str:
    return "\n".join(str(m.get("content") or "") for m in ex.messages)
