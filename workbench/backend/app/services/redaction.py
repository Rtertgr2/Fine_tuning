"""Secret redaction service — redact at write time, not read time.

Detects API keys, passwords, tokens, private keys, and other secrets in text,
replaces them with placeholders, and logs the mapping separately for audit.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Ordered patterns — more specific patterns first to avoid false negatives
SECRET_PATTERNS: list[tuple[str, str]] = [
    # AWS access keys
    (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY]"),
    # AWS secret keys (typically 40 chars base64-ish)
    (r"(?i)aws(.{0,20})?(secret|key)['\"\s:=]+[A-Za-z0-9/+=]{40}", "[REDACTED_AWS_SECRET]"),
    # Generic API keys
    (r"(?i)(api[_-]?key|apikey)['\"\s:=]+[A-Za-z0-9_\-]{20,64}", "[REDACTED_API_KEY]"),
    # Bearer tokens
    (r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}", "[REDACTED_BEARER_TOKEN]"),
    # GitHub tokens (ghp_, gho_, github_pat_)
    (r"(?:ghp_|gho_|github_pat_)[A-Za-z0-9]{36,}", "[REDACTED_GITHUB_TOKEN]"),
    # OpenAI / Anthropic style keys (variable length)
    (r"sk-[A-Za-z0-9]{32,}", "[REDACTED_SK_KEY]"),
    (r"sk-ant-[A-Za-z0-9]{32,}", "[REDACTED_ANTHROPIC_KEY]"),
    # Private keys (RSA, EC, etc.)
    (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    # Passwords in configs
    (r"(?i)(password|passwd|pwd)['\"\s:=]+[^\s'\"]{8,}", "[REDACTED_PASSWORD]"),
    # Database connection strings with passwords
    (r"(?i)(?:postgres|mysql|mongodb(?:[+a-z]+)?)://[^:\s]+:[^@\s]+@", "[REDACTED_DB_URI]"),
    # Slack tokens
    (r"xox[abprs]-[A-Za-z0-9\-]{10,}", "[REDACTED_SLACK_TOKEN]"),
    # JWT tokens
    (r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "[REDACTED_JWT]"),
    # Generic secret/token assignments
    (r"(?i)(secret|token|auth)['\"\s:=]+[A-Za-z0-9_\-]{20,}", "[REDACTED_SECRET]"),
    # IP addresses (PII-adjacent)
    (r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b", "[REDACTED_IP]"),
    # Email addresses
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]"),
]

# Compiled patterns for performance
_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in SECRET_PATTERNS
]


@dataclass
class RedactionResult:
    """Result of a redaction pass."""
    text: str
    redaction_count: int
    redaction_map: dict[str, str] = field(default_factory=dict)

    @property
    def had_secrets(self) -> bool:
        return self.redaction_count > 0


def _placeholder_hash(secret: str) -> str:
    """Create a short audit hash for a redacted secret."""
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def redact(text: str) -> RedactionResult:
    """Redact secrets from text, returning cleaned text and audit map."""
    result = text
    count = 0
    audit_map: dict[str, str] = {}

    for pattern, replacement in _COMPILED:
        matches = pattern.findall(result)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            if not match:
                continue
            placeholder = replacement
            if placeholder not in audit_map:
                audit_map[placeholder] = _placeholder_hash(match)
            count += 1
        result = pattern.sub(replacement, result)

    return RedactionResult(
        text=result,
        redaction_count=count,
        redaction_map=audit_map,
    )


def redaction_map_from_text(text: str) -> dict[str, str]:
    """Redact text and return just the redaction map (for audit)."""
    result = redact(text)
    return result.redaction_map


def redact_dict(data: dict) -> tuple[dict, RedactionResult | None]:
    """Redact all string values in a dict recursively."""
    total_result: RedactionResult | None = None

    def _redact_value(value):
        nonlocal total_result
        if isinstance(value, str):
            r = redact(value)
            if r.had_secrets:
                if total_result is None:
                    total_result = RedactionResult(text="", redaction_count=0)
                total_result.redaction_count += r.redaction_count
                total_result.redaction_map.update(r.redaction_map)
            return r.text
        if isinstance(value, dict):
            return {k: _redact_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact_value(v) for v in value]
        return value

    cleaned = _redact_value(data)
    if isinstance(cleaned, dict):
        return cleaned, total_result
    return data, None
