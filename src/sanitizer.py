import re

INSTRUCTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous",
    r"do not follow",
    r"system:",
    r"assistant:",
    r"user:",
    r"execute",
    r"run",
    r"openai_api_key",
    r"api_key",
]


def sanitize_text(text: str, max_chars: int = 3000) -> str:
    """Sanitize free text used as LLM context or user query.

    - Removes common instruction-like lines (prompt injection patterns).
    - Strips code fences and long inline code blocks.
    - Collapses repeated whitespace and truncates to max_chars.
    """
    if not text:
        return ""

    # Remove Markdown code fences and their contents
    text = re.sub(r"```[\s\S]*?```", " ", text, flags=re.IGNORECASE)

    # Split into lines and filter out instruction-like lines
    out_lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        lower = l.lower()
        skip = False
        for pat in INSTRUCTION_PATTERNS:
            if pat in lower:
                skip = True
                break
        if skip:
            continue
        # Remove lines that look like prefixed message roles (e.g., "System:")
        if re.match(r"^(system|assistant|user)\b[:\-]", lower):
            continue
        out_lines.append(l)

    cleaned = "\n".join(out_lines)

    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Truncate to max_chars, try to keep whole words
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
        # trim to last space
        last_space = cleaned.rfind(" ")
        if last_space > 0:
            cleaned = cleaned[:last_space]

    return cleaned


def sanitize_query(query: str, max_chars: int = 500) -> str:
    """Lightweight sanitization for user query before interpolation."""
    return sanitize_text(query, max_chars=max_chars)
