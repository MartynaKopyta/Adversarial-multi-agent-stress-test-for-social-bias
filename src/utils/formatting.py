"""
src/utils/formatting.py

Shared text formatting utilities.
Centralises the conversation history formatting logic
that was duplicated between persuader.py and judge.py.
"""

from __future__ import annotations


def format_history(history: list[dict[str, str]]) -> str:
    """
    Format a conversation history list as a readable string.
    Roles labelled [Persuader] and [Model].

    Args:
        history: List of {"role": "user"|"assistant", "content": str} dicts.

    Returns:
        Multi-line string with labelled turns, or a placeholder if empty.
    """
    lines = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"[Persuader]: {content}")
        elif role == "assistant":
            lines.append(f"[Model]: {content}")
    return "\n\n".join(lines) if lines else "(conversation not yet started)"