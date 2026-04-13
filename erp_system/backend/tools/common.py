from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    loaded = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return loaded if isinstance(loaded, dict) else None
    return None


def extract_json_payload(message: str) -> dict[str, Any]:
    start = message.find("{")
    if start == -1:
        return {}
    payload = message[start:]
    return json.loads(payload)


def looks_like_json_payload(message: str) -> bool:
    return "{" in message and message.rstrip().endswith("}")


def format_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows returned."
    headers = list(rows[0].keys())
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(" | ".join(str(row.get(header, "")) for header in headers))
    return "\n".join(lines)


def detect_numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    numeric_columns = []
    for key in rows[0].keys():
        values = [row.get(key) for row in rows]
        if all(isinstance(value, (int, float)) or value is None for value in values):
            numeric_columns.append(key)
    return numeric_columns


def find_keyword(message: str, keywords: list[str]) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in keywords)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")
