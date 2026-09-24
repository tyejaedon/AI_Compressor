#!/usr/bin/env python3
"""Helpers for writing/reading JSON-backed Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _title_from_path(report_path: Path) -> str:
    return report_path.stem.replace("_", " ").strip().title() or "Report"


def write_markdown_json_report(payload: dict[str, Any], report_path: str | Path, title: str | None = None) -> Path:
    """Write a report as Markdown with an embedded JSON block."""
    path = Path(report_path)
    report_title = title or _title_from_path(path)
    body = json.dumps(payload, indent=2, ensure_ascii=True)
    text = (
        f"# {report_title}\n\n"
        f"- Source: `{path.name}`\n\n"
        "```json\n"
        f"{body}\n"
        "```\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def read_markdown_json_report(report_path: str | Path) -> dict[str, Any] | None:
    """Read JSON payload from either .json or Markdown report files."""
    path = Path(report_path)
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    fence = "```json"
    start = text.find(fence)
    if start != -1:
        start += len(fence)
        end = text.find("```", start)
        if end != -1:
            block = text[start:end].strip()
            try:
                data = json.loads(block)
                return data if isinstance(data, dict) else None
            except Exception:
                return None

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

