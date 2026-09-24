#!/usr/bin/env python3
"""Helpers for applying model parameter overrides from JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_overrides(params_file: str, section: str | None = None) -> dict[str, Any]:
    path = Path(params_file)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Params file must contain a JSON object: {path}")

    if section and section in raw:
        section_obj = raw[section]
        if not isinstance(section_obj, dict):
            raise ValueError(f"Section '{section}' must be a JSON object in {path}")
        return section_obj

    return raw


def _provided_flag_set(argv: list[str] | None = None) -> set[str]:
    tokens = (argv if argv is not None else sys.argv)[1:]
    return {token.split("=", 1)[0] for token in tokens if token.startswith("--")}


def _flag_variants(key: str) -> set[str]:
    base = "--" + key.replace("_", "-")
    return {base, "--no-" + base[2:]}


def apply_overrides(
    args: Any,
    overrides: dict[str, Any],
    *,
    argv: list[str] | None = None,
) -> SimpleNamespace:
    provided_flags = _provided_flag_set(argv)
    unknown: list[str] = []
    applied: list[str] = []

    for key, value in overrides.items():
        if not hasattr(args, key):
            unknown.append(key)
            continue
        if _flag_variants(key) & provided_flags:
            continue
        setattr(args, key, value)
        applied.append(key)

    return SimpleNamespace(applied=applied, unknown=unknown)

