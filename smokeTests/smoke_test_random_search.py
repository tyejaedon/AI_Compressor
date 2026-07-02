#!/usr/bin/env python3
"""Smoke test for random_search_hyperparams.py in dry-run mode."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(project_root / "random_search_hyperparams.py"),
        "--modality",
        "image",
        "--trials",
        "3",
        "--resource-profile",
        "tiny",
        "--max-minutes",
        "1",
        "--dry-run",
        "--output-root",
        "models/random_search_smoke",
    ]

    subprocess.check_call(cmd)
    print("[+] Random-search smoke test passed")


if __name__ == "__main__":
    main()

