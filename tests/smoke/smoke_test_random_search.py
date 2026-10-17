#!/usr/bin/env python3
"""Smoke test for random_search_hyperparams.py in dry-run mode."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    cmd_image = [
        sys.executable,
        str(project_root / "src" / "training" / "random_search_hyperparams.py"),
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
    subprocess.check_call(cmd_image)

    cmd_audio = [
        sys.executable,
        str(project_root / "src" / "training" / "random_search_hyperparams.py"),
        "--modality",
        "audio",
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
    subprocess.check_call(cmd_audio)

    print("[+] Random-search smoke test passed (image and audio)")


if __name__ == "__main__":
    main()
