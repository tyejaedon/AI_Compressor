#!/usr/bin/env python3
"""Tiny local smoke test for the autoencoder trainer."""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_root = project_root / "models" / "smoke_test"

    cmd = [
        sys.executable,
        str(project_root / "src" / "training" / "train_autoencoder_image_local.py"),
        "--preset",
        "custom",
        "--data-dir",
        str(project_root / "data" / "ImageData" / "archive"),
        "--real-only",
        "--real-file-limit",
        "8",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--block-size",
        "64",
        "--output-root",
        str(output_root),
    ]

    subprocess.check_call(cmd)
    print("[+] Smoke test passed")


if __name__ == "__main__":
    main()
