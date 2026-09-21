#!/usr/bin/env python3
"""Tiny smoke test for the lossy image autoencoder trainer."""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_root = project_root / "models" / "smoke_test_image_lossy"

    cmd = [
        sys.executable,
        str(project_root / "train_autoencoder_image_lossy_local.py"),
        "--preset",
        "custom",
        "--data-dir",
        str(project_root / "data" / "ImageData" / "archive"),
        "--real-only",
        "--real-file-limit",
        "8",
        "--min-real-images",
        "8",
        "--min-split-images",
        "1",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--block-size",
        "64",
        "--latent-dim",
        "16",
        "--rate-lambda",
        "0.02",
        "--sample-count",
        "2",
        "--no-export-tflite",
        "--output-root",
        str(output_root),
    ]

    subprocess.check_call(cmd)
    print("[+] Lossy image smoke test passed")


if __name__ == "__main__":
    main()

