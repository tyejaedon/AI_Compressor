#!/usr/bin/env python3
"""Tiny local smoke test for the autoencoder trainer."""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_root = project_root / "models" / "smoke_test" / "image_cache_local"
    run_output_root = output_root / "runs"
    cache_dir = output_root / "tensor_cache"

    if output_root.exists():
        shutil.rmtree(output_root)

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
        str(run_output_root),
        "--cache-dir",
        str(cache_dir),
        "--no-export-tflite",
    ]

    first = subprocess.run(cmd, check=True, capture_output=True, text=True)
    second = subprocess.run(cmd, check=True, capture_output=True, text=True)

    first_output = first.stdout + first.stderr
    second_output = second.stdout + second.stderr

    assert "Cache train: status=built" in first_output, first_output
    assert "Cache train: status=hit" in second_output, second_output
    print("[+] Smoke test passed")


if __name__ == "__main__":
    main()
