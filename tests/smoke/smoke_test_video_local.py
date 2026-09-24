#!/usr/bin/env python3
"""Quick smoke test for video trainer."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent
    cmd = [
        sys.executable,
        str(root / "src" / "training" / "train_autoencoder_video_local.py"),
        "--preset",
        "custom",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--latent-dim",
        "96",
        "--frames",
        "6",
        "--height",
        "48",
        "--width",
        "48",
        "--data-dir",
        str(root / "data" / "VIDEO DATA"),
        "--real-max-videos",
        "3",
        "--real-max-clips",
        "120",
        "--no-export-tflite",
        "--output-root",
        str(root / "models" / "video_smoke"),
    ]
    subprocess.check_call(cmd)
    print("[+] Video smoke test passed")


if __name__ == "__main__":
    main()

