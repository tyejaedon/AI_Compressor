#!/usr/bin/env python3
"""Quick smoke test for audio trainer."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "train_autoencoder_audio_local.py"),
        "--preset",
        "custom",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--latent-dim",
        "64",
        "--clip-seconds",
        "0.5",
        "--train-samples",
        "120",
        "--val-samples",
        "32",
        "--test-samples",
        "32",
        "--output-root",
        str(root / "models" / "audio_smoke"),
    ]
    subprocess.check_call(cmd)
    print("[+] Audio smoke test passed")


if __name__ == "__main__":
    main()

