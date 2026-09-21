#!/usr/bin/env python3
"""Tiny smoke test for canonical audio dataset preparation."""

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


def write_test_wav(path: Path, sample_rate: int = 22050, seconds: float = 0.4) -> None:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    clip = 0.4 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm = np.clip(clip * 32767.0, -32768, 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_root = project_root / "models" / "smoke_test_prepare_audio_dataset" / "input"
    output_root = project_root / "models" / "smoke_test_prepare_audio_dataset" / "output"

    write_test_wav(input_root / "tone.wav")

    cmd = [
        sys.executable,
        str(project_root / "prepare_audio_dataset.py"),
        "--input-dir",
        str(input_root),
        "--output-dir",
        str(output_root),
        "--target-sample-rate",
        "16000",
        "--output-format",
        "wav",
        "--normalization",
        "rms_peak",
        "--target-rms",
        "0.12",
        "--target-peak",
        "0.95",
        "--remove-dc",
        "--overwrite",
    ]
    subprocess.check_call(cmd)

    out_audio = output_root / "tone.wav"
    sidecar = output_root / "tone.wav.metadata.json"
    report = output_root / "preparation_report.md"

    if not out_audio.exists() or not sidecar.exists() or not report.exists():
        raise RuntimeError("Expected output audio, metadata sidecar, and report were not created")

    print("[+] Audio dataset preparation smoke test passed")


if __name__ == "__main__":
    main()

