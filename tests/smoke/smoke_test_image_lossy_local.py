#!/usr/bin/env python3
"""Tiny smoke test for the lossy image autoencoder trainer."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


def _load_lossy_module(project_root):
    sys.path.insert(0, str(project_root / "src" / "training"))
    sys.path.insert(0, str(project_root / "src" / "reporting"))
    spec = importlib.util.spec_from_file_location(
        "train_autoencoder_image_lossy_local_smoke",
        str(project_root / "src" / "training" / "train_autoencoder_image_lossy_local.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_quantization_round_trip(project_root):
    """Milestone 2: quantized latents must be exactly representable at the chosen bit-depth."""
    mod = _load_lossy_module(project_root)
    x = np.random.random((2, 16, 16, 3)).astype("float32")
    for bit_depth in (4, 6, 8, 10):
        for quant_noise_anneal in (False, True):
            model = mod.build_autoencoder(16, 0.01, latent_bit_depth=bit_depth, quant_noise_anneal=quant_noise_anneal)
            latent_model = mod.get_latent_submodel(model)
            latent = latent_model(x, training=False).numpy()  # inference path: always hard-quantized
            max_level = (1 << bit_depth) - 1
            scaled = latent * max_level
            residual = np.abs(scaled - np.round(scaled))
            assert residual.max() < 1e-4, (
                f"latent not losslessly quantized at bit_depth={bit_depth} "
                f"(quant_noise_anneal={quant_noise_anneal}): max residual={residual.max()}"
            )
    print("[+] Latent quantization round-trip check passed for all bit-depths")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    check_quantization_round_trip(project_root)

    output_root = project_root / "models" / "smoke_test_image_lossy"

    cmd = [
        sys.executable,
        str(project_root / "src" / "training" / "train_autoencoder_image_lossy_local.py"),
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
        "--latent-bit-depth",
        "6",
        "--enable-entropy-coding",
        "--entropy-coding-samples",
        "2",
        "--no-export-tflite",
        "--output-root",
        str(output_root),
    ]

    subprocess.check_call(cmd)
    print("[+] Lossy image smoke test passed")


if __name__ == "__main__":
    main()

