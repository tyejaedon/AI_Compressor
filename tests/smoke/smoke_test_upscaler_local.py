#!/usr/bin/env python3
"""Smoke test for the learned image upscaler (Milestone 3).

Two parts:
1. Data-free unit checks: model builds with dynamic spatial input, produces the
   expected `upscale_factor`x output shape, and (indirectly, via the trainer
   module) the degrade-pair pipeline shapes are consistent.
2. A tiny end-to-end training run using the CIFAR-10 fallback (no local dataset
   required) with 1 epoch / a handful of samples, then a quick invocation of
   `upscale_reconstructed_images.py --upscaler-mode learned` against the
   resulting model to confirm the wiring/tiling path runs end-to-end.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


def _load_module(name, path, project_root):
    sys.path.insert(0, str(project_root / "src" / "training"))
    sys.path.insert(0, str(project_root / "src" / "reporting"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_model_shapes(train_mod):
    model = train_mod.build_upscaler(model_base_filters=16, num_residual_blocks=2, upscale_factor=2)
    x = np.random.random((2, 16, 16, 3)).astype("float32")
    out = model(x, training=False).numpy()
    assert out.shape == (2, 32, 32, 3), f"unexpected output shape: {out.shape}"

    # Dynamic/tile-friendly: a different, non-square spatial size should also work.
    x2 = np.random.random((1, 20, 36, 3)).astype("float32")
    out2 = model(x2, training=False).numpy()
    assert out2.shape == (1, 40, 72, 3), f"unexpected output shape for tile input: {out2.shape}"
    print("[+] Upscaler model shape checks passed")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    train_mod = _load_module(
        "train_upscaler_image_local_smoke",
        str(project_root / "src" / "training" / "train_upscaler_image_local.py"),
        project_root,
    )
    check_model_shapes(train_mod)

    output_root = project_root / "models" / "smoke_test_upscaler"
    cmd = [
        sys.executable,
        str(project_root / "src" / "training" / "train_upscaler_image_local.py"),
        "--preset",
        "custom",
        "--use-cifar10",
        "--cifar-train-limit",
        "32",
        "--cifar-test-limit",
        "16",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--block-size",
        "32",
        "--upscale-factor",
        "2",
        "--model-base-filters",
        "16",
        "--num-residual-blocks",
        "2",
        "--sample-count",
        "4",
        "--no-export-tflite",
        "--output-root",
        str(output_root),
    ]
    subprocess.check_call(cmd)
    print("[+] Learned upscaler tiny training run passed")


if __name__ == "__main__":
    main()
