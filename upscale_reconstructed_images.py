#!/usr/bin/env python3
"""Run image autoencoder reconstruction and upscale outputs to original size or 720p."""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from report_markdown import read_markdown_json_report, write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct and upscale real images with a trained autoencoder.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/production_bundle/best_20260629_142034/image/best_model.keras",
        help="Path to trained Keras image model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ImageData/archive/data",
        help="Directory with input images (recursive)",
    )
    parser.add_argument("--count", type=int, default=8, help="Number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--upscale-mode",
        type=str,
        default="original",
        choices=["original", "720p"],
        help="Target output size mode",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/upscaled_reconstructions",
        help="Output root directory",
    )
    return parser.parse_args()


def _read_latent_dim_from_report(report_path):
    payload = read_markdown_json_report(report_path)
    if not payload:
        return None
    latent = payload.get("config", {}).get("latent_dim")
    if isinstance(latent, int) and latent > 0:
        return latent
    return None


def load_model_with_fallback(model_path):
    model_path = Path(model_path)
    try:
        # Local trusted artifact; required for Lambda layers in some training runs.
        return tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    except Exception as err:
        weights_path = model_path.with_name("production_model.weights.h5")
        report_path = model_path.with_name("evaluation_report.md")
        if not (weights_path.exists() and report_path.exists()):
            raise err

        from train_autoencoder_image_local import build_autoencoder

        latent_dim = _read_latent_dim_from_report(report_path) or 256
        model = build_autoencoder(latent_dim)
        # Build variables before loading weights.
        _ = model(tf.zeros((1, 96, 96, 3), dtype=tf.float32), training=False)
        model.load_weights(str(weights_path))
        return model


def collect_images(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def decode_image(path):
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def target_size_keep_aspect(orig_w, orig_h, mode):
    if mode == "original":
        return int(orig_w), int(orig_h)

    if mode == "720p":
        if orig_w >= orig_h:
            target_h = 720
            target_w = int(round(target_h * (orig_w / max(orig_h, 1))))
        else:
            target_w = 720
            target_h = int(round(target_w * (orig_h / max(orig_w, 1))))
        return int(target_w), int(target_h)

    raise ValueError("Unsupported upscale mode")


def save_comparison(original, recon_small, recon_upscaled, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(recon_small, 0.0, 1.0))
    axes[1].set_title("Reconstructed (model size)")
    axes[1].axis("off")

    axes[2].imshow(np.clip(recon_upscaled, 0.0, 1.0))
    axes[2].set_title("Upscaled")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=165)
    plt.close(fig)


def main():
    args = parse_args()

    model = load_model_with_fallback(args.model)
    _, in_h, in_w, in_c = model.input_shape
    if in_c != 3:
        raise ValueError(f"Expected RGB model with 3 channels, got {model.input_shape}")

    dynamic_spatial = in_h is None or in_w is None

    paths = collect_images(args.data_dir)
    if len(paths) == 0:
        raise ValueError(f"No images found in '{args.data_dir}'")

    rng = np.random.default_rng(args.seed)
    k = min(args.count, len(paths))
    picks = [paths[int(i)] for i in rng.choice(len(paths), size=k, replace=False)]

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    out_recon = os.path.join(run_dir, "upscaled")
    out_comp = os.path.join(run_dir, "comparisons")
    os.makedirs(out_recon, exist_ok=True)
    os.makedirs(out_comp, exist_ok=True)

    samples = []
    for idx, path in enumerate(picks, start=1):
        img = decode_image(path)
        orig_h = int(img.shape[0])
        orig_w = int(img.shape[1])

        if dynamic_spatial:
            # Feed full-resolution image directly when the model supports dynamic spatial dims.
            model_in = img
        else:
            model_in = tf.image.resize(img, [in_h, in_w], method=tf.image.ResizeMethod.BILINEAR)
        model_in_h = int(model_in.shape[0])
        model_in_w = int(model_in.shape[1])
        pred = model.predict(model_in[tf.newaxis, ...], verbose=0)[0]

        target_w, target_h = target_size_keep_aspect(orig_w, orig_h, args.upscale_mode)
        pred_up = tf.image.resize(pred, [target_h, target_w], method=tf.image.ResizeMethod.BICUBIC).numpy()

        # Create comparison using original resized to target for visual parity and PSNR
        orig_target = tf.image.resize(img, [target_h, target_w], method=tf.image.ResizeMethod.BILINEAR).numpy()
        psnr_up = float(tf.image.psnr(orig_target, pred_up, max_val=1.0).numpy())

        stem = Path(path).stem
        up_path = os.path.join(out_recon, f"{idx:02d}_{stem}_upscaled.png")
        comp_path = os.path.join(out_comp, f"{idx:02d}_{stem}_comparison.png")

        tf.keras.utils.save_img(up_path, np.clip(pred_up, 0.0, 1.0))
        save_comparison(img.numpy(), pred, pred_up, f"{Path(path).name} | PSNR(upscaled)={psnr_up:.2f} dB", comp_path)

        samples.append(
            {
                "source": path,
                "name": Path(path).name,
                "original_size": [orig_w, orig_h],
                "model_input_size": [model_in_w, model_in_h],
                "upscaled_size": [target_w, target_h],
                "psnr_upscaled": psnr_up,
                "upscaled_png": up_path,
                "comparison_png": comp_path,
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "upscale_mode": args.upscale_mode,
        "output_dir": run_dir,
        "samples": samples,
    }
    report_path = os.path.join(run_dir, "upscale_report.md")
    write_markdown_json_report(report, report_path, title="Upscale Report")

    print(run_dir)
    print(report_path)


if __name__ == "__main__":
    main()

