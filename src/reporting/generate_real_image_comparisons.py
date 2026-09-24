#!/usr/bin/env python3
"""Generate side-by-side PNG comparisons for real images using a trained autoencoder."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from report_markdown import write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-image compression comparisons.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/local_run_tuned_fast/20260629_065112/best_model.keras",
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ImageData/archive",
        help="Directory with real images (recursive)",
    )
    parser.add_argument("--count", type=int, default=8, help="Number of samples to render")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/real_data_comparisons",
        help="Output root folder for generated PNGs",
    )
    return parser.parse_args()


def collect_images(data_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_native(path):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def save_pair(path, original, reconstructed, psnr, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("Original (native)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstructed, 0.0, 1.0))
    axes[1].set_title(f"Reconstructed\nPSNR={psnr:.2f} dB")
    axes[1].axis("off")

    fig.suptitle(Path(path).name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_contact_sheet(rows, out_path):
    if not rows:
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8, max(3, n * 2.2)))
    if n == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        axes[i, 0].imshow(np.clip(row["original"], 0.0, 1.0))
        axes[i, 0].set_title(f"Original: {row['name']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(row["reconstructed"], 0.0, 1.0))
        axes[i, 1].set_title(f"Compressed/Reconstructed ({row['psnr']:.2f} dB)")
        axes[i, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    input_shape = model.input_shape
    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise ValueError(f"Expected image model input shape [None, H, W, 3], got {input_shape}")
    if input_shape[1] is not None or input_shape[2] is not None:
        raise ValueError(
            "Loaded model has fixed spatial input shape. "
            "Train with the updated dynamic-resolution image trainer to run native-resolution comparisons."
        )

    image_paths = collect_images(args.data_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in '{args.data_dir}'")

    rng = np.random.default_rng(args.seed)
    pick_count = min(args.count, len(image_paths))
    indices = rng.choice(len(image_paths), size=pick_count, replace=False)
    selected = [image_paths[int(i)] for i in indices]

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    rows = []
    report_samples = []
    for i, path in enumerate(selected, start=1):
        x = load_native(path)
        pred = model.predict(x[tf.newaxis, ...], verbose=0)[0]

        if tuple(pred.shape[:2]) != tuple(x.shape[:2]):
            raise ValueError(
                f"Model output size mismatch for '{path}': "
                f"input={tuple(x.shape[:2])}, output={tuple(pred.shape[:2])}"
            )

        psnr = float(tf.image.psnr(x, pred, max_val=1.0).numpy())

        out_path = os.path.join(run_dir, f"comparison_{i:02d}.png")
        save_pair(path, x.numpy(), pred, psnr, out_path)

        rows.append(
            {
                "name": Path(path).name,
                "original": x.numpy(),
                "reconstructed": pred,
                "resolution": [int(x.shape[1]), int(x.shape[0])],
                "psnr": psnr,
                "source_path": path,
                "comparison_png": out_path,
            }
        )
        report_samples.append(
            {
                "name": Path(path).name,
                "resolution": [int(x.shape[1]), int(x.shape[0])],
                "psnr": psnr,
                "source_path": path,
                "comparison_png": out_path,
            }
        )

    contact_sheet_path = os.path.join(run_dir, "comparison_contact_sheet.png")
    save_contact_sheet(rows, contact_sheet_path)

    report = {
        "model": args.model,
        "data_dir": args.data_dir,
        "model_input_shape": input_shape,
        "count": pick_count,
        "contact_sheet": contact_sheet_path,
        "samples": report_samples,
    }
    report_path = os.path.join(run_dir, "comparison_report.md")
    write_markdown_json_report(report, report_path, title="Image Comparison Report")

    print(run_dir)
    print(contact_sheet_path)
    print(report_path)


if __name__ == "__main__":
    main()

