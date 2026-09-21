#!/usr/bin/env python3
"""Local image autoencoder trainer optimized for lossy compression and smaller output files."""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers

from param_overrides import apply_overrides, load_overrides
from report_markdown import write_markdown_json_report


class StraightThroughQuantize(layers.Layer):
    """Round in forward pass while preserving gradients."""

    def call(self, inputs):
        quantized = tf.round(inputs * 255.0) / 255.0
        return inputs + tf.stop_gradient(quantized - inputs)


class RatePenalty(layers.Layer):
    """Adds a latent magnitude penalty as a simple bitrate proxy."""

    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)

    def call(self, inputs):
        rate_proxy = tf.reduce_mean(tf.abs(inputs))
        self.add_loss(self.weight * rate_proxy)
        return inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lossy image autoencoder locally.")
    parser.add_argument(
        "--preset",
        type=str,
        default="m1-air-balanced",
        choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"],
        help="Hardware/runtime preset; use 'custom' to keep explicit flags unchanged",
    )
    parser.add_argument("--use-cifar10", action="store_true", help="Download/load CIFAR-10 and train from it")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with image files (recursive scan)")
    parser.add_argument("--train-dir", type=str, default="", help="Explicit train image directory")
    parser.add_argument("--val-dir", type=str, default="", help="Explicit validation image directory")
    parser.add_argument("--test-dir", type=str, default="", help="Explicit test image directory")
    parser.add_argument("--split-file-limit", type=int, default=0, help="Optional cap per explicit split directory")
    parser.add_argument("--output-root", type=str, default="models/image_lossy_local_run", help="Output directory for weights/reports")
    parser.add_argument("--params-file", type=str, default="", help="Optional JSON file with parameter overrides")
    parser.add_argument("--block-size", type=int, default=128, help="Training patch size")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent bottleneck channels")
    parser.add_argument("--model-base-filters", type=int, default=32, help="Base Conv2D filter count for lossy encoder/decoder")
    parser.add_argument("--model-kernel-size", type=int, default=3, help="Kernel size used in lossy Conv2D/Conv2DTranspose blocks")
    parser.add_argument("--epochs", type=int, default=60, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--real-only", action="store_true", help="Require real files from --data-dir and disable CIFAR fallback")
    parser.add_argument("--real-file-limit", type=int, default=0, help="Optional cap for discovered real images (0 means all)")
    parser.add_argument("--min-real-images", type=int, default=300, help="Minimum number of discovered real images required for production real-only runs")
    parser.add_argument("--min-split-images", type=int, default=20, help="Minimum images required in each train/val/test split for real-only runs")
    parser.add_argument("--target-psnr", type=float, default=24.0, help="Training goal for validation PSNR")
    parser.add_argument("--rate-lambda", type=float, default=0.002, help="Weight of the latent rate penalty")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality for lossy sample export (1-100)")
    parser.add_argument("--sample-count", type=int, default=64, help="Number of test samples to export")
    parser.add_argument("--run-baseline-benchmark", dest="run_baseline_benchmark", action="store_true", help="Run matched-PSNR JPEG/WebP benchmark in report")
    parser.add_argument("--no-run-baseline-benchmark", dest="run_baseline_benchmark", action="store_false", help="Disable matched-PSNR JPEG/WebP benchmark")
    parser.add_argument(
        "--target-compression-ratio",
        type=float,
        default=None,
        help="Stop early when reconstructed JPEG/original JPEG ratio <= this value (example: 0.85)",
    )
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export Android-ready .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable TFLite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Use float16 optimization when exporting TFLite")
    parser.set_defaults(export_tflite=True, run_baseline_benchmark=True)
    return parser.parse_args()


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"block_size": 96, "latent_dim": 64, "batch_size": 4, "epochs": 60, "lr": 2.5e-4, "rate_lambda": 0.003},
        "m1-air-balanced": {"block_size": 128, "latent_dim": 72, "batch_size": 4, "epochs": 72, "lr": 2e-4, "rate_lambda": 0.002},
        "m1-air-quality": {"block_size": 128, "latent_dim": 96, "batch_size": 4, "epochs": 96, "lr": 1.5e-4, "rate_lambda": 0.0015},
    }

    if args.preset in preset_map:
        provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        key_to_flag = {
            "block_size": "--block-size",
            "latent_dim": "--latent-dim",
            "batch_size": "--batch-size",
            "epochs": "--epochs",
            "lr": "--lr",
            "rate_lambda": "--rate-lambda",
        }
        for key, value in preset_map[args.preset].items():
            if key_to_flag[key] not in provided_flags:
                setattr(args, key, value)

    args.jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))
    args.sample_count = max(32, min(128, int(args.sample_count)))
    args.min_real_images = max(3, int(args.min_real_images))
    args.min_split_images = max(1, int(args.min_split_images))
    args.latent_dim = max(16, int(args.latent_dim))
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.rate_lambda = float(max(1e-6, args.rate_lambda))
    if args.target_compression_ratio is not None and args.target_compression_ratio <= 0:
        raise ValueError("target-compression-ratio must be > 0 when provided")
    return args


def collect_image_paths(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root_dir)
    if not root.exists():
        return np.array([], dtype=np.str_)
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return np.array(sorted(paths), dtype=np.str_)


def split_indices(n, val_ratio, test_ratio, rng):
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1")

    idx = rng.permutation(n)
    test_n = max(1, int(n * test_ratio))
    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Dataset too small for requested split ratios")

    train_idx = idx[:train_n]
    val_idx = idx[train_n : train_n + val_n]
    test_idx = idx[train_n + val_n :]
    return train_idx, val_idx, test_idx


def make_dataset_from_paths(paths, block_size, batch_size, seed, shuffle):
    def _decode(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [block_size, block_size], method=tf.image.ResizeMethod.AREA)
        image = tf.cast(image, tf.float32) / 255.0
        return image, image

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_dataset_from_tensor(images, block_size, batch_size, seed, shuffle):
    def _prep(image):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [block_size, block_size], method=tf.image.ResizeMethod.AREA)
        return image, image

    ds = tf.data.Dataset.from_tensor_slices(images)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_explicit_split_datasets(args):
    train_paths = collect_image_paths(args.train_dir)
    val_paths = collect_image_paths(args.val_dir)
    test_paths = collect_image_paths(args.test_dir)

    if args.split_file_limit > 0:
        train_paths = train_paths[: args.split_file_limit]
        val_paths = val_paths[: args.split_file_limit]
        test_paths = test_paths[: args.split_file_limit]

    if min(len(train_paths), len(val_paths), len(test_paths)) <= 0:
        raise ValueError("Explicit split directories must each contain at least one supported image file")

    return (
        make_dataset_from_paths(train_paths, args.block_size, args.batch_size, args.seed, True),
        make_dataset_from_paths(val_paths, args.block_size, args.batch_size, args.seed, False),
        make_dataset_from_paths(test_paths, args.block_size, args.batch_size, args.seed, False),
        {
            "source": "explicit_directories",
            "train": int(len(train_paths)),
            "validation": int(len(val_paths)),
            "test": int(len(test_paths)),
            "train_images": int(len(train_paths)),
            "val_images": int(len(val_paths)),
            "test_images": int(len(test_paths)),
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "test_dir": args.test_dir,
        },
    )


def build_datasets(args):
    rng = np.random.default_rng(args.seed)

    if args.real_only and args.use_cifar10:
        raise ValueError("--real-only cannot be combined with --use-cifar10")

    explicit = any([args.train_dir, args.val_dir, args.test_dir])
    if explicit:
        if not all([args.train_dir, args.val_dir, args.test_dir]):
            raise ValueError("When using explicit split mode, provide --train-dir, --val-dir, and --test-dir together")
        return build_explicit_split_datasets(args)

    if args.use_cifar10:
        print("[*] Downloading and preparing CIFAR-10 dataset...")
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        train_idx, val_idx, _ = split_indices(len(x_train), args.val_ratio, args.test_ratio, rng)
        train_x = x_train[train_idx]
        val_x = x_train[val_idx]
        test_x = x_test
        return (
            make_dataset_from_tensor(train_x, args.block_size, args.batch_size, args.seed, True),
            make_dataset_from_tensor(val_x, args.block_size, args.batch_size, args.seed, False),
            make_dataset_from_tensor(test_x, args.block_size, args.batch_size, args.seed, False),
            {
                "source": "cifar10",
                "train": int(len(train_x)),
                "validation": int(len(val_x)),
                "test": int(len(test_x)),
            },
        )

    image_paths = collect_image_paths(args.data_dir)
    if args.real_file_limit > 0:
        image_paths = image_paths[: args.real_file_limit]

    if args.real_only and len(image_paths) < args.min_real_images:
        raise ValueError(
            f"--real-only requires at least {args.min_real_images} real images; found {len(image_paths)} in '{args.data_dir}'."
        )

    if len(image_paths) >= 3:
        train_idx, val_idx, test_idx = split_indices(len(image_paths), args.val_ratio, args.test_ratio, rng)
        if args.real_only and min(len(train_idx), len(val_idx), len(test_idx)) < args.min_split_images:
            raise ValueError(
                "--real-only split is too small after train/val/test partitioning; increase dataset size or adjust split ratios."
            )
        return (
            make_dataset_from_paths(image_paths[train_idx], args.block_size, args.batch_size, args.seed, True),
            make_dataset_from_paths(image_paths[val_idx], args.block_size, args.batch_size, args.seed, False),
            make_dataset_from_paths(image_paths[test_idx], args.block_size, args.batch_size, args.seed, False),
            {
                "source": "filesystem",
                "train": int(len(train_idx)),
                "validation": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        )

    if args.real_only:
        raise ValueError(
            f"No real image dataset found in '{args.data_dir}'. Add real files or disable --real-only."
        )

    raise ValueError(
        f"No image dataset found in '{args.data_dir}'. "
        "Provide a real image directory with at least 3 files or enable --use-cifar10."
    )


def reconstruction_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.90 * mse + 0.10 * ssim_term


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def build_autoencoder(latent_dim, rate_lambda, model_base_filters=32, model_kernel_size=3):
    inputs = layers.Input(shape=(None, None, 3))
    base_filters = max(16, int(model_base_filters))
    kernel_size = max(1, int(model_kernel_size))
    mid_filters = max(base_filters + base_filters // 2, base_filters + 8)
    deep_filters = max(base_filters * 2, base_filters + 16)
    tail_filters = max(16, base_filters // 2)

    x = layers.Conv2D(base_filters, kernel_size, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(mid_filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv2D(deep_filters, kernel_size, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(deep_filters, kernel_size, padding="same", activation="relu")(x)

    latent_pre = layers.Conv2D(latent_dim, 1, padding="same", activation="sigmoid", name="bottleneck_pre_quant")(x)
    latent_quant = StraightThroughQuantize(name="bottleneck_quant")(latent_pre)
    latent_penalized = RatePenalty(rate_lambda, name="rate_penalty")(latent_quant)

    x = layers.Conv2D(deep_filters, kernel_size, padding="same", activation="relu")(latent_penalized)
    x = layers.Conv2DTranspose(mid_filters, kernel_size, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(tail_filters, kernel_size, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, kernel_size, padding="same", activation="sigmoid")(x)

    model = Model(inputs, outputs, name="image_lossy_autoencoder")

    return model


class TargetPSNRCallback(callbacks.Callback):
    def __init__(self, target_psnr):
        super().__init__()
        self.target_psnr = float(target_psnr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_psnr_metric", logs.get("psnr_metric"))
        if current is None:
            return
        current_val = float(np.asarray(current).item())
        if current_val >= self.target_psnr:
            print(f"\n[+] Target PSNR reached at epoch {epoch + 1}: {current_val:.3f}")
            self.model.stop_training = True


def compute_batch_lossy_size_stats(images, preds, jpeg_quality):
    recon_jpeg_sizes = []
    orig_jpeg_sizes = []
    recon_png_sizes = []

    for i in range(int(images.shape[0])):
        orig_u8 = tf.cast(tf.round(tf.clip_by_value(images[i], 0.0, 1.0) * 255.0), tf.uint8)
        pred_u8 = tf.cast(tf.round(tf.clip_by_value(preds[i], 0.0, 1.0) * 255.0), tf.uint8)

        orig_jpeg = tf.io.encode_jpeg(orig_u8, quality=jpeg_quality, optimize_size=True)
        recon_jpeg = tf.io.encode_jpeg(pred_u8, quality=jpeg_quality, optimize_size=True)
        recon_png = tf.io.encode_png(pred_u8)

        orig_jpeg_sizes.append(int(tf.strings.length(orig_jpeg).numpy()))
        recon_jpeg_sizes.append(int(tf.strings.length(recon_jpeg).numpy()))
        recon_png_sizes.append(int(tf.strings.length(recon_png).numpy()))

    mean_orig_jpeg = float(np.mean(orig_jpeg_sizes))
    mean_recon_jpeg = float(np.mean(recon_jpeg_sizes))
    mean_recon_png = float(np.mean(recon_png_sizes))
    avg_psnr = float(tf.reduce_mean(tf.image.psnr(images, preds, max_val=1.0)).numpy())

    return {
        "avg_psnr": avg_psnr,
        "mean_original_jpeg_bytes": mean_orig_jpeg,
        "mean_reconstructed_jpeg_bytes": mean_recon_jpeg,
        "mean_reconstructed_png_bytes": mean_recon_png,
        "recon_jpeg_vs_orig_jpeg_ratio": mean_recon_jpeg / max(mean_orig_jpeg, 1.0),
        "recon_png_vs_recon_jpeg_ratio": mean_recon_png / max(mean_recon_jpeg, 1.0),
    }


class TargetCompressionRatioCallback(callbacks.Callback):
    def __init__(self, dataset, jpeg_quality, sample_count, target_ratio):
        super().__init__()
        self.dataset = dataset
        self.jpeg_quality = int(jpeg_quality)
        self.sample_count = int(sample_count)
        self.target_ratio = float(target_ratio)

    def on_epoch_end(self, epoch, logs=None):
        del logs
        for images, _ in self.dataset.take(1):
            images = images[: self.sample_count]
            if int(images.shape[0]) == 0:
                return
            preds = self.model.predict(images, verbose=0)
            size_stats = compute_batch_lossy_size_stats(images, preds, self.jpeg_quality)
            ratio = float(size_stats["recon_jpeg_vs_orig_jpeg_ratio"])
            print(f"\n[+] Epoch {epoch + 1} JPEG size ratio (recon/orig): {ratio:.3f}")
            if ratio <= self.target_ratio:
                print(
                    f"[+] Target compression ratio reached at epoch {epoch + 1}: "
                    f"{ratio:.3f} <= {self.target_ratio:.3f}"
                )
                self.model.stop_training = True
            return


def _encode_decode_codec(image_u8, codec, quality):
    if codec == "jpeg":
        encoded = tf.io.encode_jpeg(image_u8, quality=int(quality), optimize_size=True)
        decoded = tf.io.decode_jpeg(encoded, channels=3)
        return encoded, decoded
    if codec == "webp":
        if not hasattr(tf.io, "encode_webp") or not hasattr(tf.io, "decode_webp"):
            return None, None
        encoded = tf.io.encode_webp(image_u8, quality=float(quality))
        decoded = tf.io.decode_webp(encoded)
        return encoded, decoded
    raise ValueError(f"Unsupported codec: {codec}")


def _evaluate_codec_grid(images_u8, codec, target_psnr):
    quality_grid = list(range(5, 100, 5))
    candidates = []

    for q in quality_grid:
        sizes = []
        psnrs = []
        for i in range(int(images_u8.shape[0])):
            encoded, decoded = _encode_decode_codec(images_u8[i], codec, q)
            if encoded is None or decoded is None:
                return {
                    "codec": codec,
                    "available": False,
                    "reason": "TensorFlow build does not expose tf.io.encode_webp/decode_webp",
                }
            sizes.append(int(tf.strings.length(encoded).numpy()))
            orig_f = tf.cast(images_u8[i], tf.float32) / 255.0
            dec_f = tf.cast(decoded, tf.float32) / 255.0
            psnrs.append(float(tf.image.psnr(orig_f, dec_f, max_val=1.0).numpy()))

        candidates.append(
            {
                "quality": int(q),
                "mean_psnr": float(np.mean(psnrs)),
                "mean_bytes": float(np.mean(sizes)),
            }
        )

    above_target = [c for c in candidates if c["mean_psnr"] >= target_psnr]
    if above_target:
        matched = min(above_target, key=lambda c: c["mean_psnr"] - target_psnr)
    else:
        matched = min(candidates, key=lambda c: abs(c["mean_psnr"] - target_psnr))

    return {
        "codec": codec,
        "available": True,
        "target_psnr": float(target_psnr),
        "matched": matched,
        "candidates": candidates,
    }


def benchmark_baseline_codecs_at_matched_psnr(model, dataset, jpeg_quality, sample_count, model_size_stats):
    for images, _ in dataset.take(1):
        images = images[:sample_count]
        if int(images.shape[0]) == 0:
            return {"available": False, "reason": "No images available for benchmark"}

        preds = model.predict(images, verbose=0)
        images_u8 = tf.cast(tf.round(tf.clip_by_value(images, 0.0, 1.0) * 255.0), tf.uint8)

        model_target_psnr = float(model_size_stats.get("avg_psnr", 0.0))
        model_recon_jpeg_bytes = float(model_size_stats.get("mean_reconstructed_jpeg_bytes", 0.0))

        jpeg_baseline = _evaluate_codec_grid(images_u8, "jpeg", model_target_psnr)
        webp_baseline = _evaluate_codec_grid(images_u8, "webp", model_target_psnr)

        benchmark = {
            "available": True,
            "sample_count": int(images.shape[0]),
            "model_target_psnr": model_target_psnr,
            "model_reconstructed_jpeg_bytes": model_recon_jpeg_bytes,
            "model_export_jpeg_quality": int(jpeg_quality),
            "jpeg_baseline": jpeg_baseline,
            "webp_baseline": webp_baseline,
        }

        if jpeg_baseline.get("available"):
            jpeg_matched = float(jpeg_baseline["matched"]["mean_bytes"])
            benchmark["model_vs_jpeg_matched_size_ratio"] = model_recon_jpeg_bytes / max(jpeg_matched, 1.0)

        if webp_baseline.get("available"):
            webp_matched = float(webp_baseline["matched"]["mean_bytes"])
            benchmark["model_vs_webp_matched_size_ratio"] = model_recon_jpeg_bytes / max(webp_matched, 1.0)

        return benchmark

    return {"available": False, "reason": "No batches available in benchmark dataset"}


def export_tflite_model(model, run_dir, use_fp16=False):
    os.makedirs(run_dir, exist_ok=True)
    tflite_path = os.path.join(run_dir, "production_model.tflite")
    with tempfile.TemporaryDirectory(prefix="lossy_tflite_export_") as tmp_dir:
        saved_model_dir = os.path.join(tmp_dir, "saved_model")
        helper_path = os.path.join(tmp_dir, "convert_tflite.py")
        if hasattr(model, "export"):
            model.export(saved_model_dir)
        else:
            tf.saved_model.save(model, saved_model_dir)

        helper_code = """
import sys
import tensorflow as tf

saved_model_dir, output_tflite, fp16_flag = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
if fp16_flag:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(output_tflite, "wb") as f:
    f.write(tflite_model)
""".strip()
        with open(helper_path, "w", encoding="utf-8") as f:
            f.write(helper_code)

        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        proc = subprocess.run(
            [sys.executable, helper_path, saved_model_dir, tflite_path, "1" if use_fp16 else "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            raise RuntimeError("TFLite converter subprocess failed: " + " | ".join(tail))
    return tflite_path


def save_preview_reconstructions(model, dataset, run_dir):
    preview_path = os.path.join(run_dir, "preview_reconstructions.png")

    for images, _ in dataset.take(1):
        n = int(min(4, images.shape[0]))
        preds = model.predict(images[:n], verbose=0)
        originals = images[:n].numpy()

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(n):
            axes[0, i].imshow(np.clip(originals[i], 0.0, 1.0))
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(np.clip(preds[i], 0.0, 1.0))
            axes[1, i].set_title("Decoded")
            axes[1, i].axis("off")

        for i in range(n, 4):
            axes[0, i].axis("off")
            axes[1, i].axis("off")

        fig.tight_layout()
        fig.savefig(preview_path, dpi=140)
        plt.close(fig)
        break

    return preview_path


def save_lossy_samples_and_stats(model, dataset, run_dir, jpeg_quality, sample_count):
    sample_dir = os.path.join(run_dir, "lossy_samples")
    os.makedirs(sample_dir, exist_ok=True)

    for images, _ in dataset.take(1):
        images = images[:sample_count]
        preds = model.predict(images, verbose=0)

        size_stats = compute_batch_lossy_size_stats(images, preds, jpeg_quality)

        for i in range(int(images.shape[0])):
            orig_u8 = tf.cast(tf.round(tf.clip_by_value(images[i], 0.0, 1.0) * 255.0), tf.uint8)
            pred_u8 = tf.cast(tf.round(tf.clip_by_value(preds[i], 0.0, 1.0) * 255.0), tf.uint8)

            orig_jpeg = tf.io.encode_jpeg(orig_u8, quality=jpeg_quality, optimize_size=True)
            recon_jpeg = tf.io.encode_jpeg(pred_u8, quality=jpeg_quality, optimize_size=True)
            recon_png = tf.io.encode_png(pred_u8)

            tf.io.write_file(os.path.join(sample_dir, f"sample_{i:02d}_orig_q{jpeg_quality}.jpg"), orig_jpeg)
            tf.io.write_file(os.path.join(sample_dir, f"sample_{i:02d}_recon_q{jpeg_quality}.jpg"), recon_jpeg)
            tf.io.write_file(os.path.join(sample_dir, f"sample_{i:02d}_recon.png"), recon_png)

        return {
            "sample_dir": sample_dir,
            "sample_count": int(images.shape[0]),
            "jpeg_quality": int(jpeg_quality),
            **size_stats,
        }

    return {
        "sample_dir": sample_dir,
        "sample_count": 0,
        "jpeg_quality": int(jpeg_quality),
    }


def plot_history(history, plot_path):
    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, hist.get("loss", []), label="train")
    axes[0].plot(epochs, hist.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, hist.get("psnr_metric", []), label="train")
    axes[1].plot(epochs, hist.get("val_psnr_metric", []), label="val")
    axes[1].set_title("PSNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, hist.get("ssim_metric", []), label="train")
    axes[2].plot(epochs, hist.get("val_ssim_metric", []), label="val")
    axes[2].set_title("SSIM")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def save_report(args, split_info, history, eval_values, lossy_stats, baseline_benchmark, run_dir, plot_path):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "split_info": split_info,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "lossy_size_stats": lossy_stats,
        "matched_psnr_baseline_benchmark": baseline_benchmark,
        "plot_path": plot_path,
    }
    report_path = os.path.join(run_dir, "evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Image Lossy Compression Evaluation Report")
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def main():
    args = parse_args()
    args = apply_preset(args)
    if args.params_file:
        override_result = apply_overrides(args, load_overrides(args.params_file, section="image_lossy"))
        if override_result.applied:
            print(f"[*] Applied {len(override_result.applied)} params from {args.params_file}")
        if override_result.unknown:
            print(f"[!] Ignored unknown params in file: {sorted(override_result.unknown)}")
    args.jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))
    args.sample_count = max(32, min(128, int(args.sample_count)))
    args.min_real_images = max(3, int(args.min_real_images))
    args.min_split_images = max(1, int(args.min_split_images))
    args.latent_dim = max(16, int(args.latent_dim))
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.rate_lambda = float(max(1e-6, args.rate_lambda))
    if args.target_compression_ratio is not None and args.target_compression_ratio <= 0:
        raise ValueError("target-compression-ratio must be > 0 when provided")
    configure_runtime(args.seed)

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, split_info = build_datasets(args)
    steps_per_epoch = max(1, math.ceil(split_info["train"] / args.batch_size))
    val_steps = max(1, math.ceil(split_info["validation"] / args.batch_size))
    test_steps = max(1, math.ceil(split_info["test"] / args.batch_size))

    model = build_autoencoder(
        args.latent_dim,
        args.rate_lambda,
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=reconstruction_loss,
        metrics=["mse", psnr_metric, ssim_metric],
    )

    best_path = os.path.join(run_dir, "best_model.keras")
    cb = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=12, min_delta=1e-4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_delta=1e-4,
            cooldown=1,
            min_lr=1e-6,
            verbose=1,
        ),
        TargetPSNRCallback(args.target_psnr),
    ]
    if args.target_compression_ratio is not None:
        cb.append(
            TargetCompressionRatioCallback(
                dataset=val_data,
                jpeg_quality=args.jpeg_quality,
                sample_count=args.sample_count,
                target_ratio=args.target_compression_ratio,
            )
        )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb,
        verbose=1,
    )

    eval_values = model.evaluate(test_data, steps=test_steps, return_dict=True, verbose=0)
    plot_path = os.path.join(run_dir, "training_metrics.png")
    plot_history(history, plot_path)
    preview_path = save_preview_reconstructions(model, test_data, run_dir)
    lossy_stats = save_lossy_samples_and_stats(model, test_data, run_dir, args.jpeg_quality, args.sample_count)
    baseline_benchmark = None
    if args.run_baseline_benchmark:
        baseline_benchmark = benchmark_baseline_codecs_at_matched_psnr(
            model=model,
            dataset=test_data,
            jpeg_quality=args.jpeg_quality,
            sample_count=args.sample_count,
            model_size_stats=lossy_stats,
        )
    report_path = save_report(args, split_info, history, eval_values, lossy_stats, baseline_benchmark, run_dir, plot_path)

    weights_path = os.path.join(run_dir, "production_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        try:
            tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)
        except Exception as exc:
            print(f"[!] TFLite export skipped: {exc}")

    print("\n[+] Lossy image training complete")
    print(f"[+] Preset: {args.preset}")
    print(f"[+] Data source: {split_info['source']}")
    print(f"[+] Rate lambda: {args.rate_lambda}")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Best model: {best_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Plot: {plot_path}")
    print(f"[+] Preview: {preview_path}")
    print(f"[+] Lossy samples: {lossy_stats.get('sample_dir')}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")

    ratio = lossy_stats.get("recon_jpeg_vs_orig_jpeg_ratio")
    if ratio is not None:
        print(f"[+] Recon JPEG size ratio vs original JPEG baseline: {float(ratio):.3f}")

    if baseline_benchmark and baseline_benchmark.get("available"):
        jpeg_ratio = baseline_benchmark.get("model_vs_jpeg_matched_size_ratio")
        webp_ratio = baseline_benchmark.get("model_vs_webp_matched_size_ratio")
        if jpeg_ratio is not None:
            print(f"[+] Model JPEG size vs matched-PSNR JPEG baseline: {float(jpeg_ratio):.3f}")
        if webp_ratio is not None:
            print(f"[+] Model JPEG size vs matched-PSNR WebP baseline: {float(webp_ratio):.3f}")
        if baseline_benchmark.get("webp_baseline", {}).get("available") is False:
            print("[!] WebP baseline unavailable in this TensorFlow build; see report for details.")

    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(
            f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}). "
            "Increase epochs, reduce --rate-lambda, or use --preset m1-air-quality."
        )

    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")


if __name__ == "__main__":
    main()


