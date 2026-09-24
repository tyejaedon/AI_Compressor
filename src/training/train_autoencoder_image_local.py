#!/usr/bin/env python3
"""Local MacBook-friendly image autoencoder trainer with Edge-Sharpening & Remaster Pipeline."""

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers
from param_overrides import apply_overrides, load_overrides

# Keep your local custom report writer intact
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
try:
    from report_markdown import write_markdown_json_report
except ImportError:
    # Fallback to dummy implementation if running in a raw environment
    def write_markdown_json_report(payload, report_path):
        import json
        with open(report_path, "w") as f:
            json.dump(payload, f, indent=4)


def parse_args_extended():
    parser = argparse.ArgumentParser(description="Train an image autoencoder locally.")
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
    parser.add_argument("--holdout-dir", type=str, default="", help="Optional external holdout directory")
    parser.add_argument("--split-file-limit", type=int, default=0, help="Optional cap per explicit split directory")
    parser.add_argument("--valid-dir", type=str, default="data/cifar10_valid", help="Local directory placeholder for CIFAR-10 validation assets")
    parser.add_argument("--output-root", type=str, default="models/local_run", help="Output directory for weights/reports")
    parser.add_argument("--params-file", type=str, default="", help="Optional JSON file with parameter overrides")
    parser.add_argument("--block-size", type=int, default=128, help="Training patch size for cropped image blocks")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent bottleneck size")
    parser.add_argument("--model-base-filters", type=int, default=64, help="Base Conv2D filter count for encoder/decoder blocks")
    parser.add_argument("--model-kernel-size", type=int, default=3, help="Kernel size used in core Conv2D blocks")
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Scaling factor applied to residual head before adding to input")
    parser.add_argument("--epochs", type=int, default=36, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--upscale-factor", type=int, default=2, help="Super-resolution factor used to degrade model inputs")
    parser.add_argument(
        "--degrade-interp",
        type=str,
        default="bicubic",
        choices=["area", "bilinear", "bicubic"],
        help="Interpolation used while downscaling real-image patches",
    )
    parser.add_argument("--real-only", action="store_true", help="Require real files and disable synthetic fallbacks")
    parser.add_argument("--real-file-limit", type=int, default=0, help="Optional cap for discovered real image files")
    parser.add_argument("--train-patches-per-image", type=int, default=6, help="Random crop samples drawn per real training image each epoch")
    parser.add_argument("--real-augment", dest="real_augment", action="store_true", help="Enable flips on real-image crops")
    parser.add_argument("--no-real-augment", dest="real_augment", action="store_false", help="Disable flips")
    parser.add_argument("--cifar-train-limit", type=int, default=0, help="Optional cap for CIFAR-10 train samples")
    parser.add_argument("--cifar-test-limit", type=int, default=0, help="Optional cap for CIFAR-10 test samples")
    parser.add_argument("--target-psnr", type=float, default=60.0, help="Training goal for validation PSNR")
    parser.add_argument("--disable-target-psnr-stop", action="store_true", help="Disable target-PSNR early stopping callback")
    parser.add_argument(
        "--target-output-jpeg-ratio",
        type=float,
        default=1.30,
        help="Target minimum output-vs-target JPEG(Q90) size ratio for upscaler detail emphasis",
    )
    parser.add_argument(
        "--disable-target-output-jpeg-stop",
        action="store_true",
        help="Disable target-output-JPEG-ratio early stopping callback",
    )
    parser.add_argument(
        "--target-output-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality used when evaluating output-vs-target size ratio",
    )
    parser.add_argument(
        "--target-output-jpeg-samples",
        type=int,
        default=32,
        help="Validation sample count used for output-vs-target JPEG ratio callback",
    )
    parser.add_argument(
        "--loss-profile",
        type=str,
        default="balanced",
        choices=["balanced", "psnr", "perceptual"],
        help="Preset weighting profile for reconstruction loss",
    )
    parser.add_argument("--loss-mse-weight", type=float, default=0.60, help="Weight for MSE term in reconstruction loss")
    parser.add_argument("--loss-l1-weight", type=float, default=0.10, help="Weight for L1 term in reconstruction loss")
    parser.add_argument("--loss-ssim-weight", type=float, default=0.15, help="Weight for SSIM term in reconstruction loss")
    parser.add_argument("--loss-edge-weight", type=float, default=0.15, help="Weight for Sobel Edge preservation loss")
    parser.add_argument("--loss-hard-weight", type=float, default=1.0, help="Extra penalty multiplier for larger residuals")
    parser.add_argument("--loss-edge-power", type=float, default=2.0, help="Exponent for edge mismatch penalty (>=1)")
    parser.add_argument("--grad-clipnorm", type=float, default=1.0, help="Gradient clip norm for Adam optimizer")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export Android-ready .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable TFLite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Use float16 optimization when exporting TFLite")
    parser.set_defaults(export_tflite=True, real_augment=True)
    return parser.parse_args()


def apply_preset(args):
    provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    preset_map = {
        "m1-air-fast": {"block_size": 96, "latent_dim": 96, "batch_size": 4, "epochs": 16, "lr": 2.2e-4},
        "m1-air-balanced": {"block_size": 128, "latent_dim": 128, "batch_size": 4, "epochs": 36, "lr": 1.5e-4},
        "m1-air-quality": {"block_size": 128, "latent_dim": 192, "batch_size": 4, "epochs": 72, "lr": 9e-5},
    }

    if args.preset in preset_map:
        key_to_flag = {
            "block_size": "--block-size",
            "latent_dim": "--latent-dim",
            "batch_size": "--batch-size",
            "epochs": "--epochs",
            "lr": "--lr",
        }

        for key, value in preset_map[args.preset].items():
            if key_to_flag[key] not in provided_flags:
                setattr(args, key, value)

    loss_profiles = {
        # Profile weights sum: (MSE, L1, SSIM, Edge)
        "balanced": (0.55, 0.15, 0.10, 0.20),
        "psnr": (0.75, 0.05, 0.05, 0.15),
        "perceptual": (0.35, 0.15, 0.25, 0.25),
    }
    if not any(flag in provided_flags for flag in ["--loss-mse-weight", "--loss-l1-weight", "--loss-ssim-weight", "--loss-edge-weight"]):
        mse_w, l1_w, ssim_w, edge_w = loss_profiles[args.loss_profile]
        args.loss_mse_weight = mse_w
        args.loss_l1_weight = l1_w
        args.loss_ssim_weight = ssim_w
        args.loss_edge_weight = edge_w

    args.grad_clipnorm = max(0.0, float(args.grad_clipnorm))
    args.loss_hard_weight = max(0.0, float(args.loss_hard_weight))
    args.loss_edge_power = max(1.0, float(args.loss_edge_power))
    args.target_output_jpeg_ratio = max(0.0, float(args.target_output_jpeg_ratio))
    args.target_output_jpeg_quality = int(np.clip(args.target_output_jpeg_quality, 1, 100))
    args.target_output_jpeg_samples = max(1, int(args.target_output_jpeg_samples))
    args.residual_scale = max(0.0, float(args.residual_scale))
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
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


def filter_paths_by_min_size(paths, min_size):
    min_side = max(1, int(min_size))
    kept = []
    skipped = 0
    for path in paths:
        try:
            image_bytes = tf.io.read_file(path)
            image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
            shape = tf.shape(image)
            h = int(shape[0].numpy())
            w = int(shape[1].numpy())
            if h >= min_side and w >= min_side:
                kept.append(str(path))
            else:
                skipped += 1
        except Exception:
            skipped += 1
    return np.array(kept, dtype=np.str_), skipped


def detect_split_dirs(data_dir):
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        return None

    def _pick(names):
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return ""

    train_dir = _pick(["train", "training"])
    val_dir = _pick(["val", "valid", "validation", "dev"])
    test_dir = _pick(["test", "testing"])
    found = [bool(train_dir), bool(val_dir), bool(test_dir)]
    if not any(found):
        return None
    return {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir,
        "complete": all(found),
    }


def make_dataset_from_paths(
    paths,
    block_size,
    batch_size,
    seed,
    shuffle,
    patches_per_image=1,
    augment=False,
    center_crop=False,
    upscale_factor=4,
    degrade_interp="bicubic",
):
    block = int(block_size)
    patch_count = max(1, int(patches_per_image))
    up_factor = max(1, int(upscale_factor))
    down_block = max(8, block // up_factor)
    interp_map = {
        "area": tf.image.ResizeMethod.AREA,
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
    }
    degrade_method = interp_map.get(str(degrade_interp).lower(), tf.image.ResizeMethod.BICUBIC)

    def _decode(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape([None, None, 3])
        return tf.clip_by_value(image, 0.0, 1.0)

    def _sample_pair(image):
        if center_crop:
            shape = tf.shape(image)
            off_h = (shape[0] - block) // 2
            off_w = (shape[1] - block) // 2
            patch = tf.image.crop_to_bounding_box(image, off_h, off_w, block, block)
        else:
            patch = tf.image.random_crop(image, [block, block, 3])
            if augment:
                patch = tf.image.random_flip_left_right(patch)
                patch = tf.image.random_flip_up_down(patch)
        patch = tf.clip_by_value(patch, 0.0, 1.0)

        if up_factor > 1:
            # Multi-stage Samsung Remaster simulation: Add compression artifacts + Downscaling Noise
            down = tf.image.resize(patch, [down_block, down_block], method=degrade_method)

            # Inject standard Gaussian sensory noise simulating low-light lenses
            noise = tf.random.normal(shape=tf.shape(down), mean=0.0, stddev=0.015, seed=seed)
            down = tf.clip_by_value(down + noise, 0.0, 1.0)

            lowres_input = tf.image.resize(down, [block, block], method=tf.image.ResizeMethod.BILINEAR)
        else:
            lowres_input = patch

        lowres_input = tf.clip_by_value(lowres_input, 0.0, 1.0)
        return lowres_input, patch

    def _path_to_patches(path):
        image = _decode(path)
        patch_ds = tf.data.Dataset.range(patch_count)
        return patch_ds.map(lambda _: _sample_pair(image), num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.interleave(
        _path_to_patches,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_dataset_from_tensor(images, block_size, batch_size, seed, shuffle):
    def _prep(image):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [block_size, block_size], method=tf.image.ResizeMethod.BILINEAR)
        return image, image

    ds = tf.data.Dataset.from_tensor_slices(images)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cifar10_datasets(args):
    print("[*] Downloading and preparing CIFAR-10 dataset...")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.valid_dir, exist_ok=True)

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    if args.cifar_train_limit > 0:
        x_train = x_train[: args.cifar_train_limit]
    if args.cifar_test_limit > 0:
        x_test = x_test[: args.cifar_test_limit]

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, _ = split_indices(len(x_train), args.val_ratio, args.test_ratio, rng)

    train_x = x_train[train_idx]
    val_x = x_train[val_idx]
    test_x = x_test

    return (
        make_dataset_from_tensor(train_x, args.block_size, args.batch_size, args.seed, shuffle=True),
        make_dataset_from_tensor(val_x, args.block_size, args.batch_size, args.seed, shuffle=False),
        make_dataset_from_tensor(test_x, args.block_size, args.batch_size, args.seed, shuffle=False),
        {
            "source": "cifar10",
            "train": int(len(train_x)),
            "validation": int(len(val_x)),
            "test": int(len(test_x)),
        },
    )


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

    train_paths, train_skipped = filter_paths_by_min_size(train_paths, args.block_size)
    val_paths, val_skipped = filter_paths_by_min_size(val_paths, args.block_size)
    test_paths, test_skipped = filter_paths_by_min_size(test_paths, args.block_size)
    if train_skipped + val_skipped + test_skipped > 0:
        print(
            "[*] Dropped "
            f"{train_skipped + val_skipped + test_skipped} images smaller than block-size={args.block_size} "
            "to avoid synthetic SR targets."
        )
    if min(len(train_paths), len(val_paths), len(test_paths)) <= 0:
        raise ValueError(
            "After size filtering, at least one split is empty. "
            "Use a smaller --block-size or provide higher-resolution images."
        )

    return (
        make_dataset_from_paths(
            train_paths,
            args.block_size,
            args.batch_size,
            args.seed,
            shuffle=True,
            patches_per_image=args.train_patches_per_image,
            augment=args.real_augment,
            center_crop=False,
            upscale_factor=args.upscale_factor,
            degrade_interp=args.degrade_interp,
        ),
        make_dataset_from_paths(
            val_paths,
            args.block_size,
            args.batch_size,
            args.seed,
            shuffle=False,
            patches_per_image=1,
            augment=False,
            center_crop=True,
            upscale_factor=args.upscale_factor,
            degrade_interp=args.degrade_interp,
        ),
        make_dataset_from_paths(
            test_paths,
            args.block_size,
            args.batch_size,
            args.seed,
            shuffle=False,
            patches_per_image=1,
            augment=False,
            center_crop=True,
            upscale_factor=args.upscale_factor,
            degrade_interp=args.degrade_interp,
        ),
        {
            "source": "explicit_directories",
            "train": int(len(train_paths) * max(1, int(args.train_patches_per_image))),
            "validation": int(len(val_paths)),
            "test": int(len(test_paths)),
            "train_images": int(len(train_paths)),
            "val_images": int(len(val_paths)),
            "test_images": int(len(test_paths)),
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "test_dir": args.test_dir,
            "block_size": int(args.block_size),
            "upscale_factor": int(max(1, int(args.upscale_factor))),
            "degrade_interp": args.degrade_interp,
        },
    )


def build_datasets(args):
    if args.real_only and args.use_cifar10:
        raise ValueError("--real-only cannot be combined with --use-cifar10")

    explicit = any([args.train_dir, args.val_dir, args.test_dir])
    if explicit:
        if not all([args.train_dir, args.val_dir, args.test_dir]):
            raise ValueError("When using explicit split mode, provide --train-dir, --val-dir, and --test-dir together")
        return build_explicit_split_datasets(args)

    if args.use_cifar10:
        return build_cifar10_datasets(args)

    detected_split_dirs = detect_split_dirs(args.data_dir)
    if detected_split_dirs is not None:
        if detected_split_dirs["complete"]:
            print("[*] Detected train/val/test subdirectories under --data-dir; using explicit split mode.")
            auto_args = argparse.Namespace(**vars(args))
            auto_args.train_dir = detected_split_dirs["train_dir"]
            auto_args.val_dir = detected_split_dirs["val_dir"]
            auto_args.test_dir = detected_split_dirs["test_dir"]
            return build_explicit_split_datasets(auto_args)
        raise ValueError(
            "Detected partial canonical split folders under --data-dir. "
            "Provide --train-dir/--val-dir/--test-dir explicitly to avoid data leakage."
        )

    rng = np.random.default_rng(args.seed)
    image_paths = collect_image_paths(args.data_dir)
    if args.real_file_limit > 0:
        image_paths = image_paths[: args.real_file_limit]

    image_paths, skipped_small = filter_paths_by_min_size(image_paths, args.block_size)
    if skipped_small > 0:
        print(
            "[*] Dropped "
            f"{skipped_small} images smaller than block-size={args.block_size} "
            "to avoid synthetic SR targets."
        )

    if len(image_paths) >= 3:
        train_idx, val_idx, test_idx = split_indices(len(image_paths), args.val_ratio, args.test_ratio, rng)
        train_samples = int(len(train_idx) * max(1, int(args.train_patches_per_image)))
        return (
            make_dataset_from_paths(
                image_paths[train_idx],
                args.block_size,
                args.batch_size,
                args.seed,
                shuffle=True,
                patches_per_image=args.train_patches_per_image,
                augment=args.real_augment,
                center_crop=False,
                upscale_factor=args.upscale_factor,
                degrade_interp=args.degrade_interp,
            ),
            make_dataset_from_paths(
                image_paths[val_idx],
                args.block_size,
                args.batch_size,
                args.seed,
                shuffle=False,
                patches_per_image=1,
                augment=False,
                center_crop=True,
                upscale_factor=args.upscale_factor,
                degrade_interp=args.degrade_interp,
            ),
            make_dataset_from_paths(
                image_paths[test_idx],
                args.block_size,
                args.batch_size,
                args.seed,
                shuffle=False,
                patches_per_image=1,
                augment=False,
                center_crop=True,
                upscale_factor=args.upscale_factor,
                degrade_interp=args.degrade_interp,
            ),
            {
                "source": "filesystem",
                "train": train_samples,
                "validation": int(len(val_idx)),
                "test": int(len(test_idx)),
                "train_images": int(len(train_idx)),
                "val_images": int(len(val_idx)),
                "test_images": int(len(test_idx)),
                "train_patches_per_image": int(max(1, int(args.train_patches_per_image))),
                "block_size": int(args.block_size),
                "upscale_factor": int(max(1, int(args.upscale_factor))),
                "degrade_interp": args.degrade_interp,
                "input_resolution": [int(max(8, args.block_size // max(1, args.upscale_factor))), int(max(8, args.block_size // max(1, args.upscale_factor)))],
                "target_resolution": [int(args.block_size), int(args.block_size)],
            },
        )

    if args.real_only:
        raise ValueError(
            f"No real image dataset found in '{args.data_dir}'. "
            "Add real image files or disable --real-only."
        )

    raise ValueError(
        f"No image dataset found in '{args.data_dir}'. "
        "Provide a directory with at least 3 real images, use explicit split directories, or enable --use-cifar10."
    )


def make_reconstruction_loss(mse_weight, l1_weight, ssim_weight, edge_weight, hard_weight=1.0, edge_power=2.0):
    """
    Creates a customized loss function featuring Edge/Sobel preservation penalties.
    Optimizes for both global pixel alignment and structural frequency sharp contrasts.
    """
    mse_w = float(mse_weight)
    l1_w = float(l1_weight)
    ssim_w = float(ssim_weight)
    edge_w = float(edge_weight)
    hard_w = max(0.0, float(hard_weight))
    edge_p = max(1.0, float(edge_power))

    total = max(mse_w + l1_w + ssim_w + edge_w, 1e-8)
    mse_w, l1_w, ssim_w, edge_w = mse_w / total, l1_w / total, ssim_w / total, edge_w / total

    def _loss(y_true, y_pred):
        error = y_true - y_pred
        abs_err = tf.abs(error)

        # Weight high-residual pixels more heavily while preserving differentiability.
        per_pixel_weight = 1.0 + hard_w * abs_err
        mse = tf.reduce_mean(tf.square(error) * per_pixel_weight)
        l1 = tf.reduce_mean(abs_err * per_pixel_weight)
        ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        # Calculate spatial image gradients to detect high-frequency edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)

        # Absolute difference in gradients enforces hard sharpening mapping
        edge_delta = tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred)
        edge_loss = tf.reduce_mean(tf.pow(edge_delta + 1e-6, edge_p))

        return mse_w * mse + l1_w * l1 + ssim_w * ssim_term + edge_w * edge_loss

    return _loss


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def build_autoencoder(latent_dim, model_base_filters=64, model_kernel_size=3, residual_scale=1.0):
    """
    High-capacity residual network optimized for fast edge reconstruction on M1 ARM architecture.
    Uses multi-stage residual processing channels to isolate edge variations without bottlenecks.
    """
    inputs = layers.Input(shape=(None, None, 3))
    base_filters = max(16, int(model_base_filters))
    kernel_size = max(1, int(model_kernel_size))
    deep_filters = max(base_filters * 2, base_filters + 8)

    # Shallow feature mapping
    x0 = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu")(inputs)
    x1 = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu")(x0)

    # Encoder stage 1
    x2 = layers.Conv2D(deep_filters, kernel_size, padding="same", activation="relu")(x1)

    # Latent dimension bottleneck compression
    bottleneck = layers.Conv2D(latent_dim, 1, padding="same", activation="relu", name="bottleneck")(x2)

    # Decoder mapping
    x3 = layers.Conv2D(deep_filters, kernel_size, padding="same", activation="relu")(bottleneck)

    # Skip-connection linking multi-resolution edge maps together
    x = layers.Add()([x3, x2])
    x = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Add()([x, x1])

    # Sharpening head isolates high-frequency delta maps
    residual = layers.Conv2D(3, kernel_size, padding="same", activation="tanh", name="residual_head")(x)

    # Output projection combines native lower-resolution inputs with isolated edge deltas
    residual_gain = max(0.0, float(residual_scale))
    outputs = layers.Lambda(lambda t: tf.clip_by_value(t[0] + residual_gain * t[1], 0.0, 1.0), name="reconstruction")([inputs, residual])

    return Model(inputs, outputs, name="remaster_autoencoder")


class TargetPSNRCallback(callbacks.Callback):
    def __init__(self, target_psnr):
        super().__init__()
        self.target_psnr = float(target_psnr)
        self.best_psnr = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_psnr_metric", logs.get("psnr_metric"))
        if current is None:
            return
        current_val = float(np.asarray(current).item())
        self.best_psnr = max(self.best_psnr, current_val)
        if current_val >= self.target_psnr:
            print(f"\n[+] Target PSNR reached at epoch {epoch + 1}: {current_val:.3f}")
            self.model.stop_training = True


class TargetOutputJpegRatioCallback(callbacks.Callback):
    def __init__(self, dataset, target_ratio, sample_count=32, jpeg_quality=90):
        super().__init__()
        self.dataset = dataset
        self.target_ratio = float(target_ratio)
        self.sample_count = max(1, int(sample_count))
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.best_ratio = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        size_metrics = compute_reconstruction_size_metrics(
            self.model,
            self.dataset,
            sample_count=self.sample_count,
            jpeg_quality=self.jpeg_quality,
        )
        ratio = size_metrics.get("output_jpeg_vs_target_jpeg_ratio")
        if ratio is None:
            return
        ratio_val = float(ratio)
        self.best_ratio = max(self.best_ratio, ratio_val)
        logs["val_output_jpeg_ratio"] = ratio_val
        print(f"\n[+] Epoch {epoch + 1} output/target JPEG(Q{self.jpeg_quality}) ratio: {ratio_val:.3f}")
        if ratio_val >= self.target_ratio:
            print(f"[+] Target output JPEG ratio reached at epoch {epoch + 1}: {ratio_val:.3f}")
            self.model.stop_training = True


def export_tflite_model(model, run_dir, use_fp16=False):
    """
    CRITICAL MAC M1 FIX: Converts directly from active memory using from_keras_model.
    Bypasses disk parsing that triggers the mlir::tf_saved_model::FreezeVariables SIGABRT crash.
    """
    os.makedirs(run_dir, exist_ok=True)
    tflite_path = os.path.join(run_dir, "production_model.tflite")

    print("[*] Initiating macOS ARM M1 memory-safe TFLite serialization pipeline...")

    # SAFE PATH: Convert directly in-memory to skip broken disk parsing loops
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Skip experimental compiler passes which cause the Constant op optimization error
    converter.experimental_new_converter = False

    if use_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return tflite_path


def save_preview_reconstructions(model, dataset, run_dir):
    preview_path = os.path.join(run_dir, "preview_reconstructions.png")
    os.makedirs(run_dir, exist_ok=True)

    for inputs, targets in dataset.take(1):
        n = int(min(4, inputs.shape[0]))
        preds = model.predict(inputs[:n], verbose=0)
        lowres_inputs = inputs[:n].numpy()
        originals = targets[:n].numpy()

        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        for i in range(n):
            axes[0, i].imshow(np.clip(lowres_inputs[i], 0.0, 1.0))
            axes[0, i].set_title("Input (degraded)")
            axes[0, i].axis("off")

            axes[1, i].imshow(np.clip(originals[i], 0.0, 1.0))
            axes[1, i].set_title("Target (original)")
            axes[1, i].axis("off")

            axes[2, i].imshow(np.clip(preds[i], 0.0, 1.0))
            axes[2, i].set_title("Decoded (Remastered)")
            axes[2, i].axis("off")

        for i in range(n, 4):
            axes[0, i].axis("off")
            axes[1, i].axis("off")
            axes[2, i].axis("off")

        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
        break

    return preview_path


def _plot_metric(ax, epochs, hist, train_key, val_key, title):
    train_vals = hist.get(train_key, [])
    val_vals = hist.get(val_key, [])
    plotted = False
    if train_vals:
        ax.plot(epochs[: len(train_vals)], train_vals, label="train")
        plotted = True
    if val_vals:
        ax.plot(epochs[: len(val_vals)], val_vals, label="val")
        plotted = True
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def plot_history(history, plot_path):
    hist = history.history
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)
    if len(epochs) == 0:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.98, "No epoch history recorded.\nCheck dataset/fit configuration.", va="top", ha="left")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_metric(axes[0], epochs, hist, "loss", "val_loss", "Loss")
    _plot_metric(axes[1], epochs, hist, "psnr_metric", "val_psnr_metric", "PSNR")
    _plot_metric(axes[2], epochs, hist, "ssim_metric", "val_ssim_metric", "SSIM")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def compute_reconstruction_size_metrics(model, dataset, sample_count=8, jpeg_quality=90):
    target_samples = max(1, int(sample_count))
    jpeg_quality = int(np.clip(jpeg_quality, 1, 100))

    in_png = []
    tgt_png = []
    pred_png = []
    tgt_jpg = []
    pred_jpg = []
    input_psnr_sum = 0.0
    output_psnr_sum = 0.0
    seen = 0

    for inputs, targets in dataset:
        batch_n = int(tf.shape(inputs)[0].numpy())
        if batch_n <= 0:
            continue
        take_n = min(target_samples - seen, batch_n)
        if take_n <= 0:
            break

        inp = inputs[:take_n]
        tgt = targets[:take_n]
        preds = model.predict(inp, verbose=0)

        input_psnr_sum += float(tf.reduce_sum(tf.image.psnr(inp, tgt, max_val=1.0)).numpy())
        output_psnr_sum += float(tf.reduce_sum(tf.image.psnr(tgt, preds, max_val=1.0)).numpy())

        for i in range(take_n):
            in_u8 = tf.cast(tf.round(tf.clip_by_value(inp[i], 0.0, 1.0) * 255.0), tf.uint8)
            tgt_u8 = tf.cast(tf.round(tf.clip_by_value(tgt[i], 0.0, 1.0) * 255.0), tf.uint8)
            pred_u8 = tf.cast(tf.round(tf.clip_by_value(preds[i], 0.0, 1.0) * 255.0), tf.uint8)
            in_png.append(int(tf.strings.length(tf.io.encode_png(in_u8)).numpy()))
            tgt_png.append(int(tf.strings.length(tf.io.encode_png(tgt_u8)).numpy()))
            pred_png.append(int(tf.strings.length(tf.io.encode_png(pred_u8)).numpy()))
            tgt_jpg.append(int(tf.strings.length(tf.io.encode_jpeg(tgt_u8, quality=jpeg_quality, optimize_size=True)).numpy()))
            pred_jpg.append(int(tf.strings.length(tf.io.encode_jpeg(pred_u8, quality=jpeg_quality, optimize_size=True)).numpy()))

        seen += take_n
        if seen >= target_samples:
            break

    if seen <= 0:
        return {"sample_count": 0}

    return {
        "sample_count": seen,
        "input_psnr_to_target": input_psnr_sum / float(seen),
        "output_psnr_to_target": output_psnr_sum / float(seen),
        "mean_input_png_bytes": float(np.mean(in_png)),
        "mean_target_png_bytes": float(np.mean(tgt_png)),
        "mean_output_png_bytes": float(np.mean(pred_png)),
        "mean_target_jpeg_q90_bytes": float(np.mean(tgt_jpg)),
        "mean_output_jpeg_q90_bytes": float(np.mean(pred_jpg)),
        "output_png_vs_target_png_ratio": float(np.mean(pred_png)) / max(float(np.mean(tgt_png)), 1.0),
        "output_jpeg_vs_target_jpeg_ratio": float(np.mean(pred_jpg)) / max(float(np.mean(tgt_jpg)), 1.0),
    }


def save_report(args, split_info, history, eval_values, run_dir, plot_path, preview_path, size_metrics, model, holdout_eval=None):
    os.makedirs(run_dir, exist_ok=True)
    artifact_sizes = {
        "training_metrics_png_bytes": int(os.path.getsize(plot_path)) if os.path.exists(plot_path) else 0,
        "preview_png_bytes": int(os.path.getsize(preview_path)) if os.path.exists(preview_path) else 0,
    }
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "split_info": split_info,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "plot_path": plot_path,
        "preview_path": preview_path,
        "model_diagnostics": {
            "parameter_count": int(model.count_params()),
            "input_shape": [None if x is None else int(x) for x in model.input_shape],
            "output_shape": [None if x is None else int(x) for x in model.output_shape],
        },
        "artifact_sizes": artifact_sizes,
        "size_metrics": size_metrics,
        "holdout_evaluation": holdout_eval,
    }
    report_path = os.path.join(run_dir, "evaluation_report.md")
    write_markdown_json_report(payload, report_path)
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def evaluate_holdout_if_requested(model, args):
    if not args.holdout_dir:
        return None

    holdout_paths = collect_image_paths(args.holdout_dir)
    if len(holdout_paths) == 0:
        return {
            "holdout_dir": args.holdout_dir,
            "status": "skipped",
            "reason": "no_images_found",
        }

    ds = make_dataset_from_paths(
        holdout_paths,
        args.block_size,
        args.batch_size,
        args.seed,
        shuffle=False,
        patches_per_image=1,
        augment=False,
        center_crop=True,
        upscale_factor=args.upscale_factor,
        degrade_interp=args.degrade_interp,
    )
    steps = max(1, math.ceil(len(holdout_paths) / args.batch_size))
    eval_values = model.evaluate(ds, steps=steps, return_dict=True, verbose=0)
    size_metrics = compute_reconstruction_size_metrics(model, ds, sample_count=8)
    return {
        "holdout_dir": args.holdout_dir,
        "status": "ok",
        "num_images": int(len(holdout_paths)),
        "metrics": {k: float(v) for k, v in eval_values.items()},
        "size_metrics": size_metrics,
    }


def main():
    args = parse_args_extended()
    args = apply_preset(args)
    if args.params_file:
        override_result = apply_overrides(args, load_overrides(args.params_file, section="image"))
        if override_result.applied:
            print(f"[*] Applied {len(override_result.applied)} params from {args.params_file}")
        if override_result.unknown:
            print(f"[!] Ignored unknown params in file: {sorted(override_result.unknown)}")
    args.grad_clipnorm = max(0.0, float(args.grad_clipnorm))
    args.residual_scale = max(0.0, float(args.residual_scale))
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    if args.real_only and args.target_psnr > 35.0:
        print("[!] Real-only run with high target PSNR; convergence may take longer.")
    configure_runtime(args.seed)

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, split_info = build_datasets(args)

    # Initialize our Upgraded Edge-Preserving Remaster Loss function
    loss_fn = make_reconstruction_loss(
        args.loss_mse_weight,
        args.loss_l1_weight,
        args.loss_ssim_weight,
        args.loss_edge_weight,
        hard_weight=args.loss_hard_weight,
        edge_power=args.loss_edge_power,
    )

    steps_per_epoch = max(1, math.ceil(split_info["train"] / args.batch_size))
    val_steps = max(1, math.ceil(split_info["validation"] / args.batch_size))
    test_steps = max(1, math.ceil(split_info["test"] / args.batch_size))

    model = build_autoencoder(
        args.latent_dim,
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
        residual_scale=args.residual_scale,
    )
    opt_kwargs = {"learning_rate": args.lr}
    if args.grad_clipnorm > 0:
        opt_kwargs["clipnorm"] = args.grad_clipnorm
    model.compile(
        optimizer=tf.keras.optimizers.Adam(**opt_kwargs),
        loss=loss_fn,
        metrics=["mse", psnr_metric, ssim_metric],
    )

    best_path = os.path.join(run_dir, "best_model.keras")
    cb = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=2, min_lr=1e-6, verbose=1),
    ]
    if not args.disable_target_psnr_stop:
        cb.append(TargetPSNRCallback(args.target_psnr))
    if not args.disable_target_output_jpeg_stop:
        cb.append(
            TargetOutputJpegRatioCallback(
                val_data,
                args.target_output_jpeg_ratio,
                sample_count=args.target_output_jpeg_samples,
                jpeg_quality=args.target_output_jpeg_quality,
            )
        )

    history = model.fit(
        train_data.repeat(),
        validation_data=val_data.repeat(),
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cb,
        verbose=1,
    )

    eval_values = model.evaluate(test_data, steps=test_steps, return_dict=True, verbose=0)
    plot_path = os.path.join(run_dir, "training_metrics.png")
    plot_history(history, plot_path)
    if not os.path.exists(plot_path) or os.path.getsize(plot_path) == 0:
        plot_history(history, plot_path)
    preview_path = save_preview_reconstructions(model, test_data, run_dir)
    size_metrics = compute_reconstruction_size_metrics(
        model,
        test_data,
        sample_count=8,
        jpeg_quality=args.target_output_jpeg_quality,
    )
    holdout_eval = evaluate_holdout_if_requested(model, args)
    report_path = save_report(args, split_info, history, eval_values, run_dir, plot_path, preview_path, size_metrics, model, holdout_eval=holdout_eval)

    weights_path = os.path.join(run_dir, "production_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        try:
            # Memory-safe local macOS ARM64 compilation
            tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)
        except Exception as exc:
            print(f"[!] TFLite export skipped: {exc}")

    print("\n[+] Remaster Training complete")
    print(f"[+] Preset: {args.preset}")
    print(f"[+] Data source: {split_info['source']}")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Best model: {best_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Plot: {plot_path}")
    print(f"[+] Preview: {preview_path}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    print(
        f"[+] Loss weights (mse/l1/ssim/edge): "
        f"{args.loss_mse_weight:.3f}/{args.loss_l1_weight:.3f}/{args.loss_ssim_weight:.3f}/{args.loss_edge_weight:.3f}"
    )
    print(f"[+] Loss profile: {args.loss_profile}")
    print(f"[+] Hard residual weight: {args.loss_hard_weight:.3f}")
    print(f"[+] Edge penalty power: {args.loss_edge_power:.3f}")
    print(f"[+] Target output JPEG ratio (Q{args.target_output_jpeg_quality}): {args.target_output_jpeg_ratio:.3f}")
    print(f"[+] Residual scale: {args.residual_scale:.3f}")
    print(f"[+] Gradient clipnorm: {args.grad_clipnorm:.3f}")
    if size_metrics.get("sample_count", 0) > 0:
        print(f"[+] Input->Target PSNR: {float(size_metrics.get('input_psnr_to_target', float('nan'))):.3f}")
        print(f"[+] Output->Target PSNR: {float(size_metrics.get('output_psnr_to_target', float('nan'))):.3f}")
        print(f"[+] Output PNG size ratio vs target PNG: {float(size_metrics.get('output_png_vs_target_png_ratio', float('nan'))):.3f}")
        out_jpeg_ratio = float(size_metrics.get('output_jpeg_vs_target_jpeg_ratio', float('nan')))
        print(
            f"[+] Output JPEG(Q{args.target_output_jpeg_quality}) size ratio vs target JPEG(Q{args.target_output_jpeg_quality}): "
            f"{out_jpeg_ratio:.3f}"
        )
        if out_jpeg_ratio < args.target_output_jpeg_ratio:
            print(
                f"[!] Output JPEG ratio target not reached yet "
                f"(target={args.target_output_jpeg_ratio:.3f}, got={out_jpeg_ratio:.3f})."
            )
    if holdout_eval and holdout_eval.get("status") == "ok":
        print(f"[+] Holdout dir: {holdout_eval.get('holdout_dir')}")
        print(f"[+] Holdout PSNR: {float(holdout_eval.get('metrics', {}).get('psnr_metric', float('nan'))):.3f}")
    elif holdout_eval and holdout_eval.get("status") != "ok":
        print(f"[!] Holdout evaluation skipped: {holdout_eval.get('reason', 'unknown')} ({holdout_eval.get('holdout_dir')})")
    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}). Increase epochs or use --preset m1-air-quality.")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
