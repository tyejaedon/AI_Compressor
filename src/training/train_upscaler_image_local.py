#!/usr/bin/env python3
"""Local trainer for a small learned super-resolution (upscaler) model.

Image Pipeline Roadmap Milestone 3: reuses the same degrade/upscale-factor
recipe as `train_autoencoder_image_local.py` (bicubic/area/bilinear downscale
+ additive Gaussian sensor noise) but, unlike that trainer, feeds the model a
genuinely lower-resolution input and asks it to reconstruct the full-resolution
patch -- i.e. real single-image super-resolution rather than same-size
denoising/deblurring. The trained model is consumed by
`upscale_reconstructed_images.py --upscaler-mode learned` as an alternative to
plain BICUBIC upscaling.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
from report_markdown import write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small learned super-resolution upscaler locally.")
    parser.add_argument(
        "--preset",
        type=str,
        default="m1-air-balanced",
        choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"],
        help="Hardware/runtime preset; use 'custom' to keep explicit flags unchanged",
    )
    parser.add_argument("--use-cifar10", action="store_true", help="Download/load CIFAR-10 and train from it")
    parser.add_argument("--cifar-train-limit", type=int, default=0, help="Optional cap for CIFAR-10 train samples")
    parser.add_argument("--cifar-test-limit", type=int, default=0, help="Optional cap for CIFAR-10 test samples")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with image files (recursive scan)")
    parser.add_argument("--train-dir", type=str, default="", help="Explicit train image directory")
    parser.add_argument("--val-dir", type=str, default="", help="Explicit validation image directory")
    parser.add_argument("--test-dir", type=str, default="", help="Explicit test image directory")
    parser.add_argument("--split-file-limit", type=int, default=0, help="Optional cap per explicit split directory")
    parser.add_argument("--real-only", action="store_true", help="Require real files from --data-dir/--train-dir and disable CIFAR fallback")
    parser.add_argument("--real-file-limit", type=int, default=0, help="Optional cap for discovered real images (0 means all)")
    parser.add_argument("--min-real-images", type=int, default=12, help="Minimum discovered real images required for real-only runs")
    parser.add_argument("--min-split-images", type=int, default=3, help="Minimum images required in each train/val/test split for real-only runs")
    parser.add_argument("--output-root", type=str, default="models/upscaler_local_run", help="Output directory for weights/reports")
    parser.add_argument("--params-file", type=str, default="", help="Optional JSON file with parameter overrides")
    parser.add_argument("--block-size", type=int, default=64, help="High-resolution target patch size (must be divisible by --upscale-factor)")
    parser.add_argument("--upscale-factor", type=int, default=2, choices=[2, 3, 4], help="Super-resolution factor (low-res input is block-size / factor)")
    parser.add_argument(
        "--degrade-interp",
        type=str,
        default="bicubic",
        choices=["area", "bilinear", "bicubic"],
        help="Interpolation used while downscaling high-res patches to synthesize low-res inputs",
    )
    parser.add_argument("--degrade-noise-std", type=float, default=0.015, help="Std-dev of additive Gaussian sensor noise on the low-res input")
    parser.add_argument("--model-base-filters", type=int, default=32, help="Base Conv2D filter count for the residual body")
    parser.add_argument("--model-kernel-size", type=int, default=3, help="Kernel size used in Conv2D blocks")
    parser.add_argument("--num-residual-blocks", type=int, default=4, help="Number of residual blocks in the upscaler body")
    parser.add_argument("--epochs", type=int, default=40, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--train-patches-per-image", type=int, default=4, help="Random crop samples drawn per real training image each epoch")
    parser.add_argument("--target-psnr", type=float, default=28.0, help="Training goal for validation PSNR")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of test samples to export in comparison panels")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export Android-ready .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable TFLite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Use float16 optimization when exporting TFLite")
    parser.set_defaults(export_tflite=True)
    return parser.parse_args()


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"block_size": 48, "model_base_filters": 24, "num_residual_blocks": 3, "batch_size": 8, "epochs": 20, "lr": 2e-4},
        "m1-air-balanced": {"block_size": 64, "model_base_filters": 32, "num_residual_blocks": 4, "batch_size": 8, "epochs": 40, "lr": 1.5e-4},
        "m1-air-quality": {"block_size": 96, "model_base_filters": 48, "num_residual_blocks": 6, "batch_size": 6, "epochs": 64, "lr": 1e-4},
    }
    if args.preset in preset_map:
        provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        key_to_flag = {
            "block_size": "--block-size",
            "model_base_filters": "--model-base-filters",
            "num_residual_blocks": "--num-residual-blocks",
            "batch_size": "--batch-size",
            "epochs": "--epochs",
            "lr": "--lr",
        }
        for key, value in preset_map[args.preset].items():
            if key_to_flag[key] not in provided_flags:
                setattr(args, key, value)

    args.upscale_factor = max(1, int(args.upscale_factor))
    if args.block_size % args.upscale_factor != 0:
        args.block_size = args.upscale_factor * max(1, round(args.block_size / args.upscale_factor))
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.num_residual_blocks = max(1, int(args.num_residual_blocks))
    args.min_real_images = max(3, int(args.min_real_images))
    args.min_split_images = max(1, int(args.min_split_images))
    args.sample_count = max(4, min(64, int(args.sample_count)))
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
    return idx[:train_n], idx[train_n : train_n + val_n], idx[train_n + val_n :]


_INTERP_MAP = {
    "area": tf.image.ResizeMethod.AREA,
    "bilinear": tf.image.ResizeMethod.BILINEAR,
    "bicubic": tf.image.ResizeMethod.BICUBIC,
}


def _degrade_to_lowres(high_res_patch, block_size, upscale_factor, degrade_interp, degrade_noise_std, seed=None):
    down_block = max(1, block_size // upscale_factor)
    degrade_method = _INTERP_MAP.get(str(degrade_interp).lower(), tf.image.ResizeMethod.BICUBIC)
    low_res = tf.image.resize(high_res_patch, [down_block, down_block], method=degrade_method)
    if degrade_noise_std > 0:
        noise = tf.random.normal(shape=tf.shape(low_res), mean=0.0, stddev=degrade_noise_std, seed=seed)
        low_res = tf.clip_by_value(low_res + noise, 0.0, 1.0)
    return tf.clip_by_value(low_res, 0.0, 1.0)


def make_sr_dataset_from_paths(
    paths,
    block_size,
    batch_size,
    seed,
    shuffle,
    upscale_factor,
    degrade_interp,
    degrade_noise_std,
    patches_per_image=1,
    augment=False,
    center_crop=False,
):
    block = int(block_size)
    patch_count = max(1, int(patches_per_image))

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
        low_res = _degrade_to_lowres(patch, block, upscale_factor, degrade_interp, degrade_noise_std, seed=seed)
        return low_res, patch

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


def make_sr_dataset_from_tensor(images, block_size, batch_size, seed, shuffle, upscale_factor, degrade_interp, degrade_noise_std):
    def _prep(image):
        image = tf.image.convert_image_dtype(image, tf.float32)
        # CIFAR-10 native resolution is 32x32; bicubic-upsample to block_size first so there is
        # a well-defined "high-res" target to degrade from (there is no higher-native-res source).
        high_res = tf.image.resize(image, [block_size, block_size], method=tf.image.ResizeMethod.BICUBIC)
        high_res = tf.clip_by_value(high_res, 0.0, 1.0)
        low_res = _degrade_to_lowres(high_res, block_size, upscale_factor, degrade_interp, degrade_noise_std, seed=seed)
        return low_res, high_res

    ds = tf.data.Dataset.from_tensor_slices(images)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cifar10_sr_datasets(args):
    print("[*] Downloading and preparing CIFAR-10 dataset (super-resolution pairs)...")
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

    common = dict(
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
        upscale_factor=args.upscale_factor,
        degrade_interp=args.degrade_interp,
        degrade_noise_std=args.degrade_noise_std,
    )
    return (
        make_sr_dataset_from_tensor(train_x, shuffle=True, **common),
        make_sr_dataset_from_tensor(val_x, shuffle=False, **common),
        make_sr_dataset_from_tensor(test_x, shuffle=False, **common),
        {
            "source": "cifar10",
            "note": "CIFAR-10 is natively 32x32; high-res targets are bicubic-upsampled to --block-size before degrading.",
            "train": int(len(train_x)),
            "validation": int(len(val_x)),
            "test": int(len(test_x)),
            "block_size": int(args.block_size),
            "upscale_factor": int(args.upscale_factor),
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

    common = dict(
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
        upscale_factor=args.upscale_factor,
        degrade_interp=args.degrade_interp,
        degrade_noise_std=args.degrade_noise_std,
    )
    return (
        make_sr_dataset_from_paths(train_paths, shuffle=True, patches_per_image=args.train_patches_per_image, augment=True, center_crop=False, **common),
        make_sr_dataset_from_paths(val_paths, shuffle=False, patches_per_image=1, augment=False, center_crop=True, **common),
        make_sr_dataset_from_paths(test_paths, shuffle=False, patches_per_image=1, augment=False, center_crop=True, **common),
        {
            "source": "explicit_directories",
            "train": int(len(train_paths) * max(1, int(args.train_patches_per_image))),
            "validation": int(len(val_paths)),
            "test": int(len(test_paths)),
            "train_images": int(len(train_paths)),
            "val_images": int(len(val_paths)),
            "test_images": int(len(test_paths)),
            "block_size": int(args.block_size),
            "upscale_factor": int(args.upscale_factor),
        },
    )


def build_datasets(args):
    if args.train_dir and args.val_dir and args.test_dir:
        return build_explicit_split_datasets(args)

    if args.data_dir:
        all_paths = collect_image_paths(args.data_dir)
        if args.real_file_limit > 0:
            all_paths = all_paths[: args.real_file_limit]
        if len(all_paths) >= args.min_real_images:
            rng = np.random.default_rng(args.seed)
            train_idx, val_idx, test_idx = split_indices(len(all_paths), args.val_ratio, args.test_ratio, rng)
            if min(len(train_idx), len(val_idx), len(test_idx)) >= args.min_split_images:
                common = dict(
                    block_size=args.block_size,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    upscale_factor=args.upscale_factor,
                    degrade_interp=args.degrade_interp,
                    degrade_noise_std=args.degrade_noise_std,
                )
                return (
                    make_sr_dataset_from_paths(
                        all_paths[train_idx], shuffle=True, patches_per_image=args.train_patches_per_image, augment=True, center_crop=False, **common
                    ),
                    make_sr_dataset_from_paths(all_paths[val_idx], shuffle=False, patches_per_image=1, augment=False, center_crop=True, **common),
                    make_sr_dataset_from_paths(all_paths[test_idx], shuffle=False, patches_per_image=1, augment=False, center_crop=True, **common),
                    {
                        "source": "filesystem",
                        "train": int(len(train_idx)),
                        "validation": int(len(val_idx)),
                        "test": int(len(test_idx)),
                        "block_size": int(args.block_size),
                        "upscale_factor": int(args.upscale_factor),
                    },
                )

    if args.real_only:
        raise ValueError(f"No real image dataset found in '{args.data_dir}'. Add real files or disable --real-only.")
    if args.use_cifar10:
        return build_cifar10_sr_datasets(args)

    raise ValueError(
        f"No image dataset found in '{args.data_dir}'. "
        "Provide a real image directory with enough files, explicit split dirs, or enable --use-cifar10."
    )


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def reconstruction_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.85 * mse + 0.15 * ssim_term


def _residual_block(x, filters, kernel_size):
    skip = x
    y = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    y = layers.Conv2D(filters, kernel_size, padding="same")(y)
    return layers.Add()([skip, y])


@tf.keras.utils.register_keras_serializable(package="ai_compressor_upscaler")
class PixelShuffle(layers.Layer):
    """Sub-pixel upsampling (`tf.nn.depth_to_space`) as a proper serializable Layer.

    A plain `layers.Lambda` cannot round-trip through `.keras` save/load in this
    Keras 3 version (its wrapped function/output_shape can't be deserialized
    standalone), which broke loading trained upscaler models in
    `upscale_reconstructed_images.py`. A registered custom Layer subclass with
    `get_config()`/`compute_output_shape()` saves and reloads correctly.
    """

    def __init__(self, upscale_factor, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = int(upscale_factor)

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        new_h = None if h is None else h * self.upscale_factor
        new_w = None if w is None else w * self.upscale_factor
        new_c = None if c is None else c // (self.upscale_factor ** 2)
        return (batch, new_h, new_w, new_c)

    def get_config(self):
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config


def build_upscaler(model_base_filters=32, model_kernel_size=3, num_residual_blocks=4, upscale_factor=2):
    """Small residual CNN + sub-pixel (depth-to-space) upsampling head.

    Accepts dynamic (None, None, 3) input so the same model can be applied to
    arbitrary-sized tiles at inference time (see `upscale_reconstructed_images.py`
    tiling support).
    """
    inputs = layers.Input(shape=(None, None, 3))
    base_filters = max(16, int(model_base_filters))
    kernel_size = max(1, int(model_kernel_size))
    up_factor = max(1, int(upscale_factor))

    x = layers.Conv2D(base_filters, kernel_size, padding="same", activation="relu", name="head_conv")(inputs)
    skip = x
    for _ in range(max(1, int(num_residual_blocks))):
        x = _residual_block(x, base_filters, kernel_size)
    x = layers.Conv2D(base_filters, kernel_size, padding="same", name="body_out")(x)
    x = layers.Add(name="global_residual")([x, skip])

    if up_factor > 1:
        x = layers.Conv2D(base_filters * (up_factor ** 2), kernel_size, padding="same", activation="relu", name="subpixel_conv")(x)
        x = PixelShuffle(up_factor, name="pixel_shuffle")(x)

    outputs = layers.Conv2D(3, kernel_size, padding="same", activation="sigmoid", name="output_rgb")(x)
    return Model(inputs, outputs, name="learned_upscaler")


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


def compute_bicubic_baseline(dataset, block_size):
    psnrs, ssims = [], []
    for low_res, high_res in dataset:
        bicubic_up = tf.image.resize(low_res, [block_size, block_size], method=tf.image.ResizeMethod.BICUBIC)
        bicubic_up = tf.clip_by_value(bicubic_up, 0.0, 1.0)
        psnrs.append(float(tf.reduce_mean(tf.image.psnr(high_res, bicubic_up, max_val=1.0)).numpy()))
        ssims.append(float(tf.reduce_mean(tf.image.ssim(high_res, bicubic_up, max_val=1.0)).numpy()))
    if not psnrs:
        return {"bicubic_psnr": None, "bicubic_ssim": None}
    return {"bicubic_psnr": float(np.mean(psnrs)), "bicubic_ssim": float(np.mean(ssims))}


def save_comparison_panel(model, dataset, run_dir, block_size, sample_count):
    panel_path = os.path.join(run_dir, "upscaler_comparison_panel.png")
    for low_res, high_res in dataset.take(1):
        n = int(min(sample_count, low_res.shape[0], 4))
        if n <= 0:
            return None
        low_res = low_res[:n]
        high_res_np = high_res[:n].numpy()
        bicubic_up = tf.clip_by_value(
            tf.image.resize(low_res, [block_size, block_size], method=tf.image.ResizeMethod.BICUBIC), 0.0, 1.0
        ).numpy()
        learned_up = np.clip(model.predict(low_res, verbose=0), 0.0, 1.0)
        low_res_np = low_res.numpy()

        fig, axes = plt.subplots(4, n, figsize=(3 * n, 12))
        if n == 1:
            axes = axes.reshape(4, 1)
        row_titles = ["Low-res input", "Bicubic upscaled", "Learned upscaled", "Ground truth"]
        rows = [low_res_np, bicubic_up, learned_up, high_res_np]
        for r in range(4):
            for c in range(n):
                axes[r, c].imshow(np.clip(rows[r][c], 0.0, 1.0))
                axes[r, c].axis("off")
                if c == 0:
                    axes[r, c].set_ylabel(row_titles[r])
            axes[r, 0].set_title(row_titles[r], loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)
        return panel_path
    return None


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


def export_tflite_model(model, run_dir, use_fp16=False):
    os.makedirs(run_dir, exist_ok=True)
    tflite_path = os.path.join(run_dir, "production_model.tflite")
    with tempfile.TemporaryDirectory(prefix="upscaler_tflite_export_") as tmp_dir:
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


def save_report(args, split_info, history, eval_values, baseline_stats, run_dir, plot_path, panel_path):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "split_info": split_info,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "bicubic_baseline": baseline_stats,
        "plot_path": plot_path,
        "comparison_panel_path": panel_path,
    }
    report_path = os.path.join(run_dir, "upscaler_evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Learned Upscaler Evaluation Report")
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
        override_result = apply_overrides(args, load_overrides(args.params_file, section="image_upscaler"))
        if override_result.applied:
            print(f"[*] Applied {len(override_result.applied)} params from {args.params_file}")
        if override_result.unknown:
            print(f"[!] Ignored unknown params in file: {sorted(override_result.unknown)}")
    configure_runtime(args.seed)

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, split_info = build_datasets(args)
    steps_per_epoch = max(1, math.ceil(split_info["train"] / args.batch_size))
    val_steps = max(1, math.ceil(split_info["validation"] / args.batch_size))
    test_steps = max(1, math.ceil(split_info["test"] / args.batch_size))

    model = build_upscaler(
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
        num_residual_blocks=args.num_residual_blocks,
        upscale_factor=args.upscale_factor,
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
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_delta=1e-4, cooldown=1, min_lr=1e-6, verbose=1),
        TargetPSNRCallback(args.target_psnr),
    ]

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
    baseline_stats = compute_bicubic_baseline(test_data.take(test_steps), args.block_size)
    plot_path = os.path.join(run_dir, "training_metrics.png")
    plot_history(history, plot_path)
    panel_path = save_comparison_panel(model, test_data, run_dir, args.block_size, args.sample_count)
    report_path = save_report(args, split_info, history, eval_values, baseline_stats, run_dir, plot_path, panel_path)

    weights_path = os.path.join(run_dir, "production_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        try:
            tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)
        except Exception as exc:
            print(f"[!] TFLite export skipped: {exc}")

    print("\n[+] Learned upscaler training complete")
    print(f"[+] Preset: {args.preset}")
    print(f"[+] Data source: {split_info['source']}")
    print(f"[+] Upscale factor: {args.upscale_factor}")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Best model: {best_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Plot: {plot_path}")
    if panel_path:
        print(f"[+] Comparison panel: {panel_path}")
    print(f"[+] Test PSNR (learned): {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    if baseline_stats.get("bicubic_psnr") is not None:
        print(f"[+] Test PSNR (bicubic baseline): {baseline_stats['bicubic_psnr']:.3f}")
        delta = float(eval_values.get("psnr_metric", 0.0)) - baseline_stats["bicubic_psnr"]
        print(f"[+] Learned vs bicubic PSNR delta: {delta:+.3f} dB")

    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(
            f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}). "
            "Increase epochs, filters, or residual blocks, or use --preset m1-air-quality."
        )

    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
