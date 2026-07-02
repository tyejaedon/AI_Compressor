#!/usr/bin/env python3
"""Local MacBook-friendly image autoencoder trainer."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers


def parse_args():
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
    parser.add_argument("--valid-dir", type=str, default="data/cifar10_valid", help="Local directory placeholder for CIFAR-10 validation assets")
    parser.add_argument("--output-root", type=str, default="models/local_run", help="Output directory for weights/reports")
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Training patch size used for CIFAR/synthetic tensor inputs",
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent bottleneck size")
    parser.add_argument("--epochs", type=int, default=24, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--allow-synthetic", action="store_true", help="Use synthetic images if data-dir is empty")
    parser.add_argument("--dummy-samples", type=int, default=4000, help="Synthetic sample count")
    parser.add_argument(
        "--extensive-dummy-samples",
        type=int,
        default=4000,
        help="Synthetic sample count for automatic extensive fallback generation",
    )
    parser.add_argument(
        "--no-auto-generate-dummy",
        dest="auto_generate_dummy",
        action="store_false",
        help="Disable automatic extensive dummy generation when data-dir is empty",
    )
    parser.add_argument("--cifar-train-limit", type=int, default=0, help="Optional cap for CIFAR-10 train samples (0 means all)")
    parser.add_argument("--cifar-test-limit", type=int, default=0, help="Optional cap for CIFAR-10 test samples (0 means all)")
    parser.add_argument("--target-psnr", type=float, default=25.0, help="Training goal for validation PSNR")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export Android-ready .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable TFLite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Use float16 optimization when exporting TFLite")
    parser.set_defaults(auto_generate_dummy=True, export_tflite=True)
    return parser.parse_args()


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"block_size": 96, "latent_dim": 64, "batch_size": 4, "epochs": 12, "lr": 2.5e-4},
        "m1-air-balanced": {"block_size": 128, "latent_dim": 96, "batch_size": 4, "epochs": 24, "lr": 1.8e-4},
        "m1-air-quality": {"block_size": 128, "latent_dim": 128, "batch_size": 4, "epochs": 36, "lr": 1.2e-4},
    }

    if args.preset in preset_map:
        # Keep preset as defaults, but do not overwrite explicit CLI flags.
        provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
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


def make_dataset_from_paths(paths, batch_size, seed, shuffle):
    def _decode(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32) / 255.0
        return image, image

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.padded_batch(
        batch_size,
        padded_shapes=([None, None, 3], [None, None, 3]),
        padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)),
    ).prefetch(tf.data.AUTOTUNE)


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


def create_extensive_dummy_images(num_samples, seed):
    """Create a large, diverse 32x32 uint8 synthetic corpus for local training fallback."""
    if num_samples < 3:
        raise ValueError("extensive-dummy-samples must be at least 3")

    rng = np.random.default_rng(seed)
    images = np.empty((num_samples, 32, 32, 3), dtype=np.uint8)

    # Pattern bank introduces structure and natural gradients for visible, realistic reconstructions.
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    gradient_x = np.stack([xx, xx**0.7, xx**1.3], axis=-1)
    gradient_y = np.stack([yy**1.1, yy, yy**0.8], axis=-1)
    radial = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    radial = np.stack([1.0 - radial, 1.0 - radial**0.6, 1.0 - radial**1.4], axis=-1)
    checker = (((np.floor(xx * 6) + np.floor(yy * 6)) % 2) * 1.0).astype(np.float32)
    checker = np.stack([checker * 0.7 + 0.15, 1.0 - checker * 0.5, checker * 0.4 + 0.3], axis=-1)
    soft_wave = 0.5 + 0.5 * np.sin((xx * 6.0 + yy * 7.0) * np.pi)
    soft_wave = np.stack([soft_wave, np.roll(soft_wave, 3, axis=0), np.roll(soft_wave, 3, axis=1)], axis=-1)

    pattern_bank = np.stack([gradient_x, gradient_y, np.clip(radial, 0.0, 1.0), checker, soft_wave], axis=0)

    chunk = 2000
    for start in range(0, num_samples, chunk):
        end = min(start + chunk, num_samples)
        n = end - start

        bank_indices = rng.integers(0, len(pattern_bank), size=n)
        base = pattern_bank[bank_indices].astype(np.float32)

        color_shift = rng.uniform(0.85, 1.15, size=(n, 1, 1, 3)).astype(np.float32)
        gaussian_noise = rng.normal(0.0, 0.05, size=(n, 32, 32, 3)).astype(np.float32)
        uniform_mix = rng.uniform(0.0, 1.0, size=(n, 32, 32, 3)).astype(np.float32)

        blended = 0.80 * base * color_shift + 0.17 * uniform_mix + 0.03 * gaussian_noise
        images[start:end] = np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)

    return images


def build_cifar10_datasets(args):
    print("[*] Downloading and preparing CIFAR-10 dataset...")

    # Keep parity with your preferred local setup flow.
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


def build_datasets(args):
    if args.use_cifar10:
        return build_cifar10_datasets(args)

    rng = np.random.default_rng(args.seed)
    image_paths = collect_image_paths(args.data_dir)

    if len(image_paths) >= 3:
        train_idx, val_idx, test_idx = split_indices(len(image_paths), args.val_ratio, args.test_ratio, rng)
        return (
            make_dataset_from_paths(image_paths[train_idx], args.batch_size, args.seed, shuffle=True),
            make_dataset_from_paths(image_paths[val_idx], args.batch_size, args.seed, shuffle=False),
            make_dataset_from_paths(image_paths[test_idx], args.batch_size, args.seed, shuffle=False),
            {
                "source": "filesystem",
                "train": int(len(train_idx)),
                "validation": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        )

    if not args.allow_synthetic and not args.auto_generate_dummy:
        raise ValueError(
            f"No image dataset found in '{args.data_dir}'. "
            "Put images there, use --allow-synthetic, or keep automatic extensive dummy fallback enabled."
        )

    synthetic_samples = args.dummy_samples
    if args.auto_generate_dummy and not args.allow_synthetic:
        synthetic_samples = max(args.extensive_dummy_samples, 3)
        print(
            f"[!] No images found in '{args.data_dir}'. "
            f"Auto-generating extensive dummy dataset with {synthetic_samples} samples."
        )
        synthetic = create_extensive_dummy_images(synthetic_samples, args.seed)
    else:
        print(f"[!] Using synthetic dataset fallback with {synthetic_samples} samples.")
        synthetic = rng.integers(0, 256, size=(synthetic_samples, 32, 32, 3), dtype=np.uint8)

    train_idx, val_idx, test_idx = split_indices(len(synthetic), args.val_ratio, args.test_ratio, rng)
    return (
        make_dataset_from_tensor(synthetic[train_idx], args.block_size, args.batch_size, args.seed, shuffle=True),
        make_dataset_from_tensor(synthetic[val_idx], args.block_size, args.batch_size, args.seed, shuffle=False),
        make_dataset_from_tensor(synthetic[test_idx], args.block_size, args.batch_size, args.seed, shuffle=False),
        {
            "source": "synthetic",
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    )


def reconstruction_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.85 * mse + 0.10 * l1 + 0.05 * ssim_term


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def build_autoencoder(latent_dim):
    inputs = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # Spatially-dynamic bottleneck keeps native input resolution end-to-end.
    x = layers.Conv2D(latent_dim, 1, padding="same", activation="relu", name="bottleneck")(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="local_autoencoder")


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
        current_val = float(current)
        self.best_psnr = max(self.best_psnr, current_val)
        if current_val >= self.target_psnr:
            print(f"\n[+] Target PSNR reached at epoch {epoch + 1}: {current_val:.3f}")
            self.model.stop_training = True


def export_tflite_model(model, run_dir, use_fp16=False):
    tflite_path = os.path.join(run_dir, "production_model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def save_preview_reconstructions(model, dataset, run_dir):
    preview_path = os.path.join(run_dir, "preview_reconstructions.png")

    for images, _ in dataset.take(1):
        preds = model.predict(images[:4], verbose=0)
        originals = images[:4].numpy()

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(4):
            axes[0, i].imshow(np.clip(originals[i], 0.0, 1.0))
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(np.clip(preds[i], 0.0, 1.0))
            axes[1, i].set_title("Decoded")
            axes[1, i].axis("off")

        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
        break

    return preview_path


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


def save_report(args, split_info, history, eval_values, run_dir, plot_path):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "split_info": split_info,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "plot_path": plot_path,
    }
    report_path = os.path.join(run_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
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
    configure_runtime(args.seed)

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, split_info = build_datasets(args)
    steps_per_epoch = max(1, math.ceil(split_info["train"] / args.batch_size))
    val_steps = max(1, math.ceil(split_info["validation"] / args.batch_size))
    test_steps = max(1, math.ceil(split_info["test"] / args.batch_size))

    model = build_autoencoder(args.latent_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=reconstruction_loss,
        metrics=["mse", psnr_metric, ssim_metric],
    )

    best_path = os.path.join(run_dir, "best_model.keras")
    cb = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=2, min_lr=1e-6, verbose=1),
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
    plot_path = os.path.join(run_dir, "training_metrics.png")
    plot_history(history, plot_path)
    preview_path = save_preview_reconstructions(model, test_data, run_dir)
    report_path = save_report(args, split_info, history, eval_values, run_dir, plot_path)

    weights_path = os.path.join(run_dir, "production_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)

    print("\n[+] Training complete")
    print(f"[+] Preset: {args.preset}")
    print(f"[+] Data source: {split_info['source']}")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Best model: {best_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Plot: {plot_path}")
    print(f"[+] Preview: {preview_path}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}). Increase epochs or use --preset m1-air-quality.")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
