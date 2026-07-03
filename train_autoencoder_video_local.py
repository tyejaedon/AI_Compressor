#!/usr/bin/env python3
"""Local video autoencoder trainer with MP4 benchmark and TFLite export."""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers

from report_markdown import write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train a video autoencoder locally.")
    parser.add_argument("--output-root", type=str, default="models/video_local_run", help="Output directory for artifacts")
    parser.add_argument("--preset", type=str, default="m1-air-balanced", choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"])
    parser.add_argument("--frames", type=int, default=8, help="Frames per clip")
    parser.add_argument("--height", type=int, default=64, help="Frame height")
    parser.add_argument("--width", type=int, default=64, help="Frame width")
    parser.add_argument("--fps", type=int, default=12, help="Frame rate")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent bottleneck size")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=24, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/VIDEO DATA", help="Directory with real video files")
    parser.add_argument("--real-val-ratio", type=float, default=0.15, help="Validation split ratio for real clips")
    parser.add_argument("--real-test-ratio", type=float, default=0.15, help="Test split ratio for real clips")
    parser.add_argument("--real-max-videos", type=int, default=0, help="Optional cap for discovered real videos (0 = all)")
    parser.add_argument("--real-max-clips", type=int, default=1800, help="Cap for extracted real clips to keep training practical")
    parser.add_argument("--real-clip-stride", type=int, default=4, help="Frame stride between extracted clips")
    parser.add_argument("--synthetic-only", action="store_true", help="Disable real-data loading and force synthetic generation")
    parser.add_argument("--train-samples", type=int, default=8000, help="Synthetic train sample count")
    parser.add_argument("--val-samples", type=int, default=1200, help="Synthetic validation sample count")
    parser.add_argument("--test-samples", type=int, default=1200, help="Synthetic test sample count")
    parser.add_argument("--target-psnr", type=float, default=25.0, help="Validation PSNR target")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable .tflite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Export float16 optimized tflite")
    parser.add_argument("--run-mp4-benchmark", action="store_true", help="Benchmark recon quality against MP4")
    parser.set_defaults(export_tflite=True)
    return parser.parse_args()


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"batch_size": 4, "latent_dim": 224, "epochs": 14, "lr": 1.8e-4},
        "m1-air-balanced": {"batch_size": 4, "latent_dim": 288, "epochs": 24, "lr": 1.2e-4},
        "m1-air-quality": {"batch_size": 3, "latent_dim": 384, "epochs": 36, "lr": 8e-5},
    }
    if args.preset in preset_map:
        for k, v in preset_map[args.preset].items():
            setattr(args, k, v)
    return args


def synth_video_clip(frames, height, width, rng):
    clip = np.zeros((frames, height, width, 3), dtype=np.float32)

    base_color = rng.uniform(0.15, 0.85, size=(3,)).astype(np.float32)
    obj_color = rng.uniform(0.2, 1.0, size=(3,)).astype(np.float32)

    grid_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    grid_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)

    start_x = rng.integers(0, max(1, width // 3))
    start_y = rng.integers(0, max(1, height // 3))
    vel_x = rng.choice([-2, -1, 1, 2])
    vel_y = rng.choice([-2, -1, 1, 2])
    box_w = rng.integers(max(6, width // 8), max(8, width // 3))
    box_h = rng.integers(max(6, height // 8), max(8, height // 3))

    for t in range(frames):
        phase = t / max(frames - 1, 1)
        bg = np.stack(
            [
                base_color[0] * (0.6 + 0.4 * xx),
                base_color[1] * (0.6 + 0.4 * yy),
                base_color[2] * (0.7 + 0.3 * np.sin((xx + yy + phase) * np.pi)),
            ],
            axis=-1,
        )

        x0 = int(np.clip(start_x + vel_x * t, 0, width - box_w))
        y0 = int(np.clip(start_y + vel_y * t, 0, height - box_h))

        frame = bg
        frame[y0 : y0 + box_h, x0 : x0 + box_w, :] = obj_color * (0.7 + 0.3 * phase)

        noise = rng.normal(0.0, 0.01, size=(height, width, 3)).astype(np.float32)
        clip[t] = np.clip(frame + noise, 0.0, 1.0)

    return clip.astype(np.float32)


def build_dataset(frames, height, width, sample_count, batch_size, seed, shuffle, repeat=False):
    rng = np.random.default_rng(seed)

    def gen():
        for _ in range(sample_count):
            clip = synth_video_clip(frames, height, width, rng)
            yield clip, clip

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(frames, height, width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(frames, height, width, 3), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(sample_count, 2048), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        # Keep input stream available across epochs when fit/evaluate uses explicit step counts.
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def collect_video_paths(root_dir):
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    root = Path(root_dir)
    if not root.exists():
        return np.array([], dtype=np.str_)
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return np.array(sorted(paths), dtype=np.str_)


def decode_video_frames(ffmpeg_bin, in_path, width, height, fps):
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(out, dtype=np.uint8)
    frame_size = width * height * 3
    frame_count = len(arr) // frame_size
    if frame_count <= 0:
        return np.empty((0, height, width, 3), dtype=np.float32)
    arr = arr[: frame_count * frame_size]
    frames_np = arr.reshape(frame_count, height, width, 3).astype(np.float32) / 255.0
    return frames_np


def resolve_ffmpeg_binary():
    ffmpeg_bin = shutil.which("ffmpeg")
    if isinstance(ffmpeg_bin, str) and ffmpeg_bin:
        return ffmpeg_bin

    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if isinstance(ffmpeg_bin, str) and ffmpeg_bin:
            return ffmpeg_bin
    except Exception:
        pass

    return None


def split_indices(n, val_ratio, test_ratio, rng):
    if n < 3:
        raise ValueError("Need at least 3 real clips for train/val/test splits")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("real-val-ratio and real-test-ratio must be >=0 and sum to < 1")

    idx = rng.permutation(n)
    test_n = max(1, int(n * test_ratio))
    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Not enough real clips for requested split ratios")
    return idx[:train_n], idx[train_n : train_n + val_n], idx[train_n + val_n :]


def extract_real_video_clips(args):
    ffmpeg_bin = resolve_ffmpeg_binary()
    if not isinstance(ffmpeg_bin, str) or not ffmpeg_bin:
        print("[!] ffmpeg not found (system or imageio-ffmpeg); skipping real video loading.")
        return None

    video_paths = collect_video_paths(args.data_dir)
    if args.real_max_videos > 0:
        video_paths = video_paths[: args.real_max_videos]
    if len(video_paths) == 0:
        return None

    clips = []
    stride = max(1, int(args.real_clip_stride))
    for path in video_paths:
        try:
            frames_np = decode_video_frames(ffmpeg_bin, path, args.width, args.height, args.fps)
        except Exception:
            continue
        if len(frames_np) < args.frames:
            continue
        for start in range(0, len(frames_np) - args.frames + 1, stride):
            clips.append(frames_np[start : start + args.frames])
            if len(clips) >= args.real_max_clips:
                break
        if len(clips) >= args.real_max_clips:
            break

    if len(clips) < 3:
        return None
    return np.asarray(clips, dtype=np.float32)


def make_dataset_from_clips(clips, batch_size, seed, shuffle, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices((clips, clips))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(clips), 2048), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_real_video_datasets(args):
    clips = extract_real_video_clips(args)
    if clips is None:
        return None

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = split_indices(len(clips), args.real_val_ratio, args.real_test_ratio, rng)
    train_clips = clips[train_idx]
    val_clips = clips[val_idx]
    test_clips = clips[test_idx]

    train_data = make_dataset_from_clips(train_clips, args.batch_size, args.seed, shuffle=True, repeat=True)
    val_data = make_dataset_from_clips(val_clips, args.batch_size, args.seed + 1, shuffle=False, repeat=True)
    test_data = make_dataset_from_clips(test_clips, args.batch_size, args.seed + 2, shuffle=False, repeat=True)
    counts = {"train": int(len(train_clips)), "val": int(len(val_clips)), "test": int(len(test_clips)), "source": "real_filesystem"}
    return train_data, val_data, test_data, counts


def video_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

    y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
    y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
    ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0))

    return 0.75 * mse + 0.15 * l1 + 0.10 * ssim_term


def psnr_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
    y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
    return tf.reduce_mean(tf.image.psnr(y_true_flat, y_pred_flat, max_val=1.0))


def ssim_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
    y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
    return tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0))


def build_video_autoencoder(frames, height, width, latent_dim):
    inputs = layers.Input(shape=(frames, height, width, 3))

    e1 = layers.Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding="same", activation="relu")(inputs)
    e1 = layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(e1)

    e2 = layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu")(e1)
    e2 = layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu")(e2)

    b = layers.Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding="same", activation="relu")(e2)
    b = layers.Conv3D(128, (3, 3, 3), padding="same", activation="relu")(b)

    x = layers.GlobalAveragePooling3D()(b)
    latent = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    t_down = max(1, frames // 2)
    h_down = max(1, height // 8)
    w_down = max(1, width // 8)

    x = layers.Dense(t_down * h_down * w_down * 128, activation="relu")(latent)
    x = layers.Reshape((t_down, h_down, w_down, 128))(x)

    x = layers.UpSampling3D(size=(1, 2, 2))(x)
    x = layers.Concatenate()([x, e2])
    x = layers.Conv3D(128, (3, 3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = layers.Concatenate()([x, e1])
    x = layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling3D(size=(1, 2, 2))(x)
    x = layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(x)
    outputs = layers.Conv3D(3, (3, 3, 3), padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="video_autoencoder")


class TargetPSNRCallback(callbacks.Callback):
    def __init__(self, target_psnr):
        super().__init__()
        self.target_psnr = float(target_psnr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_psnr_metric")
        if current is None:
            return
        value = float(np.asarray(current, dtype=np.float32))
        if value >= self.target_psnr:
            print(f"\n[+] Target PSNR reached at epoch {epoch + 1}: {value:.3f}")
            self.model.stop_training = True


def export_tflite_model(model, run_dir, use_fp16=False):
    tflite_path = os.path.join(run_dir, "video_autoencoder.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if use_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def save_video_preview(model, test_data, run_dir):
    preview_path = os.path.join(run_dir, "video_preview_reconstruction.png")

    for x, _ in test_data.take(1):
        pred = model.predict(x[:1], verbose=0)[0]
        src = x[0].numpy()

        show_frames = min(4, src.shape[0])
        fig, axes = plt.subplots(2, show_frames, figsize=(3 * show_frames, 6))
        if show_frames == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for i in range(show_frames):
            axes[0, i].imshow(np.clip(src[i], 0.0, 1.0))
            axes[0, i].set_title(f"Original t={i}")
            axes[0, i].axis("off")
            axes[1, i].imshow(np.clip(pred[i], 0.0, 1.0))
            axes[1, i].set_title(f"Decoded t={i}")
            axes[1, i].axis("off")

        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
        break

    return preview_path


def compute_psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def encode_mp4(ffmpeg, frames, fps, out_path):
    t, h, w, _ = frames.shape
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, input=(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes(), check=True)


def decode_mp4(ffmpeg, in_path, width, height):
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(out, dtype=np.uint8)
    frame_size = width * height * 3
    frame_count = len(arr) // frame_size
    arr = arr[: frame_count * frame_size]
    return arr.reshape(frame_count, height, width, 3).astype(np.float32) / 255.0


def run_mp4_benchmark(model, test_data, run_dir, fps):
    ffmpeg = shutil.which("ffmpeg")
    report_path = os.path.join(run_dir, "video_mp4_benchmark.md")
    if not isinstance(ffmpeg, str) or not ffmpeg:
        payload = {"status": "skipped", "reason": "ffmpeg_not_found"}
        write_markdown_json_report(payload, report_path, title="Video MP4 Benchmark")
        return report_path

    clips = []
    for x, _ in test_data.take(3):
        clips.extend([x[i].numpy() for i in range(min(2, x.shape[0]))])

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, clip in enumerate(clips):
            recon = model.predict(clip[np.newaxis, ...], verbose=0)[0]
            model_psnr = compute_psnr(np.clip(clip, 0.0, 1.0), np.clip(recon, 0.0, 1.0))

            for crf in [20, 24, 28]:
                mp4_path = os.path.join(tmp, f"clip_{idx}_crf{crf}.mp4")
                t, h, w, _ = clip.shape

                cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    str(crf),
                    mp4_path,
                ]
                subprocess.run(cmd, input=(np.clip(clip, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes(), check=True)

                decoded = np.asarray(decode_mp4(ffmpeg, mp4_path, w, h), dtype=np.float32)
                n = min(len(decoded), len(clip))
                mp4_psnr = compute_psnr(np.clip(clip[:n], 0.0, 1.0), np.clip(decoded[:n], 0.0, 1.0))

                raw_bytes = int(clip.size)
                size_ratio = os.path.getsize(mp4_path) / max(raw_bytes, 1)

                results.append(
                    {
                        "clip_idx": idx,
                        "crf": crf,
                        "model_psnr": model_psnr,
                        "mp4_psnr": mp4_psnr,
                        "mp4_size_ratio": size_ratio,
                    }
                )

    payload = {"status": "ok", "results": results}
    write_markdown_json_report(payload, report_path, title="Video MP4 Benchmark")
    return report_path


def save_report(args, history, eval_values, run_dir, preview_path, benchmark_path):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "preview": preview_path,
        "benchmark": benchmark_path,
    }
    report_path = os.path.join(run_dir, "video_evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Video Evaluation Report")
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)


def main():
    args = apply_preset(parse_args())
    configure_runtime(args.seed)

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8")
    if args.frames % 2 != 0:
        raise ValueError("frames must be divisible by 2")

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    real_data = None if args.synthetic_only else build_real_video_datasets(args)
    if real_data is not None:
        train_data, val_data, test_data, counts = real_data
        print(f"[*] Using real video clips from '{args.data_dir}' (train={counts['train']}, val={counts['val']}, test={counts['test']})")
        steps_per_epoch = max(1, math.ceil(counts["train"] / args.batch_size))
        val_steps = max(1, math.ceil(counts["val"] / args.batch_size))
        test_steps = max(1, math.ceil(counts["test"] / args.batch_size))
    else:
        print("[!] Real video clips not found or disabled; falling back to synthetic generation.")
        train_data = build_dataset(args.frames, args.height, args.width, args.train_samples, args.batch_size, args.seed, shuffle=True, repeat=True)
        val_data = build_dataset(args.frames, args.height, args.width, args.val_samples, args.batch_size, args.seed + 1, shuffle=False, repeat=True)
        test_data = build_dataset(args.frames, args.height, args.width, args.test_samples, args.batch_size, args.seed + 2, shuffle=False, repeat=True)
        steps_per_epoch = max(1, math.ceil(args.train_samples / args.batch_size))
        val_steps = max(1, math.ceil(args.val_samples / args.batch_size))
        test_steps = max(1, math.ceil(args.test_samples / args.batch_size))

    model = build_video_autoencoder(args.frames, args.height, args.width, args.latent_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=video_loss,
        metrics=["mse", psnr_metric, ssim_metric],
    )

    best_path = os.path.join(run_dir, "best_video_model.keras")
    cbs = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=1e-6, verbose=1),
        TargetPSNRCallback(args.target_psnr),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cbs,
        verbose=1,
    )

    eval_values = model.evaluate(test_data, steps=test_steps, return_dict=True, verbose=0)
    preview_path = save_video_preview(model, test_data, run_dir)

    benchmark_path = None
    if args.run_mp4_benchmark:
        benchmark_path = run_mp4_benchmark(model, test_data, run_dir, args.fps)

    weights_path = os.path.join(run_dir, "video_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)

    report_path = save_report(args, history, eval_values, run_dir, preview_path, benchmark_path)

    print("\n[+] Video training complete")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Preview: {preview_path}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}).")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")
    if benchmark_path:
        print(f"[+] MP4 benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()


