#!/usr/bin/env python3
"""Local video autoencoder trainer with real quantization, temporal modeling, MP4/VP9 benchmark, and TFLite export."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, regularizers

from param_overrides import apply_overrides, load_overrides

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
from report_markdown import write_markdown_json_report


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@tf.keras.utils.register_keras_serializable(package="ai_compressor_video")
class VideoStraightThroughQuantize(layers.Layer):
    """Quantize video latent tensor to 2**bit_depth discrete levels with straight-through gradients."""

    def __init__(self, bit_depth=8, quant_noise_anneal=False, **kwargs):
        super().__init__(**kwargs)
        self.bit_depth = int(bit_depth)
        self.levels = float((1 << self.bit_depth) - 1) if self.bit_depth > 0 else 0.0
        self.quant_noise_anneal = bool(quant_noise_anneal)
        self.noise_scale = None
        if self.quant_noise_anneal and self.bit_depth > 0:
            self.noise_scale = self.add_weight(
                name="quant_noise_scale",
                shape=(),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=False,
            )

    def call(self, inputs, training=None):
        if self.bit_depth <= 0:
            return inputs
        quantized = tf.round(inputs * self.levels) / self.levels
        ste = inputs + tf.stop_gradient(quantized - inputs)
        if not self.quant_noise_anneal or not training or self.noise_scale is None:
            return ste
        step = 1.0 / self.levels
        noise = tf.random.uniform(tf.shape(inputs), -0.5, 0.5, dtype=inputs.dtype) * step
        soft_quant = tf.clip_by_value(inputs + noise, 0.0, 1.0)
        alpha = tf.cast(self.noise_scale, inputs.dtype)
        return alpha * soft_quant + (1.0 - alpha) * ste

    def get_config(self):
        config = super().get_config()
        config.update({"bit_depth": self.bit_depth, "quant_noise_anneal": self.quant_noise_anneal})
        return config


class VideoQuantNoiseAnnealCallback(callbacks.Callback):
    """Linearly fades VideoStraightThroughQuantize noise_scale from 1 -> 0."""

    def __init__(self, layer, total_epochs):
        super().__init__()
        self.layer = layer
        self.total_epochs = max(1, int(total_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if self.layer.noise_scale is None:
            return
        denom = max(1, self.total_epochs - 1) if self.total_epochs > 1 else 1
        frac = max(0.0, 1.0 - (epoch / denom))
        self.layer.noise_scale.assign(frac)


@tf.keras.utils.register_keras_serializable(package="ai_compressor_video")
class VideoEntropyRatePenalty(layers.Layer):
    """Per-channel differentiable soft-histogram Shannon entropy penalty for 3D video latents."""

    def __init__(self, weight, num_bins=32, bandwidth=None, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.num_bins = int(num_bins)
        self.bandwidth = float(bandwidth) if bandwidth is not None else 1.0 / self.num_bins

    def call(self, inputs):
        if self.weight <= 0.0:
            return inputs
        bin_centers = tf.linspace(0.0, 1.0, self.num_bins)
        bin_centers = tf.reshape(bin_centers, [1] * len(inputs.shape) + [self.num_bins])
        values = tf.expand_dims(tf.clip_by_value(inputs, 0.0, 1.0), axis=-1)
        sq_dist = tf.square(values - bin_centers)
        weights_unnorm = tf.exp(-0.5 * sq_dist / (self.bandwidth ** 2))
        weights_norm = weights_unnorm / (tf.reduce_sum(weights_unnorm, axis=-1, keepdims=True) + 1e-12)

        rank = len(weights_norm.shape)
        reduce_axes = list(range(max(1, rank - 2)))
        probs = tf.reduce_mean(weights_norm, axis=reduce_axes)
        probs = probs / (tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-12)

        per_channel_entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-12) / tf.math.log(2.0), axis=-1)
        self.add_loss(self.weight * tf.reduce_mean(per_channel_entropy))
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight, "num_bins": self.num_bins, "bandwidth": self.bandwidth})
        return config


@tf.keras.utils.register_keras_serializable(package="ai_compressor_video")
class TemporalDeltaInput(layers.Layer):
    """Converts a video clip into [frame_0, frame_1 - frame_0, ..., frame_{T-1} - frame_{T-2}]
    so the encoder directly models inter-frame motion residuals instead of redundant static backgrounds."""

    def call(self, inputs):
        first = inputs[:, :1, :, :, :]
        deltas = inputs[:, 1:, :, :, :] - inputs[:, :-1, :, :, :]
        return tf.concat([first, deltas], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a video autoencoder locally.")
    parser.add_argument("--output-root", type=str, default="models/video_local_run", help="Output directory for artifacts")
    parser.add_argument("--params-file", type=str, default="", help="Optional JSON file with parameter overrides")
    parser.add_argument("--preset", type=str, default="m1-air-balanced", choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"])
    parser.add_argument("--frames", type=int, default=8, help="Frames per clip")
    parser.add_argument("--height", type=int, default=64, help="Frame height")
    parser.add_argument("--width", type=int, default=64, help="Frame width")
    parser.add_argument("--fps", type=int, default=12, help="Frame rate")
    parser.add_argument("--latent-dim", type=int, default=256, help="Legacy latent budget used to size bottleneck channels when --latent-channels=0")
    parser.add_argument("--latent-channels", type=int, default=0, help="Explicit convolutional bottleneck channel count (0 = auto from --latent-dim)")
    parser.add_argument("--model-base-filters", type=int, default=32, help="Base filter count for video encoder/decoder")
    parser.add_argument("--model-kernel-size", type=int, default=3, help="Spatial/temporal kernel size used in core blocks")
    parser.add_argument("--latent-l1", type=float, default=0.0, help="Optional L1 activity penalty on bottleneck")
    parser.add_argument("--latent-l2", type=float, default=0.0, help="Optional L2 activity penalty on bottleneck")
    parser.add_argument("--latent-bits", type=int, default=16, help="Assumed bits per latent symbol for theoretical compression-ratio estimate")
    parser.add_argument(
        "--latent-bit-depth",
        type=int,
        default=0,
        choices=[0, 4, 6, 8, 10],
        help="Real quantization bit-depth for video bottleneck (0 disables quantization to preserve float32 behavior)",
    )
    parser.add_argument("--quant-noise-anneal", action="store_true", help="Enable QAT-style uniform noise annealing when --latent-bit-depth > 0")
    parser.add_argument("--rate-lambda", type=float, default=0.0, help="Weight for per-channel soft-histogram entropy rate loss (0 disables)")
    parser.add_argument(
        "--temporal-mode",
        type=str,
        default="independent",
        choices=["independent", "residual"],
        help="Temporal input mode: 'independent' encodes raw frames; 'residual' encodes keyframe + inter-frame deltas",
    )
    parser.add_argument(
        "--loss-profile",
        type=str,
        default="balanced",
        choices=["balanced", "psnr", "perceptual"],
        help="Preset weighting profile for video reconstruction loss",
    )
    parser.add_argument("--loss-mse-weight", type=float, default=0.75, help="Weight for MSE term in video loss")
    parser.add_argument("--loss-l1-weight", type=float, default=0.15, help="Weight for L1 term in video loss")
    parser.add_argument("--loss-ssim-weight", type=float, default=0.10, help="Weight for SSIM term in video loss")
    parser.add_argument("--loss-temporal-weight", type=float, default=0.0, help="Weight for frame-to-frame temporal consistency loss")
    parser.add_argument("--grad-clipnorm", type=float, default=1.0, help="Gradient clip norm for Adam optimizer")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=24, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/VIDEO DATA",
        help="Directory with real video files (.mp4, .mov, .mkv, .avi, .webm, .m4v)",
    )
    parser.add_argument("--cache-dir", type=str, default="", help="Optional directory for on-disk uint8 clip cache")
    parser.add_argument("--decode-workers", type=int, default=1, help="Parallel worker threads for ffmpeg video decoding")
    parser.add_argument("--real-val-ratio", type=float, default=0.15, help="Validation split ratio for real videos/clips")
    parser.add_argument("--real-test-ratio", type=float, default=0.15, help="Test split ratio for real videos/clips")
    parser.add_argument("--real-max-videos", type=int, default=0, help="Optional cap for discovered real videos (0 = all)")
    parser.add_argument("--real-max-clips", type=int, default=1800, help="Cap for extracted real clips to keep training practical")
    parser.add_argument("--real-clip-stride", type=int, default=4, help="Frame stride between extracted clips")
    parser.add_argument("--real-augment", dest="real_augment", action="store_true", help="Enable horizontal flip augmentation on training clips")
    parser.add_argument("--no-real-augment", dest="real_augment", action="store_false", help="Disable training clip augmentation")
    parser.add_argument("--target-psnr", type=float, default=25.0, help="Validation PSNR target")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable .tflite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Export float16 optimized tflite")
    parser.add_argument("--run-mp4-benchmark", action="store_true", help="Benchmark recon quality against MP4/VP9")
    parser.add_argument("--benchmark-codecs", type=str, default="h264", help="Comma-separated codecs for benchmark (h264,vp9)")
    parser.set_defaults(export_tflite=True, real_augment=True)
    return parser.parse_args()


def apply_preset(args):
    provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    preset_map = {
        "m1-air-fast": {"batch_size": 4, "latent_dim": 128, "epochs": 14, "lr": 1.8e-4},
        "m1-air-balanced": {"batch_size": 4, "latent_dim": 160, "epochs": 24, "lr": 1.2e-4},
        "m1-air-quality": {"batch_size": 3, "latent_dim": 224, "epochs": 36, "lr": 8e-5},
    }
    if args.preset in preset_map:
        key_to_flag = {
            "batch_size": "--batch-size",
            "latent_dim": "--latent-dim",
            "epochs": "--epochs",
            "lr": "--lr",
        }
        for k, v in preset_map[args.preset].items():
            if key_to_flag[k] not in provided_flags:
                setattr(args, k, v)

    loss_profiles = {
        "balanced": (0.75, 0.15, 0.10),
        "psnr": (0.85, 0.10, 0.05),
        "perceptual": (0.45, 0.25, 0.30),
    }
    if not any(f in provided_flags for f in ["--loss-mse-weight", "--loss-l1-weight", "--loss-ssim-weight"]):
        mse_w, l1_w, ssim_w = loss_profiles[args.loss_profile]
        args.loss_mse_weight = mse_w
        args.loss_l1_weight = l1_w
        args.loss_ssim_weight = ssim_w

    args.model_base_filters = max(8, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.grad_clipnorm = max(0.0, float(args.grad_clipnorm))
    args.rate_lambda = max(0.0, float(args.rate_lambda))
    args.loss_temporal_weight = max(0.0, float(args.loss_temporal_weight))
    args.decode_workers = max(1, int(args.decode_workers))
    return args


def resolve_latent_channels(latent_dim, latent_channels=0):
    if int(latent_channels) > 0:
        return int(latent_channels)
    return max(4, min(32, int(latent_dim) // 8))


def collect_video_paths(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return np.array([], dtype=np.str_)
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return np.array(sorted(paths), dtype=np.str_)


def decode_video_frames_uint8(ffmpeg_bin, in_path, width, height, fps):
    """Decode video frames as uint8 (4x smaller in RAM than float32) with aspect-preserving center crop."""
    vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-vf",
        vf,
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
        return np.empty((0, height, width, 3), dtype=np.uint8)
    arr = arr[: frame_count * frame_size]
    return arr.reshape(frame_count, height, width, 3).copy()


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
        raise ValueError("Need at least 3 items for train/val/test splits")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("real-val-ratio and real-test-ratio must be >=0 and sum to < 1")

    idx = rng.permutation(n)
    test_n = max(1, int(round(n * test_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    while n - val_n - test_n <= 0:
        if test_n > 1:
            test_n -= 1
        elif val_n > 1:
            val_n -= 1
        else:
            raise ValueError("Not enough items for requested split ratios")
    train_n = n - val_n - test_n
    return idx[:train_n], idx[train_n : train_n + val_n], idx[train_n + val_n :]


def _video_cache_key(video_paths, args):
    h = hashlib.sha256()
    h.update(
        f"v2|f={args.frames}|h={args.height}|w={args.width}|fps={args.fps}|"
        f"stride={args.real_clip_stride}|max={args.real_max_clips}|"
        f"val={args.real_val_ratio}|test={args.real_test_ratio}|seed={args.seed}".encode("utf-8")
    )
    for p in sorted(str(x) for x in video_paths):
        h.update(p.encode("utf-8"))
        try:
            st = Path(p).stat()
            h.update(f"|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
        except OSError:
            h.update(b"|missing")
    return h.hexdigest()[:20]


def _extract_clips_from_frames(frames_u8, num_frames, stride, max_clips=None):
    clips = []
    if len(frames_u8) < num_frames:
        return clips
    for start in range(0, len(frames_u8) - num_frames + 1, max(1, int(stride))):
        clips.append(frames_u8[start : start + num_frames])
        if max_clips is not None and len(clips) >= max_clips:
            break
    return clips


def extract_real_video_splits(args):
    """Extract non-leaking train/val/test uint8 clips, splitting at the video-file level when >= 3 videos exist."""
    ffmpeg_bin = resolve_ffmpeg_binary()
    if not isinstance(ffmpeg_bin, str) or not ffmpeg_bin:
        raise ValueError("ffmpeg not found (system or imageio-ffmpeg); real video loading is required.")

    video_paths = collect_video_paths(args.data_dir)
    if args.real_max_videos > 0:
        video_paths = video_paths[: args.real_max_videos]
    if len(video_paths) == 0:
        raise ValueError(f"No supported video files found in '{args.data_dir}'.")

    cache_path = None
    if args.cache_dir:
        cache_root = os.path.join(os.path.abspath(os.path.expanduser(args.cache_dir)), "video_clip_cache")
        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, f"clips_{_video_cache_key(video_paths, args)}.npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return (
                cached["train"],
                cached["val"],
                cached["test"],
                {
                    "split_strategy": str(cached["split_strategy"].item()),
                    "cache_status": "hit",
                    "cache_path": cache_path,
                    "skipped_videos": [],
                    "skipped_videos_count": 0,
                },
            )

    def _decode_one(path):
        try:
            frames_u8 = decode_video_frames_uint8(ffmpeg_bin, path, args.width, args.height, args.fps)
            if len(frames_u8) < args.frames:
                return path, None, f"too_short ({len(frames_u8)} < {args.frames} frames)"
            return path, frames_u8, None
        except Exception as exc:
            return path, None, str(exc)

    decoded_videos = []
    skipped_videos = []
    if args.decode_workers > 1 and len(video_paths) > 1:
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(video_paths))) as pool:
            results = list(pool.map(_decode_one, video_paths))
    else:
        results = [_decode_one(p) for p in video_paths]

    for path, frames_u8, err in results:
        if err is not None:
            skipped_videos.append({"path": str(path), "reason": err})
        else:
            decoded_videos.append((str(path), frames_u8))

    if skipped_videos:
        print(f"[!] Skipped {len(skipped_videos)} unreadable/short video(s)")

    if len(decoded_videos) == 0:
        raise ValueError(f"No readable videos with >= {args.frames} frames found in '{args.data_dir}'.")

    rng = np.random.default_rng(args.seed)
    stride = max(1, int(args.real_clip_stride))
    max_total = max(3, int(args.real_max_clips))

    if len(decoded_videos) >= 3:
        # Video-level split: zero frame overlap / zero scene leakage between train, val, and test!
        split_strategy = "video_level"
        tr_idx, va_idx, te_idx = split_indices(len(decoded_videos), args.real_val_ratio, args.real_test_ratio, rng)

        def _collect_group(indices, budget):
            per_vid_cap = max(1, math.ceil(budget / max(1, len(indices))))
            group = []
            for i in indices:
                group.extend(_extract_clips_from_frames(decoded_videos[i][1], args.frames, stride, max_clips=per_vid_cap))
                if len(group) >= budget:
                    break
            return group[:budget]

        val_budget = max(1, int(max_total * args.real_val_ratio))
        test_budget = max(1, int(max_total * args.real_test_ratio))
        train_budget = max(1, max_total - val_budget - test_budget)

        train_clips = _collect_group(tr_idx, train_budget)
        val_clips = _collect_group(va_idx, val_budget)
        test_clips = _collect_group(te_idx, test_budget)
    else:
        # Fallback for 1-2 videos: contiguous non-overlapping temporal blocks (no overlapping frames across splits)
        split_strategy = "contiguous_temporal_blocks"
        train_clips, val_clips, test_clips = [], [], []
        for _, frames_u8 in decoded_videos:
            total_f = len(frames_u8)
            val_f = max(args.frames, int(total_f * args.real_val_ratio))
            test_f = max(args.frames, int(total_f * args.real_test_ratio))
            train_f = total_f - val_f - test_f
            if train_f < args.frames:
                continue
            train_clips.extend(_extract_clips_from_frames(frames_u8[:train_f], args.frames, stride))
            val_clips.extend(_extract_clips_from_frames(frames_u8[train_f : train_f + val_f], args.frames, stride))
            test_clips.extend(_extract_clips_from_frames(frames_u8[train_f + val_f :], args.frames, stride))
        train_clips = train_clips[: max(1, int(max_total * (1.0 - args.real_val_ratio - args.real_test_ratio)))]
        val_clips = val_clips[: max(1, int(max_total * args.real_val_ratio))]
        test_clips = test_clips[: max(1, int(max_total * args.real_test_ratio))]

    if min(len(train_clips), len(val_clips), len(test_clips)) < 1:
        raise ValueError(
            f"Need at least 1 clip in each of train/val/test splits; "
            f"got train={len(train_clips)}, val={len(val_clips)}, test={len(test_clips)}."
        )

    train_arr = np.asarray(train_clips, dtype=np.uint8)
    val_arr = np.asarray(val_clips, dtype=np.uint8)
    test_arr = np.asarray(test_clips, dtype=np.uint8)

    cache_status = "disabled"
    if cache_path:
        tmp_path = cache_path + ".tmp.npz"
        np.savez(tmp_path, train=train_arr, val=val_arr, test=test_arr, split_strategy=np.array(split_strategy))
        os.replace(tmp_path, cache_path)
        cache_status = "built"

    return (
        train_arr,
        val_arr,
        test_arr,
        {
            "split_strategy": split_strategy,
            "cache_status": cache_status,
            "cache_path": cache_path or "",
            "skipped_videos": skipped_videos[:20],
            "skipped_videos_count": len(skipped_videos),
        },
    )


def make_dataset_from_clips(clips_u8, batch_size, seed, shuffle, repeat=False, augment=False):
    def _prep(clip_u8):
        clip = tf.cast(clip_u8, tf.float32) / 255.0
        if augment:
            flip = tf.random.uniform(()) > 0.5
            clip = tf.cond(flip, lambda: tf.reverse(clip, axis=[2]), lambda: clip)
        return clip, clip

    ds = tf.data.Dataset.from_tensor_slices(clips_u8)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(clips_u8), 2048), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_real_video_datasets(args):
    train_clips, val_clips, test_clips, meta = extract_real_video_splits(args)

    train_data = make_dataset_from_clips(
        train_clips, args.batch_size, args.seed, shuffle=True, repeat=True, augment=args.real_augment
    )
    val_data = make_dataset_from_clips(val_clips, args.batch_size, args.seed + 1, shuffle=False, repeat=True, augment=False)
    test_data = make_dataset_from_clips(test_clips, args.batch_size, args.seed + 2, shuffle=False, repeat=False, augment=False)
    counts = {
        "train": int(len(train_clips)),
        "val": int(len(val_clips)),
        "test": int(len(test_clips)),
        "source": "real_filesystem",
        **meta,
    }
    return train_data, val_data, test_data, counts


def make_video_loss(mse_weight=0.75, l1_weight=0.15, ssim_weight=0.10, temporal_weight=0.0):
    mse_w = float(mse_weight)
    l1_w = float(l1_weight)
    ssim_w = float(ssim_weight)
    temp_w = max(0.0, float(temporal_weight))
    total = max(mse_w + l1_w + ssim_w + temp_w, 1e-8)
    mse_w, l1_w, ssim_w, temp_w = mse_w / total, l1_w / total, ssim_w / total, temp_w / total

    def video_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

        y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
        y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
        ssim_term = 1.0 - tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0))

        loss = mse_w * mse + l1_w * l1 + ssim_w * ssim_term
        if temp_w > 0.0:
            dt_true = y_true[:, 1:, :, :, :] - y_true[:, :-1, :, :, :]
            dt_pred = y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]
            loss = loss + temp_w * tf.reduce_mean(tf.abs(dt_true - dt_pred))
        return loss

    return video_loss


def psnr_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
    y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
    return tf.reduce_mean(tf.image.psnr(y_true_flat, y_pred_flat, max_val=1.0))


def ssim_metric(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[2], tf.shape(y_true)[3], 3))
    y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[2], tf.shape(y_pred)[3], 3))
    return tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0))


def _conv2p1d_block(x, filters, k, strides=(1, 1, 1), residual=False):
    """Factorized (2+1)D convolution block: ~2.2x fewer params than 3D Conv3D with faster M1 execution."""
    st_t, st_h, st_w = strides
    h = layers.Conv3D(filters, (1, k, k), strides=(1, st_h, st_w), padding="same")(x)
    h = layers.LeakyReLU(negative_slope=0.1)(h)
    h = layers.Conv3D(filters, (k, 1, 1), strides=(st_t, 1, 1), padding="same")(h)
    if residual and strides == (1, 1, 1) and x.shape[-1] == filters:
        h = layers.Add()([x, h])
    return layers.LeakyReLU(negative_slope=0.1)(h)


def build_video_autoencoder(
    frames,
    height,
    width,
    latent_dim,
    latent_l1=0.0,
    latent_l2=0.0,
    model_base_filters=32,
    model_kernel_size=3,
    latent_channels=0,
    latent_bit_depth=0,
    quant_noise_anneal=False,
    rate_lambda=0.0,
    temporal_mode="independent",
):
    """Compact fully convolutional (2+1)D video autoencoder with true bottleneck (no bypass skips)."""
    inputs = layers.Input(shape=(frames, height, width, 3))
    base_filters = max(8, int(model_base_filters))
    k = max(1, int(model_kernel_size))
    stage2_filters = max(base_filters * 2, base_filters + 8)
    stage3_filters = max(base_filters * 3, stage2_filters + 8)
    c_latent = resolve_latent_channels(latent_dim, latent_channels)

    x = TemporalDeltaInput(name="temporal_delta_input")(inputs) if temporal_mode == "residual" else inputs

    # Encoder: 2x temporal, 8x spatial downsampling (128x volume reduction before channels)
    x = _conv2p1d_block(x, base_filters, k, strides=(1, 2, 2))
    x = _conv2p1d_block(x, base_filters, k, residual=True)

    x = _conv2p1d_block(x, stage2_filters, k, strides=(2, 2, 2))
    x = _conv2p1d_block(x, stage2_filters, k, residual=True)

    x = _conv2p1d_block(x, stage3_filters, k, strides=(1, 2, 2))
    x = _conv2p1d_block(x, stage3_filters, k, residual=True)

    # True convolutional bottleneck (replaces GlobalAvgPool + 5M-param Dense + bypass skips)
    bottleneck_reg = None
    if latent_l1 > 0.0 or latent_l2 > 0.0:
        bottleneck_reg = regularizers.L1L2(l1=float(max(0.0, latent_l1)), l2=float(max(0.0, latent_l2)))
    latent_pre = layers.Conv3D(
        c_latent,
        (1, 1, 1),
        padding="same",
        activation="sigmoid",
        activity_regularizer=bottleneck_reg,
        name="bottleneck_pre_quant",
    )(x)
    latent = VideoStraightThroughQuantize(
        bit_depth=latent_bit_depth,
        quant_noise_anneal=quant_noise_anneal,
        name="bottleneck",
    )(latent_pre)
    latent_penalized = VideoEntropyRatePenalty(rate_lambda, name="rate_penalty")(latent)

    # Decoder: strictly reconstructs from bottleneck only (enables standalone encoder/decoder split)
    x = layers.Conv3D(stage3_filters, (1, 1, 1), padding="same")(latent_penalized)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = _conv2p1d_block(x, stage3_filters, k, residual=True)

    x = layers.UpSampling3D(size=(1, 2, 2))(x)
    x = _conv2p1d_block(x, stage2_filters, k)
    x = _conv2p1d_block(x, stage2_filters, k, residual=True)

    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = _conv2p1d_block(x, base_filters, k)
    x = _conv2p1d_block(x, base_filters, k, residual=True)

    x = layers.UpSampling3D(size=(1, 2, 2))(x)
    x = _conv2p1d_block(x, base_filters, k)
    outputs = layers.Conv3D(3, (1, k, k), padding="same", activation="sigmoid")(x)

    return Model(inputs, outputs, name="video_autoencoder")


def compute_video_latent_entropy_stats(model, test_data, frames, height, width, bit_depth=8, max_clips=16):
    """Measure real empirical Shannon entropy of the quantized video latent tensor."""
    effective_bits = int(bit_depth) if int(bit_depth) > 0 else 8
    num_levels = 1 << effective_bits
    max_level = num_levels - 1

    latent_layer = model.get_layer("bottleneck")
    latent_model = Model(model.input, latent_layer.output, name="video_latent_extractor")

    chunks = []
    seen = 0
    for batch_x, _ in test_data:
        n = int(batch_x.shape[0])
        take_n = min(max_clips - seen, n)
        if take_n <= 0:
            break
        chunks.append(latent_model.predict(batch_x[:take_n], verbose=0))
        seen += take_n
        if seen >= max_clips:
            break

    if not chunks:
        return None

    latents = np.concatenate(chunks, axis=0)
    levels = np.clip(np.round(latents * max_level), 0, max_level).astype(np.int64)
    counts = np.bincount(levels.reshape(-1), minlength=num_levels).astype(np.float64)
    total = counts.sum()
    probs = counts[counts > 0] / max(total, 1.0)
    entropy_bps = float(-np.sum(probs * np.log2(probs)))

    symbols_per_clip = int(np.prod(levels.shape[1:]))
    bits_per_clip = entropy_bps * symbols_per_clip
    bytes_per_clip = bits_per_clip / 8.0
    raw_bytes_per_clip = float(frames * height * width * 3)
    return {
        "bit_depth_evaluated": effective_bits,
        "quantized_during_training": bool(int(bit_depth) > 0),
        "latent_shape_per_clip": [int(d) for d in levels.shape[1:]],
        "latent_symbols_per_clip": symbols_per_clip,
        "latent_entropy_bits_per_symbol": entropy_bps,
        "latent_bits_per_pixel": float(bits_per_clip / max(1, frames * height * width)),
        "actual_latent_bytes_per_clip": float(bytes_per_clip),
        "actual_input_to_latent_ratio": float(raw_bytes_per_clip / max(bytes_per_clip, 1.0)),
    }


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
    os.makedirs(run_dir, exist_ok=True)
    tflite_path = os.path.join(run_dir, "video_autoencoder.tflite")
    with tempfile.TemporaryDirectory(prefix="video_tflite_export_") as tmp_dir:
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


def plot_history(history, plot_path):
    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)
    if len(epochs) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (tr_k, va_k, title) in zip(
        axes,
        [("loss", "val_loss", "Loss"), ("psnr_metric", "val_psnr_metric", "PSNR"), ("ssim_metric", "val_ssim_metric", "SSIM")],
    ):
        if hist.get(tr_k):
            ax.plot(epochs[: len(hist[tr_k])], hist[tr_k], label="train")
        if hist.get(va_k):
            ax.plot(epochs[: len(hist[va_k])], hist[va_k], label="val")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def compute_psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


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


def run_mp4_benchmark(model, test_data, run_dir, fps, codecs_csv="h264", latent_stats=None):
    ffmpeg = resolve_ffmpeg_binary()
    report_path = os.path.join(run_dir, "video_mp4_benchmark.md")
    if not isinstance(ffmpeg, str) or not ffmpeg:
        payload = {"status": "skipped", "reason": "ffmpeg_not_found"}
        write_markdown_json_report(payload, report_path, title="Video MP4 Benchmark")
        return report_path

    codecs = [c.strip().lower() for c in str(codecs_csv).split(",") if c.strip()]
    if not codecs:
        codecs = ["h264"]

    clips = []
    for x, _ in test_data.take(3):
        clips.extend([x[i].numpy() for i in range(min(2, x.shape[0]))])

    model_latent_bytes = (
        float(latent_stats["actual_latent_bytes_per_clip"]) if isinstance(latent_stats, dict) else None
    )

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, clip in enumerate(clips):
            recon = model.predict(clip[np.newaxis, ...], verbose=0)[0]
            model_psnr = compute_psnr(np.clip(clip, 0.0, 1.0), np.clip(recon, 0.0, 1.0))
            raw_bytes = int(clip.size)
            _, h, w, _ = clip.shape

            for codec in codecs:
                if codec == "h264":
                    enc_args = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
                    ext = "mp4"
                    crfs = [20, 24, 28]
                elif codec == "vp9":
                    enc_args = ["-c:v", "libvpx-vp9", "-b:v", "0", "-pix_fmt", "yuv420p"]
                    ext = "webm"
                    crfs = [24, 32, 40]
                else:
                    continue

                for crf in crfs:
                    out_path = os.path.join(tmp, f"clip_{idx}_{codec}_crf{crf}.{ext}")
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
                        *enc_args,
                        "-crf",
                        str(crf),
                        out_path,
                    ]
                    try:
                        subprocess.run(cmd, input=(np.clip(clip, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes(), check=True)
                        decoded = np.asarray(decode_mp4(ffmpeg, out_path, w, h), dtype=np.float32)
                    except Exception:
                        continue
                    n = min(len(decoded), len(clip))
                    if n <= 0:
                        continue
                    codec_psnr = compute_psnr(np.clip(clip[:n], 0.0, 1.0), np.clip(decoded[:n], 0.0, 1.0))
                    codec_bytes = os.path.getsize(out_path)
                    results.append(
                        {
                            "clip_idx": idx,
                            "codec": codec,
                            "crf": crf,
                            "model_psnr": model_psnr,
                            "mp4_psnr": codec_psnr,
                            "codec_bytes": int(codec_bytes),
                            "mp4_size_ratio": codec_bytes / max(raw_bytes, 1),
                            "model_latent_bytes": model_latent_bytes,
                            "model_latent_vs_codec_ratio": (
                                model_latent_bytes / max(codec_bytes, 1) if model_latent_bytes is not None else None
                            ),
                        }
                    )

    payload = {"status": "ok", "codecs": codecs, "results": results}
    write_markdown_json_report(payload, report_path, title="Video Codec Benchmark")
    return report_path


def estimate_video_compression_ratio(frames, height, width, latent_channels, latent_bits):
    input_bits = int(frames) * int(height) * int(width) * 3 * 8
    t_down = max(1, int(frames) // 2)
    h_down = max(1, int(height) // 8)
    w_down = max(1, int(width) // 8)
    latent_symbols = t_down * h_down * w_down * int(latent_channels)
    latent_bits_total = max(1, latent_symbols * int(max(1, latent_bits)))
    return {
        "input_bits_per_clip": int(input_bits),
        "latent_symbols_per_clip": int(latent_symbols),
        "latent_bits_per_clip": int(latent_bits_total),
        "estimated_input_to_latent_ratio": float(input_bits / latent_bits_total),
    }


def save_report(
    args,
    counts,
    history,
    eval_values,
    run_dir,
    plot_path,
    preview_path,
    benchmark_path,
    compression_estimate,
    latent_entropy_stats,
    model,
):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "split_info": counts,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "model_diagnostics": {
            "parameter_count": int(model.count_params()),
            "input_shape": [None if x is None else int(x) for x in model.input_shape],
            "output_shape": [None if x is None else int(x) for x in model.output_shape],
        },
        "plot_path": plot_path,
        "preview": preview_path,
        "benchmark": benchmark_path,
        "compression_estimate": compression_estimate,
        "latent_entropy_stats": latent_entropy_stats,
    }
    report_path = os.path.join(run_dir, "video_evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Video Evaluation Report")
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def main():
    args = apply_preset(parse_args())
    if args.params_file:
        override_result = apply_overrides(args, load_overrides(args.params_file, section="video"))
        if override_result.applied:
            print(f"[*] Applied {len(override_result.applied)} params from {args.params_file}")
        if override_result.unknown:
            print(f"[!] Ignored unknown params in file: {sorted(override_result.unknown)}")
    args.model_base_filters = max(8, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.grad_clipnorm = max(0.0, float(args.grad_clipnorm))
    configure_runtime(args.seed)

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height and width must be divisible by 8")
    if args.frames % 2 != 0:
        raise ValueError("frames must be divisible by 2")

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, counts = build_real_video_datasets(args)
    print(
        f"[*] Using real video clips from '{args.data_dir}' "
        f"(strategy={counts.get('split_strategy')}, train={counts['train']}, val={counts['val']}, test={counts['test']})"
    )
    steps_per_epoch = max(1, math.ceil(counts["train"] / args.batch_size))
    val_steps = max(1, math.ceil(counts["val"] / args.batch_size))
    test_steps = max(1, math.ceil(counts["test"] / args.batch_size))

    c_latent = resolve_latent_channels(args.latent_dim, args.latent_channels)
    model = build_video_autoencoder(
        args.frames,
        args.height,
        args.width,
        args.latent_dim,
        latent_l1=args.latent_l1,
        latent_l2=args.latent_l2,
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
        latent_channels=c_latent,
        latent_bit_depth=args.latent_bit_depth,
        quant_noise_anneal=args.quant_noise_anneal,
        rate_lambda=args.rate_lambda,
        temporal_mode=args.temporal_mode,
    )
    loss_fn = make_video_loss(
        mse_weight=args.loss_mse_weight,
        l1_weight=args.loss_l1_weight,
        ssim_weight=args.loss_ssim_weight,
        temporal_weight=args.loss_temporal_weight,
    )
    opt_kwargs = {"learning_rate": args.lr}
    if args.grad_clipnorm > 0:
        opt_kwargs["clipnorm"] = args.grad_clipnorm
    model.compile(
        optimizer=tf.keras.optimizers.Adam(**opt_kwargs),
        loss=loss_fn,
        metrics=["mse", psnr_metric, ssim_metric],
    )

    best_path = os.path.join(run_dir, "best_video_model.keras")
    cbs = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=1e-6, verbose=1),
        TargetPSNRCallback(args.target_psnr),
    ]
    if args.quant_noise_anneal and args.latent_bit_depth > 0:
        cbs.append(VideoQuantNoiseAnnealCallback(model.get_layer("bottleneck"), args.epochs))

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
    plot_path = os.path.join(run_dir, "video_training_metrics.png")
    plot_history(history, plot_path)
    preview_path = save_video_preview(model, test_data, run_dir)

    latent_entropy_stats = compute_video_latent_entropy_stats(
        model, test_data, args.frames, args.height, args.width, bit_depth=args.latent_bit_depth
    )

    benchmark_path = None
    if args.run_mp4_benchmark:
        benchmark_path = run_mp4_benchmark(
            model,
            test_data,
            run_dir,
            args.fps,
            codecs_csv=args.benchmark_codecs,
            latent_stats=latent_entropy_stats,
        )

    weights_path = os.path.join(run_dir, "video_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        try:
            tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)
        except Exception as exc:
            print(f"[!] TFLite export skipped: {exc}")

    compression_estimate = estimate_video_compression_ratio(
        args.frames, args.height, args.width, c_latent, args.latent_bit_depth if args.latent_bit_depth > 0 else args.latent_bits
    )
    report_path = save_report(
        args,
        counts,
        history,
        eval_values,
        run_dir,
        plot_path,
        preview_path,
        benchmark_path,
        compression_estimate,
        latent_entropy_stats,
        model,
    )

    print("\n[+] Video training complete")
    print(f"[+] Parameters: {model.count_params():,} (latent_channels={c_latent}, temporal_mode={args.temporal_mode})")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Plot: {plot_path}")
    print(f"[+] Preview: {preview_path}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    print(f"[+] Estimated input/latent ratio: {compression_estimate['estimated_input_to_latent_ratio']:.2f}x")
    if latent_entropy_stats:
        print(
            f"[+] Real latent entropy: {latent_entropy_stats['latent_bits_per_pixel']:.3f} bpp "
            f"(~{latent_entropy_stats['actual_latent_bytes_per_clip']:.0f} bytes/clip, "
            f"{latent_entropy_stats['actual_input_to_latent_ratio']:.2f}x vs raw RGB)"
        )
    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}).")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")
    if benchmark_path:
        print(f"[+] MP4 benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
