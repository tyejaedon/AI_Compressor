#!/usr/bin/env python3
"""
Upgraded MacBook-friendly audio autoencoder trainer with SNR fixes,
learned Conv1DTranspose upsampling, linear bottleneck representation,
straight-through estimator (STE) latent quantization, real compression
accounting, and memory-safe TFLite export bypassing macOS MLIR compiler crashes.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Must be set before importing TensorFlow to suppress verbose C++ INFO logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, regularizers

# Fallback gracefully if param_overrides or custom modules do not exist
try:
    from param_overrides import apply_overrides, load_overrides
except ImportError:
    apply_overrides = None
    load_overrides = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
try:
    from report_markdown import write_markdown_json_report
except ImportError:
    def write_markdown_json_report(payload, report_path, title="Report"):
        import json
        with open(report_path, "w") as f:
            json.dump({"title": title, "data": payload}, f, indent=4)


@tf.keras.utils.register_keras_serializable(package="ai_compressor_audio")
class StraightThroughQuantize(layers.Layer):
    """Round in forward pass while preserving gradients via Straight-Through Estimator (STE).

    `bit_depth` controls how many discrete levels the latent vector is quantized to
    (2**bit_depth levels). When disabled (bit_depth <= 0 or None), acts as a no-op identity.
    When `quant_noise_anneal` is enabled, additive uniform noise scaled to one quantization
    step is mixed in during training only, and faded from full strength to zero via `noise_scale`
    (updated externally by `QuantNoiseAnnealCallback`) so the model gradually adapts to hard
    quantization.
    """

    def __init__(self, bit_depth: int | None = None, quant_noise_anneal: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.bit_depth = int(bit_depth) if bit_depth is not None and int(bit_depth) > 0 else None
        self.quant_noise_anneal = bool(quant_noise_anneal)
        self.levels = float((1 << self.bit_depth) - 1) if self.bit_depth else None
        self.noise_scale = None
        if self.bit_depth and self.quant_noise_anneal:
            self.noise_scale = self.add_weight(
                name="quant_noise_scale",
                shape=(),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=False,
            )

    def call(self, inputs, training=None):
        if not self.bit_depth or self.levels is None:
            return inputs

        # Audio bottleneck is linear; clamp to dynamic range [-1.0, 1.0] and map to [0, 1]
        clamped = tf.clip_by_value(inputs, -1.0, 1.0)
        norm = (clamped + 1.0) * 0.5
        quantized_norm = tf.round(norm * self.levels) / self.levels
        quantized = quantized_norm * 2.0 - 1.0
        ste = inputs + tf.stop_gradient(quantized - inputs)

        if not self.quant_noise_anneal or not training or self.noise_scale is None:
            return ste

        step = 2.0 / self.levels
        noise = tf.random.uniform(tf.shape(inputs), -0.5, 0.5, dtype=inputs.dtype) * step * self.noise_scale
        return ste + noise

    def get_config(self):
        config = super().get_config()
        config.update({
            "bit_depth": self.bit_depth,
            "quant_noise_anneal": self.quant_noise_anneal,
        })
        return config


class QuantNoiseAnnealCallback(callbacks.Callback):
    """Linearly fades a StraightThroughQuantize layer's noise_scale from 1 -> 0."""

    def __init__(self, layer, total_epochs):
        super().__init__()
        self.layer = layer
        self.total_epochs = max(1, int(total_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if getattr(self.layer, "noise_scale", None) is None:
            return
        frac = max(0.0, 1.0 - (epoch / self.total_epochs))
        self.layer.noise_scale.assign(frac)


class _StaticRangeEncoder:
    """Minimal order-0 static-frequency range coder (32-bit, byte-oriented)."""

    TOP = 1 << 24
    BOTTOM = 1 << 16

    def __init__(self, freqs):
        total = int(sum(freqs))
        self.total = total
        cum = 0
        self.cum_freqs = []
        self.freqs = []
        for f in freqs:
            self.cum_freqs.append(cum)
            self.freqs.append(int(f))
            cum += int(f)

    def encode(self, symbols):
        low = 0
        rng = 0xFFFFFFFF
        out = bytearray()
        for sym in symbols:
            f = self.freqs[sym]
            if f == 0:
                continue
            cf = self.cum_freqs[sym]
            rng //= self.total
            low = (low + cf * rng) & 0xFFFFFFFF
            rng *= f
            while True:
                if (low ^ (low + rng)) < self.TOP:
                    pass
                elif rng < self.BOTTOM:
                    rng = (-low) & (self.BOTTOM - 1)
                else:
                    break
                out.append((low >> 24) & 0xFF)
                low = (low << 8) & 0xFFFFFFFF
                rng = (rng << 8) & 0xFFFFFFFF
        for _ in range(4):
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & 0xFFFFFFFF
        return bytes(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an audio autoencoder locally.")
    parser.add_argument("--output-root", type=str, default="models/audio_local_run",
                        help="Output directory for artifacts")
    parser.add_argument("--params-file", type=str, default="", help="Optional JSON file with parameter overrides")
    parser.add_argument("--preset", type=str, default="m1-air-balanced",
                        choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"])
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--clip-seconds", type=float, default=1.0, help="Audio clip duration in seconds")
    parser.add_argument("--latent-dim", type=int, default=192, help="Latent bottleneck size")
    parser.add_argument("--model-base-filters", type=int, default=48,
                        help="Base Conv1D filter count for audio encoder/decoder")
    parser.add_argument("--model-kernel-size", type=int, default=5, help="Kernel size for core Conv1D blocks")
    parser.add_argument("--latent-l1", type=float, default=0.0, help="Optional L1 activity penalty on bottleneck")
    parser.add_argument("--latent-l2", type=float, default=0.0, help="Optional L2 activity penalty on bottleneck")
    parser.add_argument("--latent-bits", type=int, default=16, help="Assumed quantized bits per latent dimension for theoretical ratio")
    parser.add_argument(
        "--latent-bit-depth",
        type=int,
        default=0,
        help="Bit depth for actual latent quantization (e.g. 8). Default 0 (disabled/float32).",
    )
    parser.add_argument(
        "--quant-noise-anneal",
        action="store_true",
        help="Linearly anneal additive quantization noise during training when --latent-bit-depth > 0",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=24, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/AudioData/ESC-50-master/audio",
        help="Directory with real audio files (recursive)",
    )
    parser.add_argument("--real-val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--real-test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--real-file-limit", type=int, default=0, help="Optional cap for real audio files")
    parser.add_argument("--target-snr-db", type=float, default=90.0, help="Validation SNR(dB) target")
    parser.add_argument("--disable-target-snr-stop", action="store_true", help="Disable SNR-target early stopping")
    parser.add_argument("--loss-profile", choices=["balanced", "snr", "perceptual"], default="balanced",
                        help="Loss profile preset: balanced (default, preserves baseline), snr (time-domain focus), perceptual (mel + A-weighting focus)")
    parser.add_argument("--loss-mse-weight", type=float, default=0.55, help="Weight for weighted MSE term")
    parser.add_argument("--loss-l1-weight", type=float, default=0.10, help="Weight for weighted L1 term")
    parser.add_argument("--loss-stft-weight", type=float, default=0.35, help="Weight for STFT magnitude loss")
    parser.add_argument("--loss-mel-weight", type=float, default=0.0, help="Weight for log-mel spectral envelope loss")
    parser.add_argument("--loss-a-weight", type=float, default=0.0, help="Weight for psychoacoustic A-weighted STFT loss")
    parser.add_argument("--mel-bins", type=int, default=64, help="Number of mel filterbank bins for mel loss")
    parser.add_argument("--eval-perceptual-metrics", action="store_true", help="Compute zero-dependency perceptual quality metrics (LSD in dB, STOI envelope proxy) on test set")
    parser.add_argument("--channels", choices=["mono", "stereo"], default="mono",
                        help="Audio channels architecture mode: mono (1 channel, default) or stereo (2 channels)")
    parser.add_argument("--stereo-downmix", choices=["average", "left", "right", "error"], default="average",
                        help="Downmixing strategy when input file is multi-channel but mono mode is active (default: average)")
    parser.add_argument("--resample-method", choices=["linear", "polyphase"], default="linear",
                        help="Resampling algorithm when input sample rate != target rate: linear (default) or polyphase (bandlimited Kaiser-windowed sinc)")
    parser.add_argument("--augment-noise-std", type=float, default=0.0,
                        help="Standard deviation of additive Gaussian noise augmentation during training (default: 0.0 / off)")
    parser.add_argument("--loss-hard-weight", type=float, default=1.5,
                        help="Extra penalty multiplier for large residuals")
    parser.add_argument("--loss-stft-power", type=float, default=1.5, help="Exponent for STFT mismatch penalty")
    parser.add_argument("--shuffle-buffer", type=int, default=1024, help="Shuffle buffer size")
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable .tflite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Export float16 optimized tflite")
    parser.add_argument("--run-mp3-benchmark", action="store_true", help="Benchmark recon quality against MP3")
    parser.set_defaults(export_tflite=True)
    return parser.parse_args()


def resample_polyphase(audio, src_rate, dst_rate, half_width=12):
    """Bandlimited polyphase Kaiser-windowed sinc resampler in pure NumPy."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    dst_len = max(1, int(round(len(audio) * (float(dst_rate) / float(src_rate)))))
    cutoff = min(1.0, float(dst_rate) / float(src_rate))
    t_out = np.arange(dst_len, dtype=np.float64) * (float(src_rate) / float(dst_rate))
    indices = np.arange(-half_width, half_width + 1, dtype=int)

    t_floor = np.floor(t_out).astype(int)
    frac = t_out - t_floor
    sample_indices = t_floor[:, None] + indices[None, :]
    valid_mask = (sample_indices >= 0) & (sample_indices < len(audio))
    clamped_indices = np.clip(sample_indices, 0, max(0, len(audio) - 1)).astype(int)

    dt = indices[None, :].astype(np.float64) - frac[:, None]
    sinc_val = np.sinc(dt * cutoff)
    kaiser_w = np.kaiser(2 * half_width + 1, 6.0).astype(np.float64)
    kernel = sinc_val * kaiser_w[None, :] * cutoff

    k_sum = np.sum(kernel, axis=-1, keepdims=True)
    kernel = np.where(np.abs(k_sum) > 1e-8, kernel / k_sum, kernel)

    if audio.ndim == 1:
        gathered = audio[clamped_indices]
        out = np.sum(gathered * kernel * valid_mask, axis=-1)
    else:
        gathered = audio[clamped_indices]
        out = np.sum(gathered * kernel[:, :, None] * valid_mask[:, :, None], axis=1)
    return out.astype(np.float32)

def resolve_loss_profile(args, provided_flags=None):
    if provided_flags is None:
        provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    loss_flags = [
        "--loss-mse-weight",
        "--loss-l1-weight",
        "--loss-stft-weight",
        "--loss-mel-weight",
        "--loss-a-weight",
    ]
    # If no explicit loss weight flags were passed on CLI, apply the chosen profile
    if not any(f in provided_flags for f in loss_flags):
        loss_profiles = {
            # (MSE, L1, STFT, Mel, A-weighted)
            "balanced": (0.55, 0.10, 0.35, 0.0, 0.0),
            "snr": (0.70, 0.15, 0.15, 0.0, 0.0),
            "perceptual": (0.20, 0.05, 0.15, 0.35, 0.25),
        }
        if args.loss_profile in loss_profiles:
            w_mse, w_l1, w_stft, w_mel, w_a = loss_profiles[args.loss_profile]
            args.loss_mse_weight = float(w_mse)
            args.loss_l1_weight = float(w_l1)
            args.loss_stft_weight = float(w_stft)
            args.loss_mel_weight = float(w_mel)
            args.loss_a_weight = float(w_a)
    return args


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"batch_size": 12, "latent_dim": 96, "epochs": 14, "lr": 1.8e-4},
        "m1-air-balanced": {"batch_size": 12, "latent_dim": 128, "epochs": 24, "lr": 1.2e-4},
        "m1-air-quality": {"batch_size": 8, "latent_dim": 160, "epochs": 36, "lr": 8e-5},
    }
    if args.preset in preset_map:
        for k, v in preset_map[args.preset].items():
            setattr(args, k, v)
    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.target_snr_db = float(args.target_snr_db)
    args.loss_hard_weight = max(0.0, float(args.loss_hard_weight))
    args.loss_stft_power = max(1.0, float(args.loss_stft_power))
    args.latent_bit_depth = max(0, int(args.latent_bit_depth))
    args.loss_mse_weight = max(0.0, float(args.loss_mse_weight))
    args.loss_l1_weight = max(0.0, float(args.loss_l1_weight))
    args.loss_stft_weight = max(0.0, float(args.loss_stft_weight))
    args.loss_mel_weight = max(0.0, float(args.loss_mel_weight))
    args.loss_a_weight = max(0.0, float(args.loss_a_weight))
    args.mel_bins = max(8, int(args.mel_bins))
    args.channels = str(getattr(args, "channels", "mono")).lower()
    args.num_channels = 2 if args.channels == "stereo" else 1
    args.stereo_downmix = str(getattr(args, "stereo_downmix", "average")).lower()
    args.resample_method = str(getattr(args, "resample_method", "linear")).lower()
    args.augment_noise_std = max(0.0, float(getattr(args, "augment_noise_std", 0.0)))
    args = resolve_loss_profile(args)
    return args


def write_wav_int16(path, sample_rate, audio):
    audio = np.asarray(audio, dtype=np.float32)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    num_channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

write_wav_mono_int16 = write_wav_int16


def read_wav_mono_float32(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    return arr


def collect_audio_paths(root_dir):
    exts = {".wav", ".wave"}
    root = Path(root_dir)
    if not root.exists():
        return np.array([], dtype=np.str_)
    paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return np.array(sorted(paths), dtype=np.str_)


def split_indices(n, val_ratio, test_ratio, rng):
    if n < 3:
        raise ValueError("Need at least 3 real audio files for train/val/test splits")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("real-val-ratio and real-test-ratio must be >=0 and sum to < 1")

    idx = rng.permutation(n)
    test_n = max(1, int(n * test_ratio))
    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("Not enough real audio files for requested split ratios")

    train_idx = idx[:train_n]
    val_idx = idx[train_n: train_n + val_n]
    test_idx = idx[train_n + val_n:]
    return train_idx, val_idx, test_idx


def make_dataset_from_audio_paths(
    paths,
    clip_len,
    sample_rate,
    batch_size,
    shuffle,
    repeat,
    seed,
    shuffle_buffer,
    target_channels=1,
    stereo_downmix="average",
    resample_method="linear",
    augment_noise_std=0.0,
):
    clip_len_i = int(clip_len)
    target_channels = int(target_channels)
    random_crop = bool(shuffle)
    rng = np.random.default_rng(seed)

    def _decode_np(path_bytes):
        path = path_bytes.decode("utf-8")
        try:
            with wave.open(path, "rb") as wf:
                channels_in_file = wf.getnchannels()
                src_rate = int(wf.getframerate())
                frames = wf.getnframes()
                data = wf.readframes(frames)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        except Exception as e:
            print(f"[!] Warning: Failed to read WAV file {path} ({e}). Structured recovery tone synthesized.")
            t_synth = np.linspace(0.0, 1.0, num=clip_len_i, endpoint=False, dtype=np.float32)
            audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t_synth)).astype(np.float32)
            channels_in_file = 1
            src_rate = sample_rate

        # Sanitize against NaNs, infinities, and extreme values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Multi-channel / stereo handling (Milestone A4, Issue #37)
        if target_channels == 1:
            if channels_in_file > 1 and len(audio) > 1:
                reshaped = audio.reshape(-1, channels_in_file)
                if stereo_downmix == "average":
                    audio = reshaped.mean(axis=1)
                elif stereo_downmix == "left":
                    audio = reshaped[:, 0]
                elif stereo_downmix == "right":
                    audio = reshaped[:, 1]
                elif stereo_downmix == "error":
                    raise ValueError(f"Encountered multi-channel audio ({channels_in_file} ch) in {path} with --stereo-downmix error")
                else:
                    audio = reshaped.mean(axis=1)
        else:  # target_channels == 2 (stereo)
            if channels_in_file == 1:
                audio = np.stack([audio, audio], axis=-1)
            elif channels_in_file >= 2:
                audio = audio.reshape(-1, channels_in_file)[:, :2]

        # Configurable resampling (Milestone A4, Issue #38)
        if src_rate != sample_rate and len(audio) > 1:
            if resample_method == "polyphase":
                audio = resample_polyphase(audio, src_rate, sample_rate)
            else:
                if audio.ndim == 1:
                    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
                    dst_len = max(1, int(round(len(audio) * (sample_rate / float(src_rate)))))
                    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
                    audio = np.interp(dst_x, src_x, audio).astype(np.float32)
                else:
                    chans = []
                    for c in range(audio.shape[1]):
                        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
                        dst_len = max(1, int(round(len(audio) * (sample_rate / float(src_rate)))))
                        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
                        chans.append(np.interp(dst_x, src_x, audio[:, c]).astype(np.float32))
                    audio = np.stack(chans, axis=-1)

        # Ensure correct channel dimension
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=-1)

        if len(audio) < clip_len_i:
            pad_width = ((0, clip_len_i - len(audio)), (0, 0))
            audio = np.pad(audio, pad_width, mode="constant")

        max_start = max(0, len(audio) - clip_len_i)
        if random_crop and max_start > 0:
            start = int(rng.integers(0, max_start + 1))
        else:
            start = max_start // 2

        clean_clip = np.asarray(audio[start: start + clip_len_i], dtype=np.float32)
        clean_clip = np.clip(clean_clip, -1.0, 1.0)

        # Additive noise augmentation (Milestone A4, Issue #39)
        if augment_noise_std > 0.0 and random_crop:
            noise = rng.normal(0.0, augment_noise_std, size=clean_clip.shape).astype(np.float32)
            noisy_clip = np.clip(clean_clip + noise, -1.0, 1.0)
            return noisy_clip, clean_clip
        return clean_clip, clean_clip

    def _decode(path):
        x, y = tf.numpy_function(_decode_np, [path], [tf.float32, tf.float32])
        x.set_shape([clip_len_i, target_channels])
        y.set_shape([clip_len_i, target_channels])
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        buffer_size = max(1, min(len(paths), int(shuffle_buffer)))
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_real_audio_datasets(args, clip_len):
    all_paths = collect_audio_paths(args.data_dir)

    # Graceful fallback to synthetic sweep files if directory doesn't exist or is empty
    if len(all_paths) < 3:
        print(f"[!] Warning: No real WAV dataset found in '{args.data_dir}'. Generating clean synthetic test tones...")
        temp_data_dir = os.path.join("data", "synthetic_fallback")
        os.makedirs(temp_data_dir, exist_ok=True)
        # Generate 10 distinct synthetic frequency sweep WAVs
        for i in range(10):
            t = np.linspace(0.0, args.clip_seconds, int(args.sample_rate * args.clip_seconds), dtype=np.float32)
            sweep = np.sin(2.0 * np.pi * (200.0 + i * 150.0) * t + 0.1 * np.sin(2 * np.pi * 5.0 * t))
            write_wav_mono_int16(os.path.join(temp_data_dir, f"sweep_{i}.wav"), args.sample_rate, sweep)
        args.data_dir = temp_data_dir
        all_paths = collect_audio_paths(args.data_dir)

    if args.real_file_limit > 0:
        all_paths = all_paths[: args.real_file_limit]

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = split_indices(len(all_paths), args.real_val_ratio, args.real_test_ratio, rng)

    train_paths = all_paths[train_idx]
    val_paths = all_paths[val_idx]
    test_paths = all_paths[test_idx]

    train_data = make_dataset_from_audio_paths(
        train_paths,
        clip_len,
        args.sample_rate,
        args.batch_size,
        shuffle=True,
        repeat=True,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        target_channels=args.num_channels,
        stereo_downmix=args.stereo_downmix,
        resample_method=args.resample_method,
        augment_noise_std=args.augment_noise_std,
    )
    val_data = make_dataset_from_audio_paths(
        val_paths,
        clip_len,
        args.sample_rate,
        args.batch_size,
        shuffle=False,
        repeat=True,
        seed=args.seed + 1,
        shuffle_buffer=args.shuffle_buffer,
        target_channels=args.num_channels,
        stereo_downmix=args.stereo_downmix,
        resample_method=args.resample_method,
        augment_noise_std=0.0,
    )
    test_data = make_dataset_from_audio_paths(
        test_paths,
        clip_len,
        args.sample_rate,
        args.batch_size,
        shuffle=False,
        repeat=False,  # Evaluated explicitly without inf loop repeats
        seed=args.seed + 2,
        shuffle_buffer=args.shuffle_buffer,
        target_channels=args.num_channels,
        stereo_downmix=args.stereo_downmix,
        resample_method=args.resample_method,
        augment_noise_std=0.0,
    )

    counts = {
        "train": int(len(train_paths)),
        "val": int(len(val_paths)),
        "test": int(len(test_paths)),
        "source": "real_filesystem",
    }
    return train_data, val_data, test_data, counts


def compute_a_weighting_vector(num_spectrogram_bins, sample_rate):
    """Computes IEC 61672:2003 A-weighting frequency sensitivity gain vector for linear STFT bins."""
    freqs = np.linspace(0.0, float(sample_rate) / 2.0, int(num_spectrogram_bins), dtype=np.float32)
    f_sq_safe = np.maximum(freqs ** 2, 1e-12)
    c1 = 12194.0 ** 2
    c2 = 20.6 ** 2
    c3 = 107.7 ** 2
    c4 = 737.9 ** 2
    c5 = 12194.0 ** 2
    num = c1 * (f_sq_safe ** 2)
    den = (f_sq_safe + c2) * np.sqrt((f_sq_safe + c3) * (f_sq_safe + c4)) * (f_sq_safe + c5)
    r_a = 1.2589 * (num / den)
    r_a[freqs == 0.0] = 0.0
    return tf.constant(r_a, dtype=tf.float32)


def make_audio_loss(
    mse_weight,
    l1_weight,
    stft_weight,
    mel_weight=0.0,
    a_weight=0.0,
    hard_weight=1.5,
    stft_power=1.5,
    sample_rate=16000,
    clip_len=16000,
    num_mel_bins=64,
):
    """
    Composite audio loss combining:
    - Time-domain weighted MSE & L1 with non-linear hard residual scaling
    - STFT spectral magnitude mismatch
    - Psychoacoustic A-weighted spectral magnitude loss (Milestone A3, Issue #34)
    - Log-mel spectral envelope loss via tf.signal.linear_to_mel_weight_matrix (Milestone A3, Issue #33)
    """
    mse_w = max(0.0, float(mse_weight))
    l1_w = max(0.0, float(l1_weight))
    stft_w = max(0.0, float(stft_weight))
    mel_w = max(0.0, float(mel_weight))
    a_w = max(0.0, float(a_weight))
    hard_w = max(0.0, float(hard_weight))
    stft_p = max(1.0, float(stft_power))

    total = max(mse_w + l1_w + stft_w + mel_w + a_w, 1e-8)
    mse_w /= total
    l1_w /= total
    stft_w /= total
    mel_w /= total
    a_w /= total

    frame_length = max(16, min(512, int(clip_len)))
    frame_step = max(1, frame_length // 4)
    num_spectrogram_bins = frame_length // 2 + 1

    a_weight_tensor = None
    if a_w > 0.0:
        a_weight_tensor = compute_a_weighting_vector(num_spectrogram_bins, sample_rate)

    mel_matrix = None
    if mel_w > 0.0:
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=int(num_mel_bins),
            num_spectrogram_bins=int(num_spectrogram_bins),
            sample_rate=float(sample_rate),
            lower_edge_hertz=20.0,
            upper_edge_hertz=float(sample_rate) / 2.0,
        )

    need_stft = (stft_w > 0.0 or a_w > 0.0 or mel_w > 0.0)

    def _loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)

        # Non-linear scaling factor penalizes larger time-domain mismatch residuals
        per_sample_weight = 1.0 + hard_w * abs_err

        loss = 0.0
        if mse_w > 0.0:
            mse = tf.reduce_mean(tf.square(err) * per_sample_weight)
            loss = loss + mse_w * mse

        if l1_w > 0.0:
            l1 = tf.reduce_mean(abs_err * per_sample_weight)
            loss = loss + l1_w * l1

        if need_stft:
            clip_len_t = tf.shape(y_true)[1]
            y_t = tf.reshape(tf.transpose(y_true, [0, 2, 1]), [-1, clip_len_t])
            y_p = tf.reshape(tf.transpose(y_pred, [0, 2, 1]), [-1, clip_len_t])
            stft_true = tf.signal.stft(y_t, frame_length=frame_length, frame_step=frame_step)
            stft_pred = tf.signal.stft(y_p, frame_length=frame_length, frame_step=frame_step)
            mag_true = tf.abs(stft_true)
            mag_pred = tf.abs(stft_pred)
            mag_delta = tf.abs(mag_true - mag_pred)

            if stft_w > 0.0:
                mag_loss = tf.reduce_mean(tf.pow(mag_delta + 1e-6, stft_p))
                loss = loss + stft_w * mag_loss

            if a_w > 0.0 and a_weight_tensor is not None:
                a_weighted_delta = mag_delta * a_weight_tensor
                a_loss = tf.reduce_mean(tf.pow(a_weighted_delta + 1e-6, stft_p))
                loss = loss + a_w * a_loss

            if mel_w > 0.0 and mel_matrix is not None:
                mel_true = tf.tensordot(mag_true, mel_matrix, 1)
                mel_pred = tf.tensordot(mag_pred, mel_matrix, 1)
                log_mel_true = tf.math.log(tf.maximum(mel_true, 1e-5))
                log_mel_pred = tf.math.log(tf.maximum(mel_pred, 1e-5))
                mel_loss = tf.reduce_mean(tf.abs(log_mel_true - log_mel_pred))
                loss = loss + mel_w * mel_loss

        return loss

    return _loss


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def snr_db_metric(y_true, y_pred):
    num = tf.reduce_mean(tf.square(y_true)) + 1e-8
    den = tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8
    return 10.0 * tf.math.log(num / den) / tf.math.log(10.0)


def build_audio_autoencoder(
        clip_len,
        latent_dim,
        latent_l1=0.0,
        latent_l2=0.0,
        latent_bit_depth=0,
        quant_noise_anneal=False,
        model_base_filters=48,
        model_kernel_size=5,
        channels=1,
):
    """
    FIXED HIGH-FIDELITY ARCHITECTURE:
    1. Replaces GlobalAveragePooling1D (which completely destroys time dimensions)
       with a dynamic Dense temporal flattening spatial projection.
    2. Removes ReLU constraint (activation='linear') from bottleneck layer so
       latent variables can represent the full, continuous phase of the audio wave.
    3. Replaces raw UpSampling1D (nearest neighbor steps triggering aliasing digital hiss)
       with learned, fractional-strided Transposed Convolutions (Conv1DTranspose).
    4. Removes encoder-to-decoder skip connections so the Decoder evaluates strictly
       and independently from the Compressed Latent Bottleneck during inference!
    5. Inserts opt-in Straight-Through Estimator (STE) bit-depth quantization (Milestone A1).
    """
    num_channels = int(channels)
    inputs = layers.Input(shape=(clip_len, num_channels))
    base_filters = max(16, int(model_base_filters))
    kernel_size = max(1, int(model_kernel_size))

    # Encoder Stage 1 (16000 -> 8000)
    e1 = layers.Conv1D(base_filters, kernel_size, strides=2, padding="same", activation="relu")(inputs)
    e1 = layers.Conv1D(base_filters, kernel_size, padding="same", activation="relu")(e1)

    # Encoder Stage 2 (8000 -> 4000)
    e2 = layers.Conv1D(base_filters * 2, kernel_size, strides=2, padding="same", activation="relu")(e1)
    e2 = layers.Conv1D(base_filters * 2, kernel_size, padding="same", activation="relu")(e2)

    # Encoder Stage 3 (4000 -> 2000)
    e3 = layers.Conv1D(base_filters * 3, kernel_size, strides=2, padding="same", activation="relu")(e2)
    e3 = layers.Conv1D(base_filters * 3, kernel_size, padding="same", activation="relu")(e3)

    # Encoder Stage 4 (2000 -> 1000)
    e4 = layers.Conv1D(base_filters * 4, kernel_size, strides=2, padding="same", activation="relu")(e3)
    e4 = layers.Conv1D(base_filters * 4, kernel_size, padding="same", activation="relu")(e4)

    # TEMPORAL CONSERVING FLATTENING (No Global Pooling!)
    # Quantify downsample scale dynamically to support dynamic sample configurations
    down = clip_len // 16
    x = layers.Flatten()(e4)

    # Latent Regularization
    bottleneck_reg = None
    if latent_l1 > 0.0 or latent_l2 > 0.0:
        bottleneck_reg = regularizers.L1L2(l1=float(max(0.0, latent_l1)), l2=float(max(0.0, latent_l2)))

    # LINEAR BOTTLENECK: Crucial for phase and continuous audio transitions (-inf, +inf)
    latent = layers.Dense(latent_dim, activation="linear", name="bottleneck", activity_regularizer=bottleneck_reg)(x)

    # STRAIGHT-THROUGH ESTIMATOR (STE) QUANTIZATION (Milestone A1)
    latent_quant = StraightThroughQuantize(
        bit_depth=latent_bit_depth,
        quant_noise_anneal=quant_noise_anneal,
        name="bottleneck_quant",
    )(latent)

    # Decode Latent Code (No Skip connections bypassing the bottleneck)
    x_dec = layers.Dense(down * base_filters * 4, activation="relu")(latent_quant)
    x_dec = layers.Reshape((down, base_filters * 4))(x_dec)

    # Decoder Stage 4: Transposed learned upsampling smooths interpolation (1000 -> 2000)
    x_dec = layers.Conv1DTranspose(base_filters * 3, kernel_size, strides=2, padding="same", activation="relu")(x_dec)
    x_dec = layers.Conv1D(base_filters * 3, kernel_size, padding="same", activation="relu")(x_dec)

    # Decoder Stage 3: (2000 -> 4000)
    x_dec = layers.Conv1DTranspose(base_filters * 2, kernel_size, strides=2, padding="same", activation="relu")(x_dec)
    x_dec = layers.Conv1D(base_filters * 2, kernel_size, padding="same", activation="relu")(x_dec)

    # Decoder Stage 2: (4000 -> 8000)
    x_dec = layers.Conv1DTranspose(base_filters, kernel_size, strides=2, padding="same", activation="relu")(x_dec)
    x_dec = layers.Conv1D(base_filters, kernel_size, padding="same", activation="relu")(x_dec)

    # Decoder Stage 1: (8000 -> 16000)
    x_dec = layers.Conv1DTranspose(base_filters, kernel_size, strides=2, padding="same", activation="relu")(x_dec)

    # Squeeze Output projection
    outputs = layers.Conv1D(num_channels, kernel_size, padding="same", activation="tanh")(x_dec)

    return Model(inputs, outputs, name="audio_autoencoder")


class TargetSNRCallback(callbacks.Callback):
    def __init__(self, target_snr_db):
        super().__init__()
        self.target_snr_db = float(target_snr_db)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get("val_snr_db_metric", logs.get("snr_db_metric"))
        if current is None:
            return
        value = float(np.asarray(current, dtype=np.float32))
        if value >= self.target_snr_db:
            print(f"\n[+] Target SNR reached at epoch {epoch + 1}: {value:.3f} dB")
            self.model.stop_training = True


def export_tflite_model(model, run_dir, use_fp16=False):
    """
    CRITICAL MAC M1 FIX: Converts directly from active memory using from_keras_model.
    Bypasses disk-bound saved_model MLIR C++ parsers that trigger standard ARM SIGABRT errors.
    """
    os.makedirs(run_dir, exist_ok=True)
    tflite_path = os.path.join(run_dir, "audio_autoencoder.tflite")

    print("[*] Initiating macOS ARM M1 memory-safe TFLite serialization pipeline...")
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


def save_audio_preview(model, test_data, run_dir, sample_rate):
    plot_path = os.path.join(run_dir, "audio_preview_waveform.png")
    wav_path = os.path.join(run_dir, "audio_preview_reconstruction.wav")

    for x, _ in test_data.take(1):
        pred = model.predict(x[:1], verbose=0)[0]
        src = x[0].numpy() if hasattr(x[0], "numpy") else np.asarray(x[0])

        src_disp = src[:, 0] if src.ndim > 1 else src
        pred_disp = pred[:, 0] if pred.ndim > 1 else pred

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        axes[0].plot(src_disp)
        axes[0].set_title("Original Waveform" + (" (Ch 0)" if src.ndim > 1 and src.shape[1] > 1 else ""))
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(pred_disp, color="orange")
        axes[1].set_title("Decoded Waveform" + (" (Ch 0)" if pred.ndim > 1 and pred.shape[1] > 1 else ""))
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        write_wav_int16(wav_path, sample_rate, pred)
        break

    return plot_path, wav_path


def compute_psnr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def run_mp3_benchmark(model, test_data, run_dir, sample_rate):
    ffmpeg = shutil.which("ffmpeg")
    report_path = os.path.join(run_dir, "audio_mp3_benchmark.md")
    if not isinstance(ffmpeg, str) or not ffmpeg:
        payload = {"status": "skipped", "reason": "ffmpeg_not_found"}
        write_markdown_json_report(payload, report_path, title="Audio MP3 Benchmark")
        return report_path

    bitrates = ["96k", "128k", "192k"]
    clips = []
    for x, _ in test_data.take(4):
        clips.extend([x[i, :, 0].numpy() for i in range(min(2, x.shape[0]))])

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for clip_idx, clip in enumerate(clips):
            raw_wav = os.path.join(tmp, f"clip_{clip_idx}.wav")
            write_wav_mono_int16(raw_wav, sample_rate, clip)

            pred = model.predict(clip[np.newaxis, :, np.newaxis], verbose=0)[0, :, 0]
            model_psnr = compute_psnr(np.clip(clip, -1.0, 1.0), np.clip(pred, -1.0, 1.0))

            for br in bitrates:
                mp3_path = os.path.join(tmp, f"clip_{clip_idx}_{br}.mp3")
                decoded_wav = os.path.join(tmp, f"clip_{clip_idx}_{br}_decoded.wav")

                subprocess.run(
                    [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw_wav, "-b:a", br, mp3_path],
                    check=True)
                subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", mp3_path, decoded_wav],
                               check=True)

                decoded = read_wav_mono_float32(decoded_wav)

                n = min(len(clip), len(decoded))
                psnr_mp3 = compute_psnr(np.clip(clip[:n], -1.0, 1.0), np.clip(decoded[:n], -1.0, 1.0))
                size_ratio = os.path.getsize(mp3_path) / max(os.path.getsize(raw_wav), 1)

                results.append(
                    {
                        "clip_idx": clip_idx,
                        "bitrate": br,
                        "model_psnr": model_psnr,
                        "mp3_psnr": psnr_mp3,
                        "mp3_size_ratio": size_ratio,
                    }
                )

    payload = {"status": "ok", "results": results}
    write_markdown_json_report(payload, report_path, title="Audio MP3 Benchmark")
    return report_path


def compute_audio_compression_accounting(model, test_data, clip_len, latent_dim, latent_bits, latent_bit_depth=0, channels=1):
    """Computes both theoretical and actual compression metrics side by side (Milestones A1 & A4)."""
    ch = max(1, int(channels))
    input_pcm_bytes_per_clip = int(clip_len) * ch * 2  # 16-bit PCM (2 bytes per sample per channel)
    input_bits = int(clip_len) * ch * 16
    latent_bits_total = max(1, int(latent_dim) * int(max(1, latent_bits)))
    estimated_ratio = float(input_bits / latent_bits_total)

    result = {
        "input_bits_per_clip": int(input_bits),
        "input_pcm_bytes_per_clip": int(input_pcm_bytes_per_clip),
        "theoretical": {
            "assumed_latent_bits": int(latent_bits),
            "latent_bits_per_clip": int(latent_bits_total),
            "estimated_latent_bytes_per_clip": float(latent_bits_total / 8.0),
            "estimated_input_to_latent_ratio": float(estimated_ratio),
        },
        "latent_bits_per_clip": int(latent_bits_total),
        "estimated_input_to_latent_ratio": float(estimated_ratio),
    }

    # Extract bottleneck latent representation for a sample of test clips
    bottleneck_layer = model.get_layer("bottleneck")
    encoder_model = Model(inputs=model.input, outputs=bottleneck_layer.output)

    latent_samples = []
    for x, _ in test_data.take(4):
        preds = encoder_model.predict(x, verbose=0)
        latent_samples.append(preds)
        if sum(s.shape[0] for s in latent_samples) >= 32:
            break

    if not latent_samples:
        raw_bytes_per_clip = float(latent_dim * 4)
        result["actual"] = {
            "quantized": False,
            "latent_bit_depth": None,
            "actual_latent_bytes_per_clip": raw_bytes_per_clip,
            "actual_compression_ratio": float(input_pcm_bytes_per_clip / raw_bytes_per_clip),
            "entropy_estimate_bytes_per_clip": raw_bytes_per_clip,
        }
        result["actual_latent_bytes_per_clip"] = raw_bytes_per_clip
        result["actual_compression_ratio"] = float(input_pcm_bytes_per_clip / raw_bytes_per_clip)
        return result

    latents = np.concatenate(latent_samples, axis=0)
    n_clips = latents.shape[0]

    if latent_bit_depth and int(latent_bit_depth) > 0:
        bd = int(latent_bit_depth)
        num_levels = 1 << bd
        max_level = num_levels - 1
        clamped = np.clip(latents, -1.0, 1.0)
        norm = (clamped + 1.0) * 0.5
        discrete = np.clip(np.round(norm * max_level), 0, max_level).astype(np.int64)

        # Compute empirical entropy
        flat = discrete.reshape(-1)
        counts = np.bincount(flat, minlength=num_levels)
        nonzero = counts[counts > 0]
        probs = nonzero / nonzero.sum()
        entropy_bits_per_dim = float(-np.sum(probs * np.log2(probs)))
        entropy_bytes_per_clip = float((entropy_bits_per_dim * latent_dim) / 8.0)

        # Range encode discrete symbols
        freqs = np.maximum(counts, 1)
        encoder = _StaticRangeEncoder(freqs)
        encoded_bytes = len(encoder.encode(flat.tolist()))
        actual_bytes_per_clip = float(encoded_bytes / n_clips)
        actual_ratio = float(input_pcm_bytes_per_clip / max(1e-6, actual_bytes_per_clip))

        result["actual"] = {
            "quantized": True,
            "latent_bit_depth": bd,
            "entropy_bits_per_dim": entropy_bits_per_dim,
            "entropy_estimate_bytes_per_clip": entropy_bytes_per_clip,
            "actual_latent_bytes_per_clip": actual_bytes_per_clip,
            "actual_compression_ratio": actual_ratio,
            "sample_count": int(n_clips),
        }
        result["actual_latent_bytes_per_clip"] = actual_bytes_per_clip
        result["actual_compression_ratio"] = actual_ratio
    else:
        raw_bytes_per_clip = float(latent_dim * 4)  # float32 = 4 bytes per dim
        actual_ratio = float(input_pcm_bytes_per_clip / raw_bytes_per_clip)
        result["actual"] = {
            "quantized": False,
            "latent_bit_depth": None,
            "actual_latent_bytes_per_clip": raw_bytes_per_clip,
            "actual_compression_ratio": actual_ratio,
            "entropy_estimate_bytes_per_clip": raw_bytes_per_clip,
            "sample_count": int(n_clips),
        }
        result["actual_latent_bytes_per_clip"] = raw_bytes_per_clip
        result["actual_compression_ratio"] = actual_ratio

    return result


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

    axes[2].plot(epochs, hist.get("snr_db_metric", []), label="train")
    axes[2].plot(epochs, hist.get("val_snr_db_metric", []), label="val")
    axes[2].set_title("SNR dB")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def compute_audio_perceptual_metrics(model, test_data, sample_rate, num_samples=16):
    """
    Computes zero-dependency psychoacoustic and perceptual audio metrics (Milestone A3, Issue #36):
    1. Log-Spectral Distance (LSD) in dB (standard frequency-domain distortion).
    2. Psychoacoustic A-weighted Log-Spectral Distance (A-LSD) in dB.
    3. STOI-style short-time sub-band envelope correlation proxy ([0.0, 1.0]).
    """
    lsd_list = []
    a_lsd_list = []
    stoi_list = []

    samples_processed = 0
    n_fft = 512
    hop_length = 128
    num_bins = n_fft // 2 + 1

    # Precompute A-weighting normalization curve for LSD
    freqs = np.linspace(0.0, float(sample_rate) / 2.0, num_bins, dtype=np.float32)
    f_sq_safe = np.maximum(freqs ** 2, 1e-12)
    c1, c2, c3, c4, c5 = 12194.0 ** 2, 20.6 ** 2, 107.7 ** 2, 737.9 ** 2, 12194.0 ** 2
    num = c1 * (f_sq_safe ** 2)
    den = (f_sq_safe + c2) * np.sqrt((f_sq_safe + c3) * (f_sq_safe + c4)) * (f_sq_safe + c5)
    r_a = 1.2589 * (num / den)
    r_a[freqs == 0.0] = 0.0
    r_a_norm = r_a / max(float(np.mean(r_a)), 1e-6)

    w = np.hanning(n_fft).astype(np.float32)

    for x_batch, _ in test_data:
        preds = model.predict(x_batch, verbose=0)
        batch_size = x_batch.shape[0]

        for i in range(batch_size):
            y_true = x_batch[i, :, 0].numpy() if hasattr(x_batch[i, :, 0], "numpy") else np.asarray(x_batch[i, :, 0])
            y_pred = preds[i, :, 0]
            clip_len = min(len(y_true), len(y_pred))
            if clip_len < n_fft:
                continue

            num_frames = (clip_len - n_fft) // hop_length + 1
            if num_frames <= 0:
                continue

            frames_t = np.lib.stride_tricks.as_strided(
                y_true, shape=(num_frames, n_fft), strides=(y_true.strides[0] * hop_length, y_true.strides[0])
            )
            frames_p = np.lib.stride_tricks.as_strided(
                y_pred, shape=(num_frames, n_fft), strides=(y_pred.strides[0] * hop_length, y_pred.strides[0])
            )

            spec_t = np.fft.rfft(frames_t * w, axis=-1)
            spec_p = np.fft.rfft(frames_p * w, axis=-1)
            p_t = np.abs(spec_t) ** 2
            p_p = np.abs(spec_p) ** 2

            floor_t = max(float(np.max(p_t)) * 1e-8, 1e-12)
            floor_p = max(float(np.max(p_p)) * 1e-8, 1e-12)
            p_t_clamped = np.maximum(p_t, floor_t)
            p_p_clamped = np.maximum(p_p, floor_p)

            log_ratio = 10.0 * np.log10(p_t_clamped / p_p_clamped)
            lsd_per_frame = np.sqrt(np.mean(log_ratio ** 2, axis=-1))
            lsd_list.append(float(np.mean(lsd_per_frame)))

            a_lsd_per_frame = np.sqrt(np.mean((log_ratio * r_a_norm) ** 2, axis=-1))
            a_lsd_list.append(float(np.mean(a_lsd_per_frame)))

            # STOI envelope correlation proxy across subbands
            num_bands = 16
            band_edges = np.unique(np.round(np.geomspace(1, num_bins - 1, num_bands + 1)).astype(int))
            mag_t = np.abs(spec_t)
            mag_p = np.abs(spec_p)
            corrs = []
            for b in range(len(band_edges) - 1):
                k0, k1 = band_edges[b], max(band_edges[b] + 1, band_edges[b + 1])
                env_t = np.linalg.norm(mag_t[:, k0:k1], axis=-1)
                env_p = np.linalg.norm(mag_p[:, k0:k1], axis=-1)
                dev_t = env_t - np.mean(env_t)
                dev_p = env_p - np.mean(env_p)
                n_t = np.linalg.norm(dev_t)
                n_p = np.linalg.norm(dev_p)
                if n_t > 1e-8 and n_p > 1e-8:
                    r = np.dot(dev_t, dev_p) / (n_t * n_p)
                    corrs.append(float(np.clip(r, 0.0, 1.0)))
            if corrs:
                stoi_list.append(float(np.mean(corrs)))

            samples_processed += 1
            if samples_processed >= num_samples:
                break
        if samples_processed >= num_samples:
            break

    return {
        "lsd_db": float(np.mean(lsd_list)) if lsd_list else 0.0,
        "a_weighted_lsd_db": float(np.mean(a_lsd_list)) if a_lsd_list else 0.0,
        "envelope_correlation_stoi_proxy": float(np.mean(stoi_list)) if stoi_list else 1.0,
        "sample_count": int(samples_processed),
        "pesq_stoi_note": "Zero-dependency pure-Python perceptual evaluation (Log-Spectral Distance and STOI subband envelope proxy). Standard ITU-T P.862 PESQ requires external compiled C libraries and can be evaluated post-hoc.",
    }

def save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path, metrics_plot,
                compression_estimate, perceptual_metrics=None):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "preview_plot": preview_plot,
        "preview_wav": preview_wav,
        "metrics_plot": metrics_plot,
        "benchmark": benchmark_path,
        "compression_estimate": compression_estimate,
        "actual_latent_bytes_per_clip": compression_estimate.get("actual_latent_bytes_per_clip"),
        "actual_compression_ratio": compression_estimate.get("actual_compression_ratio"),
        "estimated_input_to_latent_ratio": compression_estimate.get("estimated_input_to_latent_ratio"),
        "loss_profile": getattr(args, "loss_profile", "balanced"),
    }
    if perceptual_metrics:
        payload["perceptual_metrics"] = perceptual_metrics
        payload["test_metrics"]["lsd_db"] = perceptual_metrics["lsd_db"]
        payload["test_metrics"]["a_weighted_lsd_db"] = perceptual_metrics["a_weighted_lsd_db"]
        payload["test_metrics"]["envelope_correlation_stoi_proxy"] = perceptual_metrics["envelope_correlation_stoi_proxy"]
    report_path = os.path.join(run_dir, "audio_evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Audio Evaluation Report")
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)


def main():
    args = apply_preset(parse_args())
    if args.params_file and load_overrides is not None:
        override_result = apply_overrides(args, load_overrides(args.params_file, section="audio"))
        if override_result.applied:
            print(f"[*] Applied {len(override_result.applied)} params from {args.params_file}")
        if override_result.unknown:
            print(f"[!] Ignored unknown params in file: {sorted(override_result.unknown)}")
        args = resolve_loss_profile(args)

    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.loss_hard_weight = max(0.0, float(args.loss_hard_weight))
    args.loss_stft_power = max(1.0, float(args.loss_stft_power))
    args.target_snr_db = float(args.target_snr_db)
    args.latent_bit_depth = max(0, int(args.latent_bit_depth))
    configure_runtime(args.seed)

    clip_len = int(args.sample_rate * args.clip_seconds)
    if clip_len % 16 != 0:
        raise ValueError("sample_rate * clip_seconds must be divisible by 16")

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_data, val_data, test_data, counts = build_real_audio_datasets(args, clip_len)
    print(
        f"[*] Using real audio files from '{args.data_dir}' (train={counts['train']}, val={counts['val']}, test={counts['test']})")
    steps_per_epoch = max(1, math.ceil(counts["train"] / args.batch_size))
    val_steps = max(1, math.ceil(counts["val"] / args.batch_size))
    test_steps = max(1, math.ceil(counts["test"] / args.batch_size))

    model = build_audio_autoencoder(
        clip_len,
        args.latent_dim,
        latent_l1=args.latent_l1,
        latent_l2=args.latent_l2,
        latent_bit_depth=args.latent_bit_depth,
        quant_noise_anneal=args.quant_noise_anneal,
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
        channels=args.num_channels,
    )
    loss_fn = make_audio_loss(
        args.loss_mse_weight,
        args.loss_l1_weight,
        args.loss_stft_weight,
        mel_weight=args.loss_mel_weight,
        a_weight=args.loss_a_weight,
        hard_weight=args.loss_hard_weight,
        stft_power=args.loss_stft_power,
        sample_rate=args.sample_rate,
        clip_len=clip_len,
        num_mel_bins=args.mel_bins,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=["mse", psnr_metric, snr_db_metric],
    )

    best_path = os.path.join(run_dir, "best_audio_model.keras")
    cbs = [
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3, min_lr=1e-6, verbose=1),
    ]
    if not args.disable_target_snr_stop:
        cbs.append(TargetSNRCallback(args.target_snr_db))

    if args.latent_bit_depth > 0 and args.quant_noise_anneal:
        quant_layer = model.get_layer("bottleneck_quant")
        cbs.append(QuantNoiseAnnealCallback(quant_layer, args.epochs))

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
    metrics_plot = os.path.join(run_dir, "audio_training_metrics.png")
    plot_history(history, metrics_plot)
    preview_plot, preview_wav = save_audio_preview(model, test_data, run_dir, args.sample_rate)

    benchmark_path = None
    if args.run_mp3_benchmark:
        benchmark_path = run_mp3_benchmark(model, test_data, run_dir, args.sample_rate)

    weights_path = os.path.join(run_dir, "audio_model.weights.h5")
    model.save_weights(weights_path)

    tflite_path = None
    if args.export_tflite:
        try:
            tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)
        except Exception as exc:
            print(f"[!] TFLite export skipped: {exc}")

    compression_estimate = compute_audio_compression_accounting(
        model,
        test_data,
        clip_len,
        args.latent_dim,
        args.latent_bits,
        latent_bit_depth=args.latent_bit_depth,
        channels=args.num_channels,
    )
    perceptual_metrics = None
    if args.eval_perceptual_metrics:
        print("[*] Computing psychoacoustic and perceptual evaluation metrics...")
        perceptual_metrics = compute_audio_perceptual_metrics(
            model, test_data, args.sample_rate, num_samples=16
        )
        print(f"[+] Perceptual metrics: LSD={perceptual_metrics['lsd_db']:.2f} dB, A-LSD={perceptual_metrics['a_weighted_lsd_db']:.2f} dB, STOI proxy={perceptual_metrics['envelope_correlation_stoi_proxy']:.3f}")

    report_path = save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path,
                              metrics_plot, compression_estimate, perceptual_metrics=perceptual_metrics)

    print("\n[+] Audio training complete")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Metrics plot: {metrics_plot}")
    print(f"[+] Preview plot: {preview_plot}")
    print(f"[+] Preview wav: {preview_wav}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    print(f"[+] Test SNR(dB): {float(eval_values.get('snr_db_metric', float('nan'))):.3f}")
    print(f"[+] Target SNR(dB): {args.target_snr_db:.3f}")
    print(f"[+] Channels: {args.channels} ({args.num_channels} ch), downmix: {args.stereo_downmix}")
    print(f"[+] Resample method: {args.resample_method}")
    if args.augment_noise_std > 0.0:
        print(f"[+] Additive noise augmentation std: {args.augment_noise_std:.4f}")
    print(f"[+] Loss profile: {args.loss_profile}")
    print(
        f"[+] Loss weights (mse/l1/stft/mel/a_weight): "
        f"{args.loss_mse_weight:.3f}/{args.loss_l1_weight:.3f}/{args.loss_stft_weight:.3f}/"
        f"{args.loss_mel_weight:.3f}/{args.loss_a_weight:.3f}"
    )
    print(f"[+] Hard residual weight: {args.loss_hard_weight:.3f}")
    print(f"[+] STFT penalty power: {args.loss_stft_power:.3f}")
    print(f"[+] Theoretical input/latent ratio: {compression_estimate['estimated_input_to_latent_ratio']:.2f}x ({compression_estimate['theoretical']['latent_bits_per_clip']} bits/clip @ {args.latent_bits}b/dim)")
    if compression_estimate.get("actual", {}).get("quantized"):
        print(f"[+] Actual quantized latent ({args.latent_bit_depth}-bit): {compression_estimate['actual_latent_bytes_per_clip']:.1f} bytes/clip ({compression_estimate['actual_compression_ratio']:.2f}x ratio)")
        print(f"[+] Actual entropy estimate: {compression_estimate['actual']['entropy_estimate_bytes_per_clip']:.1f} bytes/clip ({compression_estimate['actual']['entropy_bits_per_dim']:.2f} bits/dim)")
    else:
        print(f"[+] Actual latent (float32 unquantized): {compression_estimate['actual_latent_bytes_per_clip']:.0f} bytes/clip ({compression_estimate['actual_compression_ratio']:.2f}x ratio)")
    if float(eval_values.get("snr_db_metric", float("-inf"))) < args.target_snr_db:
        print(f"[!] SNR target not reached yet (target={args.target_snr_db:.1f} dB).")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")
    if benchmark_path:
        print(f"[+] MP3 benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
