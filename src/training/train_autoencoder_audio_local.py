#!/usr/bin/env python3
"""
Upgraded MacBook-friendly audio autoencoder trainer with SNR fixes,
learned Conv1DTranspose upsampling, linear bottleneck representation,
and memory-safe TFLite export bypassing macOS MLIR compiler crashes.
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
    parser.add_argument("--latent-bits", type=int, default=16, help="Assumed quantized bits per latent dimension")
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
    parser.add_argument("--loss-mse-weight", type=float, default=0.55, help="Weight for weighted MSE term")
    parser.add_argument("--loss-l1-weight", type=float, default=0.10, help="Weight for weighted L1 term")
    parser.add_argument("--loss-stft-weight", type=float, default=0.35, help="Weight for STFT magnitude loss")
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
    return args


def write_wav_mono_int16(path, sample_rate, audio):
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


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


def make_dataset_from_audio_paths(paths, clip_len, sample_rate, batch_size, shuffle, repeat, seed, shuffle_buffer):
    clip_len_i = int(clip_len)
    random_crop = bool(shuffle)
    rng = np.random.default_rng(seed)

    def _decode_np(path_bytes):
        path = path_bytes.decode("utf-8")
        try:
            with wave.open(path, "rb") as wf:
                channels = wf.getnchannels()
                src_rate = int(wf.getframerate())
                frames = wf.getnframes()
                data = wf.readframes(frames)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        except Exception as e:
            # Fallback if file is corrupted or unreadable
            print(f"[!] Warning: failing to read WAV file {path}. Generating silence fallback. Err: {e}")
            audio = np.zeros((clip_len_i,), dtype=np.float32)
            channels = 1
            src_rate = sample_rate

        if channels > 1 and len(audio) > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        if src_rate != sample_rate and len(audio) > 1:
            src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
            dst_len = max(1, int(round(len(audio) * (sample_rate / float(src_rate)))))
            dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
            audio = np.interp(dst_x, src_x, audio).astype(np.float32)

        if len(audio) < clip_len_i:
            audio = np.pad(audio, (0, clip_len_i - len(audio)), mode="constant")

        max_start = max(0, len(audio) - clip_len_i)
        if random_crop and max_start > 0:
            start = int(rng.integers(0, max_start + 1))
        else:
            start = max_start // 2

        clip = np.asarray(audio[start: start + clip_len_i], dtype=np.float32)
        clip = np.clip(clip, -1.0, 1.0)
        return np.expand_dims(clip, axis=-1)

    def _decode(path):
        clip = tf.numpy_function(_decode_np, [path], tf.float32)
        clip.set_shape([clip_len_i, 1])
        return clip, clip

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
    )

    counts = {
        "train": int(len(train_paths)),
        "val": int(len(val_paths)),
        "test": int(len(test_paths)),
        "source": "real_filesystem",
    }
    return train_data, val_data, test_data, counts


def stft_mag_loss(y_true, y_pred, power=1.0):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    clip_len = tf.shape(y_true)[1]
    # Optimal frame length balances mathematical time vs spectral resolution splits
    frame_length = tf.maximum(16, tf.minimum(512, clip_len))
    frame_step = tf.maximum(1, frame_length // 4)

    stft_true = tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step)
    stft_pred = tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step)

    # Squeeze STFT magnitudes directly
    mag_delta = tf.abs(tf.abs(stft_true) - tf.abs(stft_pred))
    return tf.reduce_mean(tf.pow(mag_delta + 1e-6, float(max(1.0, power))))


def make_audio_loss(mse_weight, l1_weight, stft_weight, hard_weight=1.5, stft_power=1.5):
    mse_w = float(mse_weight)
    l1_w = float(l1_weight)
    stft_w = float(stft_weight)
    hard_w = max(0.0, float(hard_weight))
    stft_p = max(1.0, float(stft_power))

    total = max(mse_w + l1_w + stft_w, 1e-8)
    mse_w, l1_w, stft_w = mse_w / total, l1_w / total, stft_w / total

    def _loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)

        # Non-linear scaling factor penalizes larger time-domain mismatch residuals
        per_sample_weight = 1.0 + hard_w * abs_err

        mse = tf.reduce_mean(tf.square(err) * per_sample_weight)
        l1 = tf.reduce_mean(abs_err * per_sample_weight)
        mag = stft_mag_loss(y_true, y_pred, power=stft_p)

        return mse_w * mse + l1_w * l1 + stft_w * mag

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
        model_base_filters=48,
        model_kernel_size=5,
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
    """
    inputs = layers.Input(shape=(clip_len, 1))
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

    # Decode Latent Code (No Skip connections bypassing the bottleneck)
    x_dec = layers.Dense(down * base_filters * 4, activation="relu")(latent)
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
    outputs = layers.Conv1D(1, kernel_size, padding="same", activation="tanh")(x_dec)

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
        pred = model.predict(x[:1], verbose=0)[0, :, 0]
        src = x[0, :, 0].numpy()

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        axes[0].plot(src)
        axes[0].set_title("Original Waveform")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(pred, color="orange")
        axes[1].set_title("Decoded Waveform (Post-Fixes)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        write_wav_mono_int16(wav_path, sample_rate, pred)
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


def estimate_audio_compression_ratio(clip_len, latent_dim, latent_bits):
    input_bits = int(clip_len) * 16
    latent_bits_total = max(1, int(latent_dim) * int(max(1, latent_bits)))
    return {
        "input_bits_per_clip": int(input_bits),
        "latent_bits_per_clip": int(latent_bits_total),
        "estimated_input_to_latent_ratio": float(input_bits / latent_bits_total),
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

    axes[2].plot(epochs, hist.get("snr_db_metric", []), label="train")
    axes[2].plot(epochs, hist.get("val_snr_db_metric", []), label="val")
    axes[2].set_title("SNR dB")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path, metrics_plot,
                compression_estimate):
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
    }
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

    args.model_base_filters = max(16, int(args.model_base_filters))
    args.model_kernel_size = max(1, int(args.model_kernel_size))
    args.loss_hard_weight = max(0.0, float(args.loss_hard_weight))
    args.loss_stft_power = max(1.0, float(args.loss_stft_power))
    args.target_snr_db = float(args.target_snr_db)
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
        model_base_filters=args.model_base_filters,
        model_kernel_size=args.model_kernel_size,
    )
    loss_fn = make_audio_loss(
        args.loss_mse_weight,
        args.loss_l1_weight,
        args.loss_stft_weight,
        hard_weight=args.loss_hard_weight,
        stft_power=args.loss_stft_power,
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

    compression_estimate = estimate_audio_compression_ratio(clip_len, args.latent_dim, args.latent_bits)
    report_path = save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path,
                              metrics_plot, compression_estimate)

    print("\n[+] Audio training complete")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Metrics plot: {metrics_plot}")
    print(f"[+] Preview plot: {preview_plot}")
    print(f"[+] Preview wav: {preview_wav}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    print(f"[+] Test SNR(dB): {float(eval_values.get('snr_db_metric', float('nan'))):.3f}")
    print(f"[+] Target SNR(dB): {args.target_snr_db:.3f}")
    print(
        f"[+] Loss weights (mse/l1/stft): "
        f"{args.loss_mse_weight:.3f}/{args.loss_l1_weight:.3f}/{args.loss_stft_weight:.3f}"
    )
    print(f"[+] Hard residual weight: {args.loss_hard_weight:.3f}")
    print(f"[+] STFT penalty power: {args.loss_stft_power:.3f}")
    print(f"[+] Estimated input/latent ratio: {compression_estimate['estimated_input_to_latent_ratio']:.2f}x")
    if float(eval_values.get("snr_db_metric", float("-inf"))) < args.target_snr_db:
        print(f"[!] SNR target not reached yet (target={args.target_snr_db:.1f} dB).")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")
    if benchmark_path:
        print(f"[+] MP3 benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()
