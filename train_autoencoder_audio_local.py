#!/usr/bin/env python3
"""Local audio autoencoder trainer with MP3 benchmark and TFLite export."""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Must be set before importing TensorFlow to suppress verbose C++ INFO logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers

from report_markdown import write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train an audio autoencoder locally.")
    parser.add_argument("--output-root", type=str, default="models/audio_local_run", help="Output directory for artifacts")
    parser.add_argument("--preset", type=str, default="m1-air-balanced", choices=["m1-air-fast", "m1-air-balanced", "m1-air-quality", "custom"])
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--clip-seconds", type=float, default= 1.0, help="Audio clip duration in seconds")
    parser.add_argument("--latent-dim", type=int, default=192, help="Latent bottleneck size")
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
    parser.add_argument("--real-val-ratio", type=float, default=0.15, help="Validation split ratio for real audio files")
    parser.add_argument("--real-test-ratio", type=float, default=0.15, help="Test split ratio for real audio files")
    parser.add_argument("--real-file-limit", type=int, default=0, help="Optional cap for discovered real audio files (0 = all)")
    parser.add_argument("--synthetic-only", action="store_true", help="Disable real-data loading and force synthetic generation")
    parser.add_argument("--train-samples", type=int, default=9000, help="Synthetic train sample count")
    parser.add_argument("--val-samples", type=int, default=3000, help="Synthetic validation sample count")
    parser.add_argument("--test-samples", type=int, default=3000, help="Synthetic test sample count")
    parser.add_argument("--target-psnr", type=float, default=30.0, help="Validation PSNR target")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1024,
        help="Shuffle buffer size for synthetic training data (lower values start faster)",
    )
    parser.add_argument("--export-tflite", dest="export_tflite", action="store_true", help="Export .tflite model")
    parser.add_argument("--no-export-tflite", dest="export_tflite", action="store_false", help="Disable .tflite export")
    parser.add_argument("--tflite-fp16", action="store_true", help="Export float16 optimized tflite")
    parser.add_argument("--run-mp3-benchmark", action="store_true", help="Benchmark recon quality against MP3")
    parser.add_argument(
        "--synthetic-profile",
        type=str,
        default="expansive",
        choices=["legacy", "expansive"],
        help="Synthetic audio generation profile",
    )
    parser.set_defaults(export_tflite=True)
    return parser.parse_args()


def apply_preset(args):
    preset_map = {
        "m1-air-fast": {"batch_size": 12, "latent_dim": 160, "epochs": 14, "lr": 1.8e-4},
        "m1-air-balanced": {"batch_size": 12, "latent_dim": 224, "epochs": 24, "lr": 1.2e-4},
        "m1-air-quality": {"batch_size": 8, "latent_dim": 320, "epochs": 36, "lr": 8e-5},
    }
    if args.preset in preset_map:
        for k, v in preset_map[args.preset].items():
            setattr(args, k, v)
    return args


def synthesize_audio_clip_legacy(sample_rate, clip_len, rng):
    t = np.linspace(0.0, clip_len / sample_rate, clip_len, endpoint=False, dtype=np.float32)

    base_freq = rng.uniform(80.0, 1200.0)
    harmonic = rng.uniform(1.8, 3.5)
    fm_rate = rng.uniform(0.3, 5.0)
    fm_depth = rng.uniform(2.0, 30.0)

    signal = 0.55 * np.sin(2.0 * np.pi * base_freq * t)
    signal += 0.25 * np.sin(2.0 * np.pi * base_freq * harmonic * t)
    signal += 0.10 * np.sin(2.0 * np.pi * (base_freq + fm_depth * np.sin(2.0 * np.pi * fm_rate * t)) * t)

    chirp_start = rng.uniform(50.0, 500.0)
    chirp_end = rng.uniform(1000.0, 5000.0)
    k = (chirp_end - chirp_start) / max(t[-1], 1e-4)
    signal += 0.10 * np.sin(2.0 * np.pi * (chirp_start * t + 0.5 * k * t * t))

    envelope = 0.65 + 0.35 * np.sin(2.0 * np.pi * rng.uniform(0.2, 1.1) * t + rng.uniform(0, 2 * np.pi))
    signal *= envelope

    pinkish_noise = rng.normal(0.0, 0.02, size=clip_len).astype(np.float32)
    signal += pinkish_noise
    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)


def _random_envelope(clip_len, rng):
    points = int(rng.integers(5, 10))
    x = np.linspace(0, clip_len - 1, points, dtype=np.float32)
    y = rng.uniform(0.2, 1.0, size=points).astype(np.float32)
    y[0] *= rng.uniform(0.2, 0.6)
    y[-1] *= rng.uniform(0.2, 0.7)
    return np.interp(np.arange(clip_len, dtype=np.float32), x, y).astype(np.float32)


def _smoothed_noise(clip_len, rng, scale):
    white = rng.normal(0.0, 1.0, size=clip_len).astype(np.float32)
    kernel_len = int(rng.integers(7, 41))
    if kernel_len % 2 == 0:
        kernel_len += 1
    kernel = np.hanning(kernel_len).astype(np.float32)
    kernel /= np.sum(kernel) + 1e-8
    return (np.convolve(white, kernel, mode="same") * scale).astype(np.float32)


def synthesize_audio_clip_expansive(sample_rate, clip_len, rng):
    t = np.arange(clip_len, dtype=np.float32) / float(sample_rate)
    signal = np.zeros(clip_len, dtype=np.float32)

    families = ["harmonic", "chirp", "am", "fm", "pulses"]
    weights = np.array([0.28, 0.20, 0.19, 0.19, 0.14], dtype=np.float64)
    weights /= np.sum(weights)
    component_count = int(rng.integers(2, 5))
    selected = rng.choice(families, size=component_count, replace=False, p=weights)

    for family in selected:
        gain = float(rng.uniform(0.12, 0.42))
        if family == "harmonic":
            f0 = float(rng.uniform(60.0, 1100.0))
            partials = int(rng.integers(2, 8))
            comp = np.zeros_like(signal)
            for p in range(1, partials + 1):
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                amp = (1.0 / p) * float(rng.uniform(0.7, 1.3))
                comp += amp * np.sin(2.0 * np.pi * f0 * p * t + phase)
            signal += gain * comp
        elif family == "chirp":
            start = float(rng.uniform(40.0, 900.0))
            end = float(rng.uniform(900.0, 5000.0))
            dur = max(t[-1], 1e-4)
            k = (end - start) / dur
            signal += gain * np.sin(2.0 * np.pi * (start * t + 0.5 * k * t * t))
        elif family == "am":
            fc = float(rng.uniform(90.0, 2000.0))
            fm = float(rng.uniform(0.5, 12.0))
            depth = float(rng.uniform(0.2, 0.95))
            mod = (1.0 - depth) + depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * fm * t + rng.uniform(0, 2 * np.pi)))
            signal += gain * mod * np.sin(2.0 * np.pi * fc * t + rng.uniform(0, 2 * np.pi))
        elif family == "fm":
            carrier = float(rng.uniform(90.0, 1200.0))
            mod_f = float(rng.uniform(0.4, 15.0))
            mod_d = float(rng.uniform(5.0, 90.0))
            inst_freq = carrier + mod_d * np.sin(2.0 * np.pi * mod_f * t + rng.uniform(0, 2 * np.pi))
            signal += gain * np.sin(2.0 * np.pi * inst_freq * t)
        else:
            comp = np.zeros_like(signal)
            pulse_count = int(rng.integers(4, 20))
            for _ in range(pulse_count):
                center = int(rng.integers(0, clip_len))
                width = int(rng.integers(max(2, clip_len // 200), max(6, clip_len // 35)))
                left = max(0, center - width)
                right = min(clip_len, center + width)
                if right <= left:
                    continue
                window = np.hanning(right - left).astype(np.float32)
                tone = np.sin(2.0 * np.pi * rng.uniform(120.0, 3500.0) * t[left:right] + rng.uniform(0, 2 * np.pi))
                comp[left:right] += window * tone
            signal += gain * comp

    signal *= _random_envelope(clip_len, rng)
    signal += _smoothed_noise(clip_len, rng, scale=float(rng.uniform(0.004, 0.03)))
    signal += rng.normal(0.0, float(rng.uniform(0.0008, 0.008)), size=clip_len).astype(np.float32)

    rms = float(np.sqrt(np.mean(signal * signal) + 1e-8))
    target_rms = float(rng.uniform(0.12, 0.30))
    signal = signal * (target_rms / max(rms, 1e-6))
    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)


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


def build_dataset(sample_rate, clip_len, sample_count, batch_size, seed, shuffle, repeat=False, profile="legacy", shuffle_buffer=1024):
    rng = np.random.default_rng(seed)
    synth_fn = synthesize_audio_clip_expansive if profile == "expansive" else synthesize_audio_clip_legacy

    def gen():
        for _ in range(sample_count):
            clip = synth_fn(sample_rate, clip_len, rng)
            clip = np.expand_dims(clip, axis=-1)
            yield clip, clip

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(clip_len, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(clip_len, 1), dtype=tf.float32),
        ),
    )
    if shuffle:
        buffer_size = max(1, min(sample_count, int(shuffle_buffer)))
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        # Keep input stream available across epochs when fit/evaluate uses explicit step counts.
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
    val_idx = idx[train_n : train_n + val_n]
    test_idx = idx[train_n + val_n :]
    return train_idx, val_idx, test_idx


def make_dataset_from_audio_paths(paths, clip_len, sample_rate, batch_size, shuffle, repeat, seed, shuffle_buffer):
    clip_len_i = int(clip_len)
    random_crop = bool(shuffle)
    rng = np.random.default_rng(seed)

    def _decode_np(path_bytes):
        path = path_bytes.decode("utf-8")
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            src_rate = int(wf.getframerate())
            frames = wf.getnframes()
            data = wf.readframes(frames)

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
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

        clip = np.asarray(audio[start : start + clip_len_i], dtype=np.float32)
        clip = np.clip(clip, -1.0, 1.0)
        return np.expand_dims(clip, axis=-1)

    def _decode(path):
        clip = tf.numpy_function(_decode_np, [path], tf.float32)
        clip = tf.ensure_shape(clip, [clip_len_i, 1])
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
    if args.real_file_limit > 0:
        all_paths = all_paths[: args.real_file_limit]
    if len(all_paths) < 3:
        return None

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
        repeat=True,
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


def stft_mag_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    clip_len = tf.shape(y_true)[1]
    frame_length = tf.maximum(16, tf.minimum(512, clip_len))
    frame_step = tf.maximum(1, frame_length // 4)
    stft_true = tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step)
    stft_pred = tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step)
    return tf.reduce_mean(tf.abs(tf.abs(stft_true) - tf.abs(stft_pred)))


def audio_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    mag = stft_mag_loss(y_true, y_pred)
    return 0.70 * mse + 0.15 * l1 + 0.15 * mag


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def snr_db_metric(y_true, y_pred):
    num = tf.reduce_mean(tf.square(y_true)) + 1e-8
    den = tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8
    return 10.0 * tf.math.log(num / den) / tf.math.log(10.0)


def build_audio_autoencoder(clip_len, latent_dim):
    inputs = layers.Input(shape=(clip_len, 1))

    e1 = layers.Conv1D(48, 9, strides=2, padding="same", activation="relu")(inputs)
    e1 = layers.Conv1D(48, 5, padding="same", activation="relu")(e1)

    e2 = layers.Conv1D(96, 7, strides=2, padding="same", activation="relu")(e1)
    e2 = layers.Conv1D(96, 5, padding="same", activation="relu")(e2)

    e3 = layers.Conv1D(160, 5, strides=2, padding="same", activation="relu")(e2)
    e3 = layers.Conv1D(160, 3, padding="same", activation="relu")(e3)

    b = layers.Conv1D(224, 5, strides=2, padding="same", activation="relu")(e3)
    b = layers.Conv1D(224, 3, padding="same", activation="relu")(b)

    x = layers.GlobalAveragePooling1D()(b)
    latent = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    down = clip_len // 16
    x = layers.Dense(down * 224, activation="relu")(latent)
    x = layers.Reshape((down, 224))(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, e3])
    x = layers.Conv1D(160, 5, padding="same", activation="relu")(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, e2])
    x = layers.Conv1D(96, 5, padding="same", activation="relu")(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, e1])
    x = layers.Conv1D(48, 5, padding="same", activation="relu")(x)

    x = layers.UpSampling1D(2)(x)
    outputs = layers.Conv1D(1, 9, padding="same", activation="tanh")(x)

    return Model(inputs, outputs, name="audio_autoencoder")


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
    tflite_path = os.path.join(run_dir, "audio_autoencoder.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
        axes[0].set_title("Original")
        axes[1].plot(pred)
        axes[1].set_title("Decoded")
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

                subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", raw_wav, "-b:a", br, mp3_path], check=True)
                subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", mp3_path, decoded_wav], check=True)

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


def save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path, metrics_plot):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in eval_values.items()},
        "preview_plot": preview_plot,
        "preview_wav": preview_wav,
        "metrics_plot": metrics_plot,
        "benchmark": benchmark_path,
    }
    report_path = os.path.join(run_dir, "audio_evaluation_report.md")
    write_markdown_json_report(payload, report_path, title="Audio Evaluation Report")
    return report_path


def configure_runtime(seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.keras.utils.set_random_seed(seed)


def main():
    args = apply_preset(parse_args())
    configure_runtime(args.seed)

    clip_len = int(args.sample_rate * args.clip_seconds)
    if clip_len % 16 != 0:
        raise ValueError("sample_rate * clip_seconds must be divisible by 16")

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    real_data = None if args.synthetic_only else build_real_audio_datasets(args, clip_len)
    if real_data is not None:
        train_data, val_data, test_data, counts = real_data
        print(f"[*] Using real audio files from '{args.data_dir}' (train={counts['train']}, val={counts['val']}, test={counts['test']})")
        steps_per_epoch = max(1, math.ceil(counts["train"] / args.batch_size))
        val_steps = max(1, math.ceil(counts["val"] / args.batch_size))
        test_steps = max(1, math.ceil(counts["test"] / args.batch_size))
    else:
        print("[!] Real audio files not found or disabled; falling back to synthetic generation.")
        train_data = build_dataset(
            args.sample_rate,
            clip_len,
            args.train_samples,
            args.batch_size,
            args.seed,
            shuffle=True,
            repeat=True,
            profile=args.synthetic_profile,
            shuffle_buffer=args.shuffle_buffer,
        )
        val_data = build_dataset(
            args.sample_rate,
            clip_len,
            args.val_samples,
            args.batch_size,
            args.seed + 1,
            shuffle=False,
            repeat=True,
            profile=args.synthetic_profile,
            shuffle_buffer=args.shuffle_buffer,
        )
        test_data = build_dataset(
            args.sample_rate,
            clip_len,
            args.test_samples,
            args.batch_size,
            args.seed + 2,
            shuffle=False,
            repeat=True,
            profile=args.synthetic_profile,
            shuffle_buffer=args.shuffle_buffer,
        )

        steps_per_epoch = max(1, math.ceil(args.train_samples / args.batch_size))
        val_steps = max(1, math.ceil(args.val_samples / args.batch_size))
        test_steps = max(1, math.ceil(args.test_samples / args.batch_size))

    model = build_audio_autoencoder(clip_len, args.latent_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=audio_loss,
        metrics=["mse", psnr_metric, snr_db_metric],
    )

    best_path = os.path.join(run_dir, "best_audio_model.keras")
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
        tflite_path = export_tflite_model(model, run_dir, use_fp16=args.tflite_fp16)

    report_path = save_report(args, history, eval_values, run_dir, preview_plot, preview_wav, benchmark_path, metrics_plot)

    print("\n[+] Audio training complete")
    print(f"[+] Weights: {weights_path}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Metrics plot: {metrics_plot}")
    print(f"[+] Preview plot: {preview_plot}")
    print(f"[+] Preview wav: {preview_wav}")
    print(f"[+] Test PSNR: {float(eval_values.get('psnr_metric', float('nan'))):.3f}")
    if float(eval_values.get("psnr_metric", 0.0)) < args.target_psnr:
        print(f"[!] PSNR target not reached yet (target={args.target_psnr:.1f}).")
    if tflite_path:
        print(f"[+] TFLite: {tflite_path}")
    if benchmark_path:
        print(f"[+] MP3 benchmark: {benchmark_path}")


if __name__ == "__main__":
    main()


