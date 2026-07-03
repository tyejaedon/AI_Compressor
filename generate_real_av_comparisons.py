#!/usr/bin/env python3
"""Generate real-data comparison PNGs for audio and video autoencoders."""

import argparse
import os
import subprocess
import wave
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from report_markdown import write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Generate real-data comparisons for audio/video models.")
    parser.add_argument(
        "--audio-model",
        type=str,
        default="models/audio_realdata_opt_smoke/20260629_145612/best_audio_model.keras",
        help="Path to trained audio Keras model",
    )
    parser.add_argument(
        "--video-model",
        type=str,
        default="models/video_realdata_opt_smoke/20260629_145842/best_video_model.keras",
        help="Path to trained video Keras model",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/AudioData/ESC-50-master/audio",
        help="Directory with real WAV files",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/VIDEO DATA",
        help="Directory with real MP4 files",
    )
    parser.add_argument("--audio-count", type=int, default=6, help="Number of audio comparisons")
    parser.add_argument("--video-count", type=int, default=4, help="Number of video comparisons")
    parser.add_argument("--video-fps", type=int, default=10, help="Decode fps for video samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/real_data_comparisons",
        help="Root output folder",
    )
    return parser.parse_args()


def collect_files(root_dir, extensions):
    root = Path(root_dir)
    if not root.exists():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def read_wav_mono(path):
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = int(wf.getframerate())
        frames = wf.getnframes()
        data = wf.readframes(frames)

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_np(audio, src_rate, dst_rate):
    if src_rate == dst_rate or len(audio) <= 1:
        return np.asarray(audio, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
    dst_len = max(1, int(round(len(audio) * (dst_rate / float(src_rate)))))
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def prepare_audio_clip(audio, clip_len, rng):
    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)), mode="constant")
    max_start = max(0, len(audio) - clip_len)
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    clip = np.asarray(audio[start : start + clip_len], dtype=np.float32)
    return np.clip(clip, -1.0, 1.0)


def save_audio_pair_png(name, original, recon, psnr_db, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    axes[0, 0].plot(original)
    axes[0, 0].set_title("Original waveform")
    axes[0, 1].plot(recon)
    axes[0, 1].set_title("Reconstructed waveform")

    axes[1, 0].specgram(original, NFFT=256, Fs=1.0, noverlap=128)
    axes[1, 0].set_title("Original spectrogram")
    axes[1, 1].specgram(recon, NFFT=256, Fs=1.0, noverlap=128)
    axes[1, 1].set_title("Reconstructed spectrogram")

    fig.suptitle(f"{name} | PSNR={psnr_db:.2f} dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_psnr(x, y):
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def resolve_ffmpeg_binary():
    ffmpeg_bin = shutil_which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_bin:
            return ffmpeg_bin
    except Exception:
        pass
    return None


def shutil_which(cmd):
    from shutil import which

    return which(cmd)


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
    return arr.reshape(frame_count, height, width, 3).astype(np.float32) / 255.0


def save_video_pair_png(name, original, recon, psnr_db, out_path):
    frames = min(4, original.shape[0])
    fig, axes = plt.subplots(2, frames, figsize=(3 * frames, 6))
    if frames == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(frames):
        axes[0, i].imshow(np.clip(original[i], 0.0, 1.0))
        axes[0, i].set_title(f"Original t={i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(recon[i], 0.0, 1.0))
        axes[1, i].set_title(f"Reconstructed t={i}")
        axes[1, i].axis("off")

    fig.suptitle(f"{name} | PSNR={psnr_db:.2f} dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_audio_comparisons(args, run_dir, rng):
    audio_dir = os.path.join(run_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    model = tf.keras.models.load_model(args.audio_model, compile=False)
    clip_len = int(model.input_shape[1])
    sample_rate = 16000

    paths = collect_files(args.audio_dir, {".wav", ".wave"})
    if not paths:
        return {"status": "skipped", "reason": "no_audio_files"}

    count = min(args.audio_count, len(paths))
    picked = [paths[i] for i in rng.choice(len(paths), size=count, replace=False)]
    samples = []

    for i, path in enumerate(picked, start=1):
        audio, src_rate = read_wav_mono(path)
        audio = resample_np(audio, src_rate, sample_rate)
        clip = prepare_audio_clip(audio, clip_len, rng)
        pred = model.predict(clip[np.newaxis, :, np.newaxis], verbose=0)[0, :, 0]
        psnr = compute_psnr(np.clip(clip, -1.0, 1.0), np.clip(pred, -1.0, 1.0))

        out_png = os.path.join(audio_dir, f"audio_comparison_{i:02d}.png")
        save_audio_pair_png(Path(path).name, clip, pred, psnr, out_png)
        samples.append(
            {
                "source_path": path,
                "name": Path(path).name,
                "psnr": psnr,
                "comparison_png": out_png,
            }
        )

    return {"status": "ok", "model": args.audio_model, "samples": samples}


def generate_video_comparisons(args, run_dir, rng):
    video_dir = os.path.join(run_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    ffmpeg_bin = resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        return {"status": "skipped", "reason": "ffmpeg_not_available"}

    model = tf.keras.models.load_model(args.video_model, compile=False)
    _, frames, height, width, _ = model.input_shape

    paths = collect_files(args.video_dir, {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
    if not paths:
        return {"status": "skipped", "reason": "no_video_files"}

    count = min(args.video_count, len(paths))
    picked = [paths[i] for i in rng.choice(len(paths), size=count, replace=False)]
    samples = []

    for i, path in enumerate(picked, start=1):
        decoded = decode_video_frames(ffmpeg_bin, path, int(width), int(height), args.video_fps)
        if len(decoded) < int(frames):
            continue
        max_start = max(0, len(decoded) - int(frames))
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        clip = decoded[start : start + int(frames)]

        pred = model.predict(clip[np.newaxis, ...], verbose=0)[0]
        psnr = compute_psnr(np.clip(clip, 0.0, 1.0), np.clip(pred, 0.0, 1.0))

        out_png = os.path.join(video_dir, f"video_comparison_{i:02d}.png")
        save_video_pair_png(Path(path).name, clip, pred, psnr, out_png)
        samples.append(
            {
                "source_path": path,
                "name": Path(path).name,
                "psnr": psnr,
                "comparison_png": out_png,
            }
        )

    if not samples:
        return {"status": "skipped", "reason": "insufficient_decodable_video_clips"}

    return {"status": "ok", "model": args.video_model, "samples": samples, "ffmpeg": ffmpeg_bin}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S") + "_av")
    os.makedirs(run_dir, exist_ok=True)

    audio_result = generate_audio_comparisons(args, run_dir, rng)
    video_result = generate_video_comparisons(args, run_dir, rng)

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "audio": audio_result,
        "video": video_result,
    }
    report_path = os.path.join(run_dir, "av_comparison_report.md")
    write_markdown_json_report(report, report_path, title="Audio Video Comparison Report")

    print(run_dir)
    print(report_path)


if __name__ == "__main__":
    main()

