#!/usr/bin/env python3
"""Prepare mixed-codec audio into canonical mono PCM16 WAV or float32 NPY with metadata sidecars."""

import argparse
import json
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from report_markdown import write_markdown_json_report


SUPPORTED_EXTS = {
    ".wav",
    ".wave",
    ".mp3",
    ".mpeg",
    ".mpg",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert mixed audio codecs into a canonical training dataset.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing source audio files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write canonical dataset")
    parser.add_argument("--target-sample-rate", type=int, default=16000, help="Target sample rate for converted audio")
    parser.add_argument("--output-format", type=str, choices=["wav", "npy"], default="wav", help="Canonical output format")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "peak", "rms", "rms_peak"],
        default="rms_peak",
        help="Normalization mode applied after decoding",
    )
    parser.add_argument("--target-peak", type=float, default=0.95, help="Target peak amplitude for peak/rms_peak normalization")
    parser.add_argument("--target-rms", type=float, default=0.12, help="Target RMS for rms/rms_peak normalization")
    parser.add_argument("--remove-dc", action="store_true", help="Subtract mean from each clip before normalization")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of discovered files (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files")
    return parser.parse_args()


def collect_audio_paths(input_dir):
    root = Path(input_dir)
    if not root.exists():
        return []
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths)


def read_wav_mono_float32(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        src_rate = int(wf.getframerate())
        frames = wf.getnframes()
        data = wf.readframes(frames)

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, src_rate


def resample_linear(audio, src_rate, target_rate):
    if src_rate == target_rate or len(audio) <= 1:
        return audio.astype(np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float32)
    dst_len = max(1, int(round(len(audio) * (target_rate / float(src_rate)))))
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def decode_with_ffmpeg(path, ffmpeg_bin, target_rate):
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr.strip()}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced empty decode for {path}")
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def get_source_metadata(path, ffprobe_bin):
    if not ffprobe_bin:
        return {
            "format": {"format_name": None, "duration": None, "bit_rate": None},
            "stream": {"codec_name": None, "sample_rate": None, "channels": None},
            "ffprobe_available": False,
        }

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=format_name,duration,bit_rate",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,codec_type",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "format": {"format_name": None, "duration": None, "bit_rate": None},
            "stream": {"codec_name": None, "sample_rate": None, "channels": None},
            "ffprobe_available": True,
            "ffprobe_error": proc.stderr.strip(),
        }

    payload = json.loads(proc.stdout or "{}")
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    return {
        "format": {
            "format_name": fmt.get("format_name"),
            "duration": float(fmt["duration"]) if fmt.get("duration") not in (None, "") else None,
            "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") not in (None, "") else None,
        },
        "stream": {
            "codec_name": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") not in (None, "") else None,
            "channels": int(audio_stream["channels"]) if audio_stream.get("channels") not in (None, "") else None,
        },
        "ffprobe_available": True,
    }


def normalize_audio(audio, mode, target_peak, target_rms, remove_dc):
    audio = np.asarray(audio, dtype=np.float32)
    sample_count = int(audio.shape[0])

    if remove_dc and sample_count > 0:
        audio = audio - float(np.mean(audio))

    pre_peak = float(np.max(np.abs(audio))) if sample_count > 0 else 0.0
    pre_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if sample_count > 0 else 0.0

    gain = 1.0
    if sample_count > 0 and mode == "peak" and pre_peak > 0:
        gain = target_peak / pre_peak
    elif sample_count > 0 and mode == "rms" and pre_rms > 0:
        gain = target_rms / pre_rms
    elif sample_count > 0 and mode == "rms_peak":
        if pre_rms > 0:
            gain = target_rms / pre_rms
        audio = audio * gain
        peak_after_rms = float(np.max(np.abs(audio))) if sample_count > 0 else 0.0
        if peak_after_rms > target_peak > 0:
            audio = audio * (target_peak / peak_after_rms)
            gain *= target_peak / peak_after_rms
        audio = np.clip(audio, -1.0, 1.0)
        post_peak = float(np.max(np.abs(audio))) if sample_count > 0 else 0.0
        post_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if sample_count > 0 else 0.0
        return audio.astype(np.float32), {
            "pre_peak": pre_peak,
            "pre_rms": pre_rms,
            "post_peak": post_peak,
            "post_rms": post_rms,
            "gain_applied": gain,
        }

    audio = np.clip(audio * gain, -1.0, 1.0)
    post_peak = float(np.max(np.abs(audio))) if sample_count > 0 else 0.0
    post_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if sample_count > 0 else 0.0
    return audio.astype(np.float32), {
        "pre_peak": pre_peak,
        "pre_rms": pre_rms,
        "post_peak": post_peak,
        "post_rms": post_rms,
        "gain_applied": gain,
    }


def write_pcm16_wav(path, sample_rate, audio):
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def convert_one_file(path, input_root, output_root, args, ffmpeg_bin, ffprobe_bin):
    rel = path.relative_to(input_root)
    output_suffix = ".wav" if args.output_format == "wav" else ".npy"
    output_path = (output_root / rel).with_suffix(output_suffix)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")

    if output_path.exists() and not args.overwrite:
        return {
            "status": "skipped_exists",
            "input_path": str(path),
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_meta = get_source_metadata(path, ffprobe_bin)

    try:
        if ffmpeg_bin:
            audio = decode_with_ffmpeg(path, ffmpeg_bin, args.target_sample_rate)
            decode_mode = "ffmpeg"
        else:
            if path.suffix.lower() not in {".wav", ".wave"}:
                raise RuntimeError("ffmpeg is required for non-WAV inputs")
            wav_audio, src_rate = read_wav_mono_float32(path)
            audio = resample_linear(wav_audio, src_rate, args.target_sample_rate)
            decode_mode = "wave_fallback"

        normalized, norm_stats = normalize_audio(
            audio=audio,
            mode=args.normalization,
            target_peak=float(args.target_peak),
            target_rms=float(args.target_rms),
            remove_dc=bool(args.remove_dc),
        )

        if args.output_format == "wav":
            write_pcm16_wav(output_path, args.target_sample_rate, normalized)
        else:
            np.save(output_path, normalized.astype(np.float32))

        sidecar = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "ok",
            "input": {
                "path": str(path),
                "extension": path.suffix.lower(),
                **source_meta,
            },
            "output": {
                "path": str(output_path),
                "format": args.output_format,
                "sample_rate": int(args.target_sample_rate),
                "channels": 1,
                "dtype": "int16" if args.output_format == "wav" else "float32",
                "num_samples": int(len(normalized)),
                "duration_sec": float(len(normalized) / max(1, args.target_sample_rate)),
                "num_bytes": int(output_path.stat().st_size),
            },
            "conversion_settings": {
                "decode_mode": decode_mode,
                "target_sample_rate": int(args.target_sample_rate),
                "output_format": args.output_format,
                "normalization": args.normalization,
                "target_peak": float(args.target_peak),
                "target_rms": float(args.target_rms),
                "remove_dc": bool(args.remove_dc),
            },
            "normalization_stats": norm_stats,
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        return {
            "status": "ok",
            "input_path": str(path),
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "num_samples": int(len(normalized)),
            "output_bytes": int(output_path.stat().st_size),
        }
    except Exception as exc:  # pylint: disable=broad-except
        error_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "failed",
            "input_path": str(path),
            "error": str(exc),
        }
        sidecar_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        return {
            "status": "failed",
            "input_path": str(path),
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "error": str(exc),
        }


def main():
    args = parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    files = collect_audio_paths(input_root)
    if args.limit > 0:
        files = files[: args.limit]

    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    if not files:
        raise ValueError(f"No supported audio files found in '{input_root}'")

    print(f"[*] Found {len(files)} audio file(s)")
    if not ffmpeg_bin:
        print("[!] ffmpeg not found. Only WAV/WAVE fallback decoding is available.")

    results = []
    for idx, path in enumerate(files, start=1):
        print(f"  - [{idx}/{len(files)}] {path}")
        result = convert_one_file(path, input_root, output_root, args, ffmpeg_bin, ffprobe_bin)
        results.append(result)

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") == "failed"]
    skipped = [r for r in results if r.get("status") == "skipped_exists"]

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": vars(args),
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "ffmpeg_found": bool(ffmpeg_bin),
        "ffprobe_found": bool(ffprobe_bin),
        "counts": {
            "total": len(results),
            "ok": len(ok),
            "failed": len(failed),
            "skipped_exists": len(skipped),
        },
        "output_total_bytes": int(sum(int(r.get("output_bytes", 0)) for r in ok)),
        "failures": failed,
        "converted_files": ok,
    }

    report_path = output_root / "preparation_report.md"
    write_markdown_json_report(report_payload, str(report_path), title="Audio Dataset Preparation Report")

    print("\n[+] Audio dataset preparation complete")
    print(f"[+] Output root: {output_root}")
    print(f"[+] Report: {report_path}")
    print(f"[+] Converted: {len(ok)} | Skipped: {len(skipped)} | Failed: {len(failed)}")


if __name__ == "__main__":
    main()

