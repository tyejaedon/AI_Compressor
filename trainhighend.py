#!/usr/bin/env python3
"""High-End Neural Compression Trainer Orchestrator.

Orchestrates training across all 5 autoencoder and super-resolution models:
1. image_standard: High-capacity edge-sharpening residual autoencoder (train_autoencoder_image_local.py)
2. image_lossy: Rate-distortion autoencoder with Shannon entropy accounting (train_autoencoder_image_lossy_local.py)
3. image_upscaler: Learned sub-pixel super-resolution upscaler (train_upscaler_image_local.py)
4. audio: Continuous phase linear-bottleneck audio autoencoder (train_autoencoder_audio_local.py)
5. video: 3D spatio-temporal video autoencoder (train_autoencoder_video_local.py)

Usage:
    ./trainhighend --help
    python trainhighend.py --dry-run
    python trainhighend.py --only image_upscaler image_lossy
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "reporting"))
try:
    from report_markdown import read_markdown_json_report, write_markdown_json_report
except ImportError:
    read_markdown_json_report = None
    write_markdown_json_report = None

# High-end optimal hyperparameter recipes per modality
HIGHEND_JOBS: list[dict[str, Any]] = [
    {
        "name": "image_standard",
        "description": "High-capacity edge-sharpening image autoencoder",
        "script": "src/training/train_autoencoder_image_local.py",
        "report_file": "evaluation_report.md",
        "flags": {
            "preset": "custom",
            "use_cifar10": True,
            "cifar_train_limit": 20000,
            "cifar_test_limit": 5000,
            "block_size": 32,
            "latent_dim": 192,
            "model_base_filters": 64,
            "model_kernel_size": 3,
            "batch_size": 32,
            "epochs": 12,
            "lr": 2.5e-4,
            "upscale_factor": 2,
            "degrade_interp": "bicubic",
            "loss_profile": "balanced",
            "target_psnr": 60.0,
            "disable_target_psnr_stop": True,
            "no_export_tflite": True,
        },
    },
    {
        "name": "image_lossy",
        "description": "Rate-distortion lossy autoencoder with empirical Shannon entropy accounting",
        "script": "src/training/train_autoencoder_image_lossy_local.py",
        "report_file": "evaluation_report.md",
        "flags": {
            "preset": "custom",
            "use_cifar10": True,
            "block_size": 32,
            "latent_dim": 48,
            "model_base_filters": 32,
            "model_kernel_size": 3,
            "batch_size": 32,
            "epochs": 10,
            "lr": 2.5e-4,
            "rate_loss_mode": "entropy",
            "rate_lambda": 0.0006,
            "latent_bit_depth": 8,
            "quant_noise_anneal": True,
            "enable_entropy_coding": True,
            "entropy_coding_samples": 16,
            "jpeg_quality": 85,
            "sample_count": 32,
            "target_psnr": 60.0,
            "run_baseline_benchmark": True,
            "no_export_tflite": True,
        },
    },
    {
        "name": "image_upscaler",
        "description": "Learned sub-pixel super-resolution upscaler",
        "script": "src/training/train_upscaler_image_local.py",
        "report_file": "upscaler_evaluation_report.md",
        "flags": {
            "preset": "custom",
            "data_dir": "",
            "use_cifar10": True,
            "cifar_train_limit": 20000,
            "cifar_test_limit": 5000,
            "block_size": 32,
            "upscale_factor": 2,
            "degrade_interp": "bicubic",
            "model_base_filters": 40,
            "model_kernel_size": 3,
            "num_residual_blocks": 5,
            "batch_size": 32,
            "epochs": 14,
            "lr": 2.5e-4,
            "target_psnr": 35.0,
            "sample_count": 8,
            "no_export_tflite": True,
        },
    },
    {
        "name": "audio",
        "description": "Continuous phase linear-bottleneck audio autoencoder with 8-bit QAT quantization",
        "script": "src/training/train_autoencoder_audio_local.py",
        "report_file": "audio_evaluation_report.md",
        "flags": {
            "preset": "custom",
            "data_dir": "data/AudioData/ESC-50-master/audio",
            "real_file_limit": 300,
            "clip_seconds": 0.5,
            "latent_dim": 96,
            "latent_bit_depth": 8,
            "quant_noise_anneal": True,
            "model_base_filters": 32,
            "model_kernel_size": 5,
            "batch_size": 16,
            "epochs": 16,
            "lr": 1.2e-4,
            "target_snr_db": 90.0,
            "disable_target_snr_stop": True,
            "loss_profile": "balanced",
            "eval_perceptual_metrics": True,
            "run_mp3_benchmark": True,
            "no_export_tflite": True,
        },
    },
    {
        "name": "video",
        "description": "3D spatio-temporal Conv3D video autoencoder",
        "script": "src/training/train_autoencoder_video_local.py",
        "report_file": "video_evaluation_report.md",
        "flags": {
            "preset": "custom",
            "data_dir": "data/VIDEO DATA",
            "frames": 6,
            "height": 48,
            "width": 48,
            "fps": 12,
            "latent_dim": 256,
            "model_base_filters": 24,
            "model_kernel_size": 3,
            "batch_size": 4,
            "epochs": 8,
            "lr": 1.0e-4,
            "real_max_clips": 160,
            "real_clip_stride": 3,
            "target_psnr": 60.0,
            "run_mp4_benchmark": True,
            "no_export_tflite": True,
        },
    },
]


def dict_to_cli(flags: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        else:
            cli.extend([flag, str(value)])
    return cli


def find_latest_report(search_dir: Path, filename: str) -> Path | None:
    reports = sorted(search_dir.rglob(filename), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def load_report_payload(report_path: Path) -> dict[str, Any] | None:
    if read_markdown_json_report is not None:
        payload = read_markdown_json_report(report_path)
        if payload is not None:
            return payload
    try:
        content = report_path.read_text(encoding="utf-8")
        if "```json" in content:
            raw = content.split("```json", 1)[1].split("```", 1)[0]
            return json.loads(raw)
        return json.loads(content)
    except Exception:
        return None


def extract_key_metrics(payload: dict[str, Any], job_name: str) -> dict[str, Any]:
    test_metrics = payload.get("test_metrics", {})
    metrics: dict[str, Any] = {
        "psnr": test_metrics.get("psnr_metric"),
        "mse": test_metrics.get("mse"),
        "loss": test_metrics.get("loss"),
        "ssim": test_metrics.get("ssim_metric"),
        "snr_db": test_metrics.get("snr_db_metric"),
    }
    if job_name == "image_upscaler":
        base = payload.get("bicubic_baseline", {})
        metrics["bicubic_psnr"] = base.get("bicubic_psnr")
        metrics["bicubic_ssim"] = base.get("bicubic_ssim")
    elif job_name == "image_lossy":
        rate_stats = payload.get("latent_rate_stats", {})
        metrics["bpp"] = rate_stats.get("latent_bits_per_pixel")
        metrics["entropy_bytes"] = rate_stats.get("estimated_bitstream_bytes_per_image")
    elif job_name == "audio":
        metrics["actual_compression_ratio"] = payload.get("actual_compression_ratio")
        metrics["estimated_ratio"] = payload.get("estimated_input_to_latent_ratio")
        metrics["actual_bytes"] = payload.get("actual_latent_bytes_per_clip")
        pm = payload.get("perceptual_metrics", {})
        if pm:
            metrics["lsd_db"] = pm.get("lsd_db")
            metrics["stoi_proxy"] = pm.get("envelope_correlation_stoi_proxy")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-End Neural Compression Trainer Orchestrator (all 5 models with optimal hyperparams)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/highend_run",
        help="Base directory where timestamped high-end runs are stored",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter executable to invoke trainers with",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only specified jobs (choices: image_standard, image_lossy, image_upscaler, audio, video)",
    )
    parser.add_argument(
        "--epochs-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to each job's epoch count (e.g. 0.5 for fast verify, 1.0 for standard)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display plan and command lines without executing training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_jobs = [j for j in HIGHEND_JOBS if not args.only or j["name"] in args.only]
    if not selected_jobs:
        valid_names = [j["name"] for j in HIGHEND_JOBS]
        print(f"[x] Error: No valid jobs selected from: {args.only}. Valid: {valid_names}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT / args.output_root / timestamp).resolve()
    log_dir = run_root / "logs"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" HIGH-END NEURAL COMPRESSION TRAINING ORCHESTRATOR")
    print("=" * 80)
    print(f"[+] Project root:    {ROOT}")
    print(f"[+] Output root:     {run_root}")
    print(f"[+] Python exec:     {args.python}")
    print(f"[+] Selected jobs:   {[j['name'] for j in selected_jobs]}")
    print(f"[+] Epochs scale:    {args.epochs_scale}")
    print(f"[+] Dry run:         {args.dry_run}")
    print("=" * 80)

    summary_records: list[dict[str, Any]] = []
    overall_start = time.monotonic()

    for idx, job in enumerate(selected_jobs, start=1):
        name = str(job["name"])
        desc = str(job["description"])
        script = str(job["script"])
        report_file = str(job["report_file"])
        flags = dict(job["flags"])

        if args.epochs_scale != 1.0 and "epochs" in flags:
            scaled_epochs = max(1, int(round(flags["epochs"] * args.epochs_scale)))
            flags["epochs"] = scaled_epochs

        job_out = run_root / name
        flags["output_root"] = str(job_out)
        cmd = [args.python, str(ROOT / script), *dict_to_cli(flags)]

        print(f"\n[{idx}/{len(selected_jobs)}] STARTING: {name} ({desc})")
        print(f"    Command: {' '.join(shlex.quote(c) for c in cmd)}")

        if args.dry_run:
            summary_records.append({
                "job": name,
                "status": "dry_run",
                "hyperparams": flags,
            })
            continue

        log_path = log_dir / f"{name}.log"
        t0 = time.monotonic()
        status = "ok"

        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                status = f"failed(code={proc.returncode})"

        elapsed = time.monotonic() - t0
        print(f"    -> Finished in {elapsed:.1f}s ({elapsed / 60:.2f} min) with status: {status}")

        report_path = find_latest_report(job_out, report_file)
        payload = load_report_payload(report_path) if report_path else None
        metrics = extract_key_metrics(payload, name) if payload else {}

        summary_records.append({
            "job": name,
            "status": status,
            "duration_seconds": round(elapsed, 2),
            "report_path": str(report_path) if report_path else None,
            "metrics": metrics,
            "hyperparams": flags,
        })

        if metrics:
            print(f"    Metrics: {json.dumps(metrics, indent=2)}")

    total_time = time.monotonic() - overall_start
    print("\n" + "=" * 80)
    print(f" HIGH-END TRAINING COMPLETE (Total time: {total_time / 60:.2f} min)")
    print("=" * 80)

    if not args.dry_run:
        summary_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_duration_minutes": round(total_time / 60, 2),
            "jobs": summary_records,
        }
        summary_json = run_root / "highend_summary.json"
        summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"[+] Summary saved to: {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
