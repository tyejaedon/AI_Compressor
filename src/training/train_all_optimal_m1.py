#!/usr/bin/env python3
"""Sequential M1 trainer with retries until target PSNR is reached per modality."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
from report_markdown import read_markdown_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all modality trainers sequentially and retry with stronger hyperparameters until PSNR target is met.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Project root containing training scripts",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for all training subprocesses",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/optimal_m1_retry",
        help="Base output directory where each modality run folder is written",
    )
    parser.add_argument(
        "--target-psnr",
        type=float,
        default=33.0,
        help="Target PSNR passed to each trainer",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Maximum retry attempts per modality",
    )
    parser.add_argument(
        "--fullres-image-count",
        type=int,
        default=20,
        help="How many full-resolution images to evaluate for image upscaler PSNR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def as_cli(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def find_latest_report(run_root: Path, report_name: str) -> Path | None:
    reports = sorted(run_root.rglob(report_name), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def read_test_psnr(report_path: Path) -> float | None:
    payload = read_markdown_json_report(report_path)
    if payload is None:
        return None
    value = payload.get("test_metrics", {}).get("psnr_metric")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def dict_to_cli(extra: dict[str, object]) -> list[str]:
    cli: list[str] = []
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def get_image_upscale_psnr(project_root: Path, python_exec: str, model_path: Path, output_root: str, sample_count: int) -> tuple[float | None, Path | None]:
    run_root = project_root / output_root / "image_standard_fullres_eval"
    cmd = [
        python_exec,
        "src/reporting/upscale_reconstructed_images.py",
        "--model",
        str(model_path),
        "--data-dir",
        "data/ImageData",
        "--count",
        str(sample_count),
        "--upscale-mode",
        "original",
        "--output-root",
        str(run_root),
    ]
    proc = subprocess.run(cmd, cwd=project_root)
    if proc.returncode != 0:
        return None, None

    report_path = find_latest_report(run_root, "upscale_report.md")
    if report_path is None:
        return None, None

    payload = read_markdown_json_report(report_path)
    if payload is None:
        return None, report_path

    samples = payload.get("samples", [])
    vals = [float(s.get("psnr_upscaled")) for s in samples if isinstance(s, dict) and isinstance(s.get("psnr_upscaled"), (int, float))]
    if not vals:
        return None, report_path
    return sum(vals) / len(vals), report_path


def modality_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    root = args.output_root.rstrip("/")
    target_psnr = str(args.target_psnr)
    common = {"target_psnr": target_psnr, "no_export_tflite": True}

    return [
        {
            "name": "image_standard_upscaler",
            "script": "src/training/train_autoencoder_image_local.py",
            "report": "evaluation_report.md",
            "output_root": f"{root}/image_standard",
            "base": {
                **common,
                "preset": "custom",
                "train_dir": "data/cifar10 2/train",
                "val_dir": "data/cifar10 2/val",
                "test_dir": "data/cifar10 2/test",
                "holdout_dir": "data/ImageData",
                "real_only": True,
                "upscale_factor": 2,
                "degrade_interp": "bicubic",
            },
            "profiles": [
                {"epochs": 10, "batch_size": 8, "block_size": 64, "latent_dim": 128, "lr": 1.2e-4, "train_patches_per_image": 2},
                {"epochs": 18, "batch_size": 8, "block_size": 96, "latent_dim": 192, "lr": 1.0e-4, "train_patches_per_image": 2},
                {"epochs": 30, "batch_size": 6, "block_size": 128, "latent_dim": 256, "lr": 8e-5, "train_patches_per_image": 3},
                {"epochs": 48, "batch_size": 6, "block_size": 128, "latent_dim": 320, "lr": 6e-5, "train_patches_per_image": 4},
            ],
            "use_fullres_check": True,
        },
        {
            "name": "image_lossy",
            "script": "src/training/train_autoencoder_image_lossy_local.py",
            "report": "evaluation_report.md",
            "output_root": f"{root}/image_lossy",
            "base": {
                **common,
                "preset": "custom",
                "train_dir": "data/cifar10 2/train",
                "val_dir": "data/cifar10 2/val",
                "test_dir": "data/cifar10 2/test",
                "real_only": True,
                "run_baseline_benchmark": True,
                "target_compression_ratio": 0.80,
            },
            "profiles": [
                {"epochs": 16, "batch_size": 8, "block_size": 64, "latent_dim": 48, "lr": 1.2e-4, "rate_lambda": 0.01, "jpeg_quality": 86, "sample_count": 16},
                {"epochs": 28, "batch_size": 8, "block_size": 96, "latent_dim": 64, "lr": 9e-5, "rate_lambda": 0.006, "jpeg_quality": 88, "sample_count": 24},
                {"epochs": 40, "batch_size": 6, "block_size": 128, "latent_dim": 96, "lr": 7e-5, "rate_lambda": 0.004, "jpeg_quality": 90, "sample_count": 32},
            ],
            "use_fullres_check": False,
        },
        {
            "name": "audio",
            "script": "src/training/train_autoencoder_audio_local.py",
            "report": "audio_evaluation_report.md",
            "output_root": f"{root}/audio",
            "base": {
                **common,
                "preset": "custom",
                "data_dir": "data/AudioData",
                "run_mp3_benchmark": True,
                "clip_seconds": 1.0,
            },
            "profiles": [
                {"epochs": 20, "batch_size": 12, "latent_dim": 224, "lr": 1.0e-4, "real_file_limit": 1200},
                {"epochs": 36, "batch_size": 12, "latent_dim": 320, "lr": 8e-5, "real_file_limit": 1800},
                {"epochs": 56, "batch_size": 10, "latent_dim": 384, "lr": 6e-5, "real_file_limit": 2400},
            ],
            "use_fullres_check": False,
        },
        {
            "name": "video",
            "script": "src/training/train_autoencoder_video_local.py",
            "report": "video_evaluation_report.md",
            "output_root": f"{root}/video",
            "base": {
                **common,
                "preset": "custom",
                "data_dir": "data/VIDEO DATA",
                "run_mp4_benchmark": True,
                "frames": 8,
                "height": 64,
                "width": 64,
                "real_max_clips": 1200,
            },
            "profiles": [
                {"epochs": 18, "batch_size": 3, "latent_dim": 224, "lr": 1.0e-4, "real_max_clips": 1200, "real_clip_stride": 2},
                {"epochs": 32, "batch_size": 3, "latent_dim": 320, "lr": 8e-5, "real_max_clips": 1800, "real_clip_stride": 2},
                {"epochs": 48, "batch_size": 2, "latent_dim": 384, "lr": 6e-5, "real_max_clips": 2600, "real_clip_stride": 1},
            ],
            "use_fullres_check": False,
        },
    ]


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    (project_root / args.output_root).mkdir(parents=True, exist_ok=True)

    plans = modality_plan(args)
    print(f"[+] Project root: {project_root}")
    print(f"[+] Python: {args.python}")
    print(f"[+] Planned modalities: {len(plans)}")
    print(f"[+] Target PSNR: {args.target_psnr:.3f}")

    results: dict[str, dict[str, object]] = {}

    for mod_idx, plan in enumerate(plans, start=1):
        name = str(plan["name"])
        script = str(plan["script"])
        report_name = str(plan["report"])
        output_root = str(plan["output_root"])
        base = dict(plan["base"])
        profiles = list(plan["profiles"])
        fullres = bool(plan["use_fullres_check"])

        print(f"\n[{mod_idx}/{len(plans)}] {name}")
        print(f"[+] Output root: {output_root}")

        reached = False
        for attempt in range(1, args.max_attempts + 1):
            profile = dict(profiles[min(attempt - 1, len(profiles) - 1)])
            if attempt > len(profiles):
                # Keep nudging capacity upward after profile list is exhausted.
                profile["epochs"] = int(profile.get("epochs", 120)) + 20 * (attempt - len(profiles))
                profile["lr"] = max(float(profile.get("lr", 1e-4)) * 0.85, 2e-5)

            merged = {**base, **profile, "output_root": output_root}
            cmd = [args.python, script, *dict_to_cli(merged)]

            tweak_preview = json.dumps(profile, sort_keys=True)
            print(f"\n  - Attempt {attempt}/{args.max_attempts}")
            print(f"    Tweaks: {tweak_preview}")
            print(f"    Command: {as_cli(cmd)}")

            if args.dry_run:
                continue

            started = datetime.now()
            proc = subprocess.run(cmd, cwd=project_root)
            elapsed = datetime.now() - started
            if proc.returncode != 0:
                print(f"    [x] Training failed (exit={proc.returncode}, elapsed={elapsed})")
                return proc.returncode

            run_root = project_root / output_root
            report_path = find_latest_report(run_root, report_name)
            if report_path is None:
                print("    [x] No evaluation report found after run")
                return 2

            psnr = read_test_psnr(report_path)
            metric_name = "test_psnr"
            if psnr is None:
                print(f"    [x] Could not parse PSNR from {report_path}")
                return 2

            if fullres:
                model_candidates = sorted(report_path.parent.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not model_candidates:
                    print(f"    [x] No .keras model found in {report_path.parent}")
                    return 2
                fullres_psnr, fullres_report = get_image_upscale_psnr(
                    project_root=project_root,
                    python_exec=args.python,
                    model_path=model_candidates[0],
                    output_root=args.output_root,
                    sample_count=args.fullres_image_count,
                )
                if fullres_psnr is None:
                    print("    [x] Full-resolution upscale validation failed")
                    return 2
                metric_name = "fullres_upscaled_psnr"
                psnr = float(fullres_psnr)
                print(f"    [i] Full-res report: {fullres_report}")

            print(f"    [+] {metric_name}={psnr:.3f} (elapsed={elapsed})")
            if psnr >= args.target_psnr:
                results[name] = {
                    "attempt": attempt,
                    "metric_name": metric_name,
                    "metric": psnr,
                    "report": str(report_path),
                    "hyperparams": profile,
                }
                reached = True
                print(f"    [+] Target reached for {name}")
                break

        if args.dry_run:
            continue

        if not reached:
            print(f"[x] {name} did not reach target PSNR {args.target_psnr:.3f} after {args.max_attempts} attempts")
            return 3

    if not args.dry_run:
        print("\n[+] All modalities reached target PSNR")
        for name, payload in results.items():
            print(f"  - {name}: {payload['metric_name']}={payload['metric']:.3f} on attempt {payload['attempt']}")
            print(f"    report: {payload['report']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

