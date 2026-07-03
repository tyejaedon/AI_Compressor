#!/usr/bin/env python3
"""Random-search hyperparameter tuner for local autoencoder trainers.

This utility launches multiple short training runs under a resource budget,
collects PSNR from each run report, and recommends the best config.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from report_markdown import read_markdown_json_report


@dataclass
class TrialResult:
    trial_id: int
    params: dict[str, Any]
    score_psnr: float
    duration_sec: float
    report_path: str | None
    run_dir: str
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for hyperparameters under local resource constraints.")
    parser.add_argument("--modality", choices=["image", "audio", "video"], required=True, help="Which trainer to tune")
    parser.add_argument("--trials", type=int, default=8, help="Maximum number of random trials")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--max-minutes", type=float, default=90.0, help="Wall-clock budget for the whole search")
    parser.add_argument("--resource-profile", choices=["tiny", "balanced"], default="balanced", help="Resource budget profile")
    parser.add_argument("--output-root", type=str, default="models/random_search", help="Where trial folders and summary are written")
    parser.add_argument("--top-k", type=int, default=3, help="How many top runs to include in the summary")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute trainers; simulate scores to validate workflow")
    parser.add_argument("--goal-psnr", type=float, default=30.0, help="Target PSNR to stop early when reached")
    parser.add_argument("--until-goal", action="store_true", help="Keep sampling until goal PSNR or max-total-trials is reached")
    parser.add_argument("--max-total-trials", type=int, default=200, help="Safety cap when using --until-goal")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional trainer arg, can be passed multiple times (example: --extra-arg --target-psnr --extra-arg 24)",
    )
    return parser.parse_args()


def modality_to_script(modality: str) -> str:
    mapping = {
        "image": "train_autoencoder_image_local.py",
        "audio": "train_autoencoder_audio_local.py",
        "video": "train_autoencoder_video_local.py",
    }
    return mapping[modality]


def modality_to_report_name(modality: str) -> str:
    mapping = {
        "image": "evaluation_report.md",
        "audio": "audio_evaluation_report.md",
        "video": "video_evaluation_report.md",
    }
    return mapping[modality]


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_params(modality: str, rng: random.Random, profile: str) -> dict[str, Any]:
    if modality == "image":
        epoch_options = [4, 6, 8] if profile == "tiny" else [12, 16, 20, 24, 30]
        return {
            "preset": "custom",
            "block_size": rng.choice([32, 48, 64, 96]),
            "latent_dim": rng.choice([64, 96, 128, 192, 256]),
            "batch_size": rng.choice([4, 6, 8]),
            "epochs": rng.choice(epoch_options),
            "lr": round(sample_log_uniform(rng, 5e-5, 3.2e-4), 7),
        }

    if modality == "audio":
        epoch_options = [4, 6, 8] if profile == "tiny" else [10, 14, 18, 24]
        return {
            "preset": "custom",
            "latent_dim": rng.choice([160, 224, 320, 384, 512]),
            "batch_size": rng.choice([8, 12, 16]),
            "epochs": rng.choice(epoch_options),
            "lr": round(sample_log_uniform(rng, 5e-5, 2.5e-4), 7),
            "clip_seconds": rng.choice([0.5, 0.75, 1.0]),
            "synthetic_profile": rng.choice(["legacy", "expansive"]),
        }

    epoch_options = [4, 6, 8] if profile == "tiny" else [8, 12, 16, 20]
    frame_options = [4, 6, 8] if profile == "tiny" else [4, 6, 8]
    return {
        "preset": "custom",
        "latent_dim": rng.choice([192, 256, 320, 384, 512]),
        "batch_size": rng.choice([2, 3, 4]),
        "epochs": rng.choice(epoch_options),
        "lr": round(sample_log_uniform(rng, 4e-5, 2.0e-4), 7),
        "frames": rng.choice(frame_options),
        "height": rng.choice([32, 48, 64]),
        "width": rng.choice([32, 48, 64]),
    }


def fixed_resource_args(modality: str, profile: str, project_root: Path) -> list[str]:
    if modality == "image":
        missing_dir = project_root / "data" / "_random_search_missing"
        if profile == "tiny":
            return [
                "--data-dir",
                str(missing_dir),
                "--extensive-dummy-samples",
                "1200",
                "--target-psnr",
                "22",
                "--no-export-tflite",
            ]
        return [
            "--data-dir",
            str(missing_dir),
            "--extensive-dummy-samples",
            "3000",
            "--target-psnr",
            "30",
            "--no-export-tflite",
        ]

    if modality == "audio":
        if profile == "tiny":
            return [
                "--synthetic-only",
                "--train-samples",
                "600",
                "--val-samples",
                "150",
                "--test-samples",
                "150",
                "--target-psnr",
                "24",
                "--no-export-tflite",
            ]
        return [
            "--synthetic-only",
            "--train-samples",
            "5000",
            "--val-samples",
            "1000",
            "--test-samples",
            "1000",
            "--target-psnr",
            "30",
            "--no-export-tflite",
        ]

    if profile == "tiny":
        return [
            "--synthetic-only",
            "--train-samples",
            "400",
            "--val-samples",
            "100",
            "--test-samples",
            "100",
            "--fps",
            "10",
            "--target-psnr",
            "22",
            "--no-export-tflite",
        ]
    return [
        "--synthetic-only",
        "--train-samples",
        "2400",
        "--val-samples",
        "600",
        "--test-samples",
        "600",
        "--fps",
        "12",
        "--target-psnr",
        "30",
        "--no-export-tflite",
    ]


def params_to_cli_args(params: dict[str, Any]) -> list[str]:
    cli = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        cli.extend([flag, str(value)])
    return cli


def find_report(trial_root: Path, report_name: str) -> Path | None:
    candidates = sorted(trial_root.rglob(report_name), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def read_psnr(report_path: Path) -> float | None:
    payload = read_markdown_json_report(report_path)
    if payload is None:
        return None
    value = payload.get("test_metrics", {}).get("psnr_metric")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def simulated_score(params: dict[str, Any], rng: random.Random, modality: str) -> float:
    base = {"image": 21.5, "audio": 24.0, "video": 20.0}[modality]
    lr = float(params.get("lr", 1e-4))
    latent = float(params.get("latent_dim", 64))
    epochs = float(params.get("epochs", 8))

    lr_bonus = max(0.0, 2.0 - abs(math.log10(lr) + 4.0))
    latent_bonus = min(3.0, latent / 128.0)
    epoch_bonus = min(2.5, epochs / 6.0)
    noise = rng.uniform(-0.6, 0.6)
    return round(base + lr_bonus + latent_bonus + epoch_bonus + noise, 4)


def trial_command(
    python_exec: str,
    trainer_script: Path,
    trial_dir: Path,
    params: dict[str, Any],
    modality: str,
    profile: str,
    project_root: Path,
    extra_args: list[str],
) -> list[str]:
    cmd = [python_exec, str(trainer_script)]
    cmd.extend(params_to_cli_args(params))
    cmd.extend(["--output-root", str(trial_dir)])
    cmd.extend(fixed_resource_args(modality, profile, project_root))
    cmd.extend(extra_args)
    return cmd


def summarize_ranges(trials: list[TrialResult], keys: list[str], top_n: int) -> dict[str, Any]:
    if not trials:
        return {}

    top = sorted(trials, key=lambda t: t.score_psnr, reverse=True)[:top_n]
    ranges: dict[str, Any] = {}
    for key in keys:
        values = [t.params.get(key) for t in top if key in t.params]
        if not values:
            continue
        if all(isinstance(v, (int, float)) for v in values):
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            ranges[key] = {
                "min": round(min(nums), 7),
                "max": round(max(nums), 7),
                "mean": round(sum(nums) / len(nums), 7),
            }
        else:
            counts: dict[str, int] = {}
            for v in values:
                text = str(v)
                counts[text] = counts.get(text, 0) + 1
            ranges[key] = {"frequency": counts}
    return ranges


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    project_root = Path(__file__).resolve().parent
    script_path = project_root / modality_to_script(args.modality)
    report_name = modality_to_report_name(args.modality)

    if not script_path.exists():
        raise FileNotFoundError(f"Trainer script not found: {script_path}")

    search_root = project_root / args.output_root / args.modality
    search_root.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    budget_sec = max(1.0, args.max_minutes * 60.0)
    results: list[TrialResult] = []

    print(f"[*] Starting random search: modality={args.modality}, trials={args.trials}, budget={args.max_minutes:.1f} min")

    requested_trials = args.max_total_trials if args.until_goal else args.trials
    goal_reached = False

    for trial_idx in range(1, requested_trials + 1):
        if not args.until_goal and trial_idx > args.trials:
            break
        elapsed = time.time() - started_at
        if elapsed >= budget_sec:
            print(f"[!] Budget reached after {trial_idx - 1} completed trial(s).")
            break

        trial_params = sample_params(args.modality, rng, args.resource_profile)
        trial_dir = search_root / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        start_trial = time.time()
        if args.dry_run:
            score = simulated_score(trial_params, rng, args.modality)
            duration = time.time() - start_trial
            result = TrialResult(
                trial_id=trial_idx,
                params=trial_params,
                score_psnr=score,
                duration_sec=duration,
                report_path=None,
                run_dir=str(trial_dir),
                status="simulated",
            )
            print(f"  - Trial {trial_idx:03d}: simulated PSNR={score:.3f}")
            results.append(result)
            continue

        cmd = trial_command(
            python_exec=sys.executable,
            trainer_script=script_path,
            trial_dir=trial_dir,
            params=trial_params,
            modality=args.modality,
            profile=args.resource_profile,
            project_root=project_root,
            extra_args=args.extra_arg,
        )

        print(f"  - Trial {trial_idx:03d}: running {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_trial

        if proc.returncode != 0:
            err_path = trial_dir / "trial_error.log"
            err_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
            results.append(
                TrialResult(
                    trial_id=trial_idx,
                    params=trial_params,
                    score_psnr=float("-inf"),
                    duration_sec=duration,
                    report_path=None,
                    run_dir=str(trial_dir),
                    status="failed",
                    error=f"trainer exit={proc.returncode}; see {err_path}",
                )
            )
            print(f"    [!] Failed (exit {proc.returncode})")
            continue

        report_path = find_report(trial_dir, report_name)
        if report_path is None:
            results.append(
                TrialResult(
                    trial_id=trial_idx,
                    params=trial_params,
                    score_psnr=float("-inf"),
                    duration_sec=duration,
                    report_path=None,
                    run_dir=str(trial_dir),
                    status="failed",
                    error="report file not found",
                )
            )
            print("    [!] Failed (report file not found)")
            continue

        psnr = read_psnr(report_path)
        if psnr is None:
            results.append(
                TrialResult(
                    trial_id=trial_idx,
                    params=trial_params,
                    score_psnr=float("-inf"),
                    duration_sec=duration,
                    report_path=str(report_path),
                    run_dir=str(report_path.parent),
                    status="failed",
                    error="psnr_metric missing in report",
                )
            )
            print("    [!] Failed (PSNR not found in report)")
            continue

        result = TrialResult(
            trial_id=trial_idx,
            params=trial_params,
            score_psnr=psnr,
            duration_sec=duration,
            report_path=str(report_path),
            run_dir=str(report_path.parent),
            status="ok",
        )
        results.append(result)
        print(f"    [+] PSNR={psnr:.3f} ({duration:.1f}s)")

        if args.until_goal and psnr >= args.goal_psnr:
            goal_reached = True
            print(f"[+] Goal reached (PSNR {psnr:.3f} >= {args.goal_psnr:.3f}) at trial {trial_idx:03d}")
            break

    successful = [r for r in results if math.isfinite(r.score_psnr)]
    successful_sorted = sorted(successful, key=lambda r: r.score_psnr, reverse=True)

    best = successful_sorted[0] if successful_sorted else None

    key_candidates = {
        "image": ["block_size", "latent_dim", "batch_size", "epochs", "lr"],
        "audio": ["latent_dim", "batch_size", "epochs", "lr", "clip_seconds", "synthetic_profile"],
        "video": ["latent_dim", "batch_size", "epochs", "lr", "frames", "height", "width"],
    }
    ranges = summarize_ranges(successful_sorted, key_candidates[args.modality], top_n=max(1, args.top_k))

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": args.modality,
        "resource_profile": args.resource_profile,
        "requested_trials": args.trials,
        "until_goal": args.until_goal,
        "max_total_trials": args.max_total_trials,
        "goal_psnr": args.goal_psnr,
        "goal_reached": goal_reached,
        "completed_trials": len(results),
        "successful_trials": len(successful_sorted),
        "budget_minutes": args.max_minutes,
        "elapsed_minutes": round((time.time() - started_at) / 60.0, 3),
        "best_trial": None
        if best is None
        else {
            "trial_id": best.trial_id,
            "psnr_metric": best.score_psnr,
            "params": best.params,
            "report_path": best.report_path,
            "run_dir": best.run_dir,
            "duration_sec": round(best.duration_sec, 3),
        },
        "top_trials": [
            {
                "trial_id": r.trial_id,
                "psnr_metric": r.score_psnr,
                "params": r.params,
                "report_path": r.report_path,
                "run_dir": r.run_dir,
                "duration_sec": round(r.duration_sec, 3),
                "status": r.status,
            }
            for r in successful_sorted[: max(1, args.top_k)]
        ],
        "educated_guess_ranges": ranges,
        "all_trials": [
            {
                "trial_id": r.trial_id,
                "psnr_metric": r.score_psnr,
                "params": r.params,
                "duration_sec": round(r.duration_sec, 3),
                "report_path": r.report_path,
                "run_dir": r.run_dir,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
    }

    summary_path = search_root / f"search_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[+] Random search complete")
    print(f"[+] Summary: {summary_path}")
    if best is not None:
        print(f"[+] Best trial: {best.trial_id:03d}")
        print(f"[+] Best PSNR: {best.score_psnr:.3f}")
        print(f"[+] Best params: {json.dumps(best.params, sort_keys=True)}")
    else:
        print("[!] No successful trial found. Check trial_error.log files in each trial folder.")


if __name__ == "__main__":
    main()


