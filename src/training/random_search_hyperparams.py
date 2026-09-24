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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
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


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


class SearchProgress:
    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.width = 24
        self.started_at = time.time()

    def update(self, completed: int, status: str) -> None:
        done = max(0, min(completed, self.total))
        fraction = done / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.started_at
        eta = (elapsed / done) * (self.total - done) if done > 0 else 0.0
        line = (
            f"[progress] [{bar}] {done:>3}/{self.total:<3} "
            f"({fraction * 100:5.1f}%) elapsed {_fmt_duration(elapsed)} "
            f"eta {_fmt_duration(eta)} | {status}"
        )
        print(line)

    def finish(self) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for hyperparameters under local resource constraints.")
    parser.add_argument("--modality", choices=["image", "audio", "video"], required=True, help="Which trainer to tune")
    parser.add_argument(
        "--image-trainer",
        choices=["standard", "lossy"],
        default="standard",
        help="Trainer variant for image modality; ignored for audio/video",
    )
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
    parser.add_argument(
        "--search-config",
        type=str,
        default="documentation/random_search_profile.json",
        help="Path to JSON config that defines sample space, fixed args, and summary keys",
    )
    return parser.parse_args()


def modality_to_script(modality: str, image_trainer: str) -> str:
    mapping = {
        "image": "train_autoencoder_image_lossy_local.py" if image_trainer == "lossy" else "train_autoencoder_image_local.py",
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


def _resolve_profile_path(project_root: Path, search_config: str) -> Path:
    config_path = Path(search_config)
    if config_path.is_absolute():
        return config_path
    return project_root / config_path


def load_search_config(project_root: Path, search_config: str) -> tuple[dict[str, Any], Path | None]:
    cfg_path = _resolve_profile_path(project_root, search_config)
    if not cfg_path.exists():
        return {}, None
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Search config must be a JSON object: {cfg_path}")
    return data, cfg_path


def _sample_from_spec(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Found empty list in search config sample_space")
        return rng.choice(spec)
    if isinstance(spec, dict):
        if "choices" in spec:
            choices = spec["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("'choices' must be a non-empty list in search config")
            return rng.choice(choices)
        if "log_uniform" in spec:
            low, high = spec["log_uniform"]
            value = sample_log_uniform(rng, float(low), float(high))
            decimals = int(spec.get("round", 0))
            return round(value, decimals) if decimals > 0 else value
        if "uniform" in spec:
            low, high = spec["uniform"]
            value = rng.uniform(float(low), float(high))
            decimals = int(spec.get("round", 0))
            return round(value, decimals) if decimals > 0 else value
        if "int" in spec:
            low, high = spec["int"]
            return int(rng.randint(int(low), int(high)))
        if "int_step" in spec:
            low, high, step = spec["int_step"]
            values = list(range(int(low), int(high) + 1, int(step)))
            if not values:
                raise ValueError("'int_step' produced no values in search config")
            return int(rng.choice(values))
        raise ValueError(f"Unsupported sample spec in search config: {spec}")
    return spec


def _sample_from_config_entry(entry: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for key, spec in entry.items():
        sampled[key] = _sample_from_spec(spec, rng)
    return sampled


def _get_config_sample_entry(config: dict[str, Any], modality: str, image_trainer: str) -> dict[str, Any] | None:
    sample_space = config.get("sample_space")
    if not isinstance(sample_space, dict):
        return None
    mod_entry = sample_space.get(modality)
    if modality == "image" and isinstance(mod_entry, dict):
        mod_entry = mod_entry.get(image_trainer)
    if isinstance(mod_entry, dict):
        return mod_entry
    return None


def sample_params(
    modality: str,
    rng: random.Random,
    profile: str,
    image_trainer: str,
    search_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = search_config or {}
    cfg_entry = _get_config_sample_entry(cfg, modality, image_trainer)
    if cfg_entry:
        return _sample_from_config_entry(cfg_entry, rng)

    if modality == "image":
        if image_trainer == "lossy":
            epoch_options = [4, 6, 8] if profile == "tiny" else [12, 16, 20, 24, 30]
            return {
                "preset": "custom",
                "block_size": rng.choice([32, 48, 64, 96]),
                "latent_dim": rng.choice([12, 16, 20, 24, 32]),
                "model_base_filters": rng.choice([16, 24, 32, 40, 48]),
                "model_kernel_size": rng.choice([3, 5]),
                "batch_size": rng.choice([4, 6, 8]),
                "epochs": rng.choice(epoch_options),
                "lr": round(sample_log_uniform(rng, 5e-5, 3.2e-4), 7),
                "rate_lambda": round(sample_log_uniform(rng, 0.01, 0.08), 5),
                "jpeg_quality": rng.choice([70, 76, 82, 88]),
                "sample_count": rng.choice([4, 6, 8]),
            }
        epoch_options = [4, 6, 8] if profile == "tiny" else [12, 16, 20, 24, 30]
        return {
            "preset": "custom",
            "block_size": rng.choice([32, 48, 64, 96]),
            "latent_dim": rng.choice([64, 96, 128, 192, 256]),
            "model_base_filters": rng.choice([32, 48, 64, 80]),
            "model_kernel_size": rng.choice([3, 5]),
            "batch_size": rng.choice([4, 6, 8]),
            "epochs": rng.choice(epoch_options),
            "lr": round(sample_log_uniform(rng, 5e-5, 3.2e-4), 7),
        }

    if modality == "audio":
        epoch_options = [4, 6, 8] if profile == "tiny" else [10, 14, 18, 24]
        return {
            "preset": "custom",
            "latent_dim": rng.choice([160, 224, 320, 384, 512]),
            "model_base_filters": rng.choice([32, 40, 48, 56]),
            "model_kernel_size": rng.choice([3, 5, 7]),
            "batch_size": rng.choice([8, 12, 16]),
            "epochs": rng.choice(epoch_options),
            "lr": round(sample_log_uniform(rng, 5e-5, 2.5e-4), 7),
            "clip_seconds": rng.choice([0.5, 0.75, 1.0]),
        }

    epoch_options = [4, 6, 8] if profile == "tiny" else [8, 12, 16, 20]
    frame_options = [4, 6, 8] if profile == "tiny" else [4, 6, 8]
    return {
        "preset": "custom",
        "latent_dim": rng.choice([192, 256, 320, 384, 512]),
        "model_base_filters": rng.choice([16, 24, 32, 40]),
        "model_kernel_size": rng.choice([3, 5]),
        "batch_size": rng.choice([2, 3, 4]),
        "epochs": rng.choice(epoch_options),
        "lr": round(sample_log_uniform(rng, 4e-5, 2.0e-4), 7),
        "frames": rng.choice(frame_options),
        "height": rng.choice([32, 48, 64]),
        "width": rng.choice([32, 48, 64]),
    }


def _get_config_fixed_args(
    config: dict[str, Any],
    modality: str,
    profile: str,
    image_trainer: str,
) -> list[str] | None:
    fixed_args = config.get("fixed_resource_args")
    if not isinstance(fixed_args, dict):
        return None

    mod_entry = fixed_args.get(modality)
    if modality == "image" and isinstance(mod_entry, dict):
        mod_entry = mod_entry.get(image_trainer)
    if not isinstance(mod_entry, dict):
        return None

    profile_entry = mod_entry.get(profile)
    if not isinstance(profile_entry, list):
        return None
    return [str(x) for x in profile_entry]


def fixed_resource_args(
    modality: str,
    profile: str,
    project_root: Path,
    image_trainer: str,
    search_config: dict[str, Any] | None = None,
) -> list[str]:
    cfg = search_config or {}
    cfg_args = _get_config_fixed_args(cfg, modality, profile, image_trainer)
    if cfg_args is not None:
        return cfg_args

    if modality == "image":
        cifar_root = project_root / "data" / "cifar10 2"
        train_dir = cifar_root / "train"
        val_dir = cifar_root / "val"
        test_dir = cifar_root / "test"
        if image_trainer == "lossy":
            if profile == "tiny":
                return [
                    "--train-dir",
                    str(train_dir),
                    "--val-dir",
                    str(val_dir),
                    "--test-dir",
                    str(test_dir),
                    "--real-only",
                    "--target-psnr",
                    "22",
                    "--target-compression-ratio",
                    "0.90",
                    "--no-export-tflite",
                ]
            return [
                "--train-dir",
                str(train_dir),
                "--val-dir",
                str(val_dir),
                "--test-dir",
                str(test_dir),
                "--real-only",
                "--target-psnr",
                "30",
                "--target-compression-ratio",
                "0.85",
                "--no-export-tflite",
            ]
        if profile == "tiny":
            return [
                "--train-dir",
                str(train_dir),
                "--val-dir",
                str(val_dir),
                "--test-dir",
                str(test_dir),
                "--real-only",
                "--target-psnr",
                "22",
                "--no-export-tflite",
            ]
        return [
            "--train-dir",
            str(train_dir),
            "--val-dir",
            str(val_dir),
            "--test-dir",
            str(test_dir),
            "--real-only",
            "--target-psnr",
            "30",
            "--no-export-tflite",
        ]

    if modality == "audio":
        audio_dir = project_root / "data" / "AudioData" / "ESC-50-master" / "audio"
        if profile == "tiny":
            return [
                "--data-dir",
                str(audio_dir),
                "--real-file-limit",
                "400",
                "--target-psnr",
                "24",
                "--no-export-tflite",
            ]
        return [
            "--data-dir",
            str(audio_dir),
            "--real-file-limit",
            "1600",
            "--target-psnr",
            "30",
            "--no-export-tflite",
        ]

    video_dir = project_root / "data" / "VIDEO DATA"
    if profile == "tiny":
        return [
            "--data-dir",
            str(video_dir),
            "--real-max-videos",
            "4",
            "--real-max-clips",
            "220",
            "--fps",
            "10",
            "--target-psnr",
            "22",
            "--no-export-tflite",
        ]
    return [
        "--data-dir",
        str(video_dir),
        "--real-max-videos",
        "10",
        "--real-max-clips",
        "900",
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


def read_metric_score(report_path: Path, modality: str) -> float | None:
    payload = read_markdown_json_report(report_path)
    if payload is None:
        return None
    metric_key = "snr_db_metric" if modality == "audio" else "psnr_metric"
    value = payload.get("test_metrics", {}).get(metric_key)
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
    image_trainer: str,
    search_config: dict[str, Any] | None = None,
) -> list[str]:
    cmd = [python_exec, str(trainer_script)]
    cmd.extend(params_to_cli_args(params))
    cmd.extend(["--output-root", str(trial_dir)])
    cmd.extend(fixed_resource_args(modality, profile, project_root, image_trainer, search_config=search_config))
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
    if args.modality != "image" and args.image_trainer != "standard":
        print("[!] --image-trainer is only used with --modality image; proceeding with modality defaults.")
    rng = random.Random(args.seed)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    search_config, search_config_path = load_search_config(project_root, args.search_config)
    script_path = script_dir / modality_to_script(args.modality, args.image_trainer)
    report_name = modality_to_report_name(args.modality)

    if not script_path.exists():
        raise FileNotFoundError(f"Trainer script not found: {script_path}")

    search_leaf = args.modality
    if args.modality == "image" and args.image_trainer == "lossy":
        search_leaf = "image_lossy"
    search_root = project_root / args.output_root / search_leaf
    search_root.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    budget_sec = max(1.0, args.max_minutes * 60.0)
    results: list[TrialResult] = []

    print(
        f"[*] Starting random search: modality={args.modality}, image_trainer={args.image_trainer}, "
        f"trials={args.trials}, budget={args.max_minutes:.1f} min"
    )

    requested_trials = args.max_total_trials if args.until_goal else args.trials
    goal_reached = False
    progress = SearchProgress(total=requested_trials)

    for trial_idx in range(1, requested_trials + 1):
        if not args.until_goal and trial_idx > args.trials:
            break
        elapsed = time.time() - started_at
        if elapsed >= budget_sec:
            print(f"[!] Budget reached after {trial_idx - 1} completed trial(s).")
            break

        progress_status = "running"
        try:
            trial_params = sample_params(
                args.modality,
                rng,
                args.resource_profile,
                args.image_trainer,
                search_config=search_config,
            )
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
                progress_status = f"trial {trial_idx:03d} simulated score {score:.3f}"
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
                image_trainer=args.image_trainer,
                search_config=search_config,
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
                progress_status = f"trial {trial_idx:03d} failed (exit {proc.returncode})"
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
                progress_status = f"trial {trial_idx:03d} failed (missing report)"
                continue

            score = read_metric_score(report_path, args.modality)
            if score is None:
                results.append(
                    TrialResult(
                        trial_id=trial_idx,
                        params=trial_params,
                        score_psnr=float("-inf"),
                        duration_sec=duration,
                        report_path=str(report_path),
                        run_dir=str(report_path.parent),
                        status="failed",
                        error="target metric missing in report",
                    )
                )
                print("    [!] Failed (target metric not found in report)")
                progress_status = f"trial {trial_idx:03d} failed (missing metric)"
                continue

            result = TrialResult(
                trial_id=trial_idx,
                params=trial_params,
                score_psnr=score,
                duration_sec=duration,
                report_path=str(report_path),
                run_dir=str(report_path.parent),
                status="ok",
            )
            results.append(result)
            metric_label = "SNR(dB)" if args.modality == "audio" else "PSNR"
            print(f"    [+] {metric_label}={score:.3f} ({duration:.1f}s)")
            progress_status = f"trial {trial_idx:03d} ok {metric_label} {score:.3f}"

            if args.until_goal and score >= args.goal_psnr:
                goal_reached = True
                print(f"[+] Goal reached ({metric_label} {score:.3f} >= {args.goal_psnr:.3f}) at trial {trial_idx:03d}")
                progress_status = f"goal reached at trial {trial_idx:03d}"
                break
        finally:
            progress.update(trial_idx, progress_status)

    progress.finish()

    successful = [r for r in results if math.isfinite(r.score_psnr)]
    successful_sorted = sorted(successful, key=lambda r: r.score_psnr, reverse=True)

    best = successful_sorted[0] if successful_sorted else None

    key_candidates = {
        "image": ["block_size", "latent_dim", "model_base_filters", "model_kernel_size", "batch_size", "epochs", "lr"],
        "audio": ["latent_dim", "model_base_filters", "model_kernel_size", "batch_size", "epochs", "lr", "clip_seconds"],
        "video": ["latent_dim", "model_base_filters", "model_kernel_size", "batch_size", "epochs", "lr", "frames", "height", "width"],
    }
    if args.modality == "image" and args.image_trainer == "lossy":
        key_candidates["image"] = [
            "block_size",
            "latent_dim",
            "model_base_filters",
            "model_kernel_size",
            "batch_size",
            "epochs",
            "lr",
            "rate_lambda",
            "jpeg_quality",
            "sample_count",
        ]

    summary_cfg = search_config.get("summary_keys") if isinstance(search_config, dict) else None
    if isinstance(summary_cfg, dict):
        override = summary_cfg.get(args.modality)
        if args.modality == "image" and isinstance(override, dict):
            override = override.get(args.image_trainer)
        if isinstance(override, list) and override:
            key_candidates[args.modality] = [str(x) for x in override]

    ranges = summarize_ranges(successful_sorted, key_candidates[args.modality], top_n=max(1, args.top_k))

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": args.modality,
        "image_trainer": args.image_trainer,
        "resource_profile": args.resource_profile,
        "search_config_path": None if search_config_path is None else str(search_config_path),
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


