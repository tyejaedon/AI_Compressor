#!/usr/bin/env python3
"""End-to-end production pipeline for image/audio models.

Stages:
1) Random-search hyperparameter gathering
2) Sequential training: image upscaler -> image lossy -> audio (optional video)
3) Production bundle copy into a fresh folder
4) Optional prune of all other report-backed model runs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from report_markdown import read_markdown_json_report, write_markdown_json_report


REPORT_BY_STAGE = {
    "image_standard": "evaluation_report.md",
    "image_lossy": "evaluation_report.md",
    "audio": "audio_evaluation_report.md",
    "video": "video_evaluation_report.md",
}

REPORT_NAMES = {
    "evaluation_report.md",
    "audio_evaluation_report.md",
    "video_evaluation_report.md",
}

COPY_EXTS = {".json", ".md", ".tflite", ".h5", ".keras", ".png", ".wav"}


@dataclass
class StageResult:
    name: str
    command: list[str]
    run_dir: Path | None
    report_path: Path | None
    metric: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full production pipeline for CIFAR image and ESC-50 audio models.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--python", type=str, default=sys.executable)

    parser.add_argument("--search-config", type=str, default="documentation/random_search_profile.json")
    parser.add_argument("--search-output-root", type=str, default="models/production_pipeline/search")
    parser.add_argument("--train-output-root", type=str, default="models/production_pipeline/runs")
    parser.add_argument("--production-root", type=str, default="models/production_bundle")

    parser.add_argument("--trials-image-standard", type=int, default=10)
    parser.add_argument("--trials-image-lossy", type=int, default=10)
    parser.add_argument("--trials-audio", type=int, default=8)
    parser.add_argument("--trials-video", type=int, default=6)

    parser.add_argument("--minutes-image-standard", type=float, default=30.0)
    parser.add_argument("--minutes-image-lossy", type=float, default=30.0)
    parser.add_argument("--minutes-audio", type=float, default=30.0)
    parser.add_argument("--minutes-video", type=float, default=30.0)

    parser.add_argument("--resource-profile", choices=["tiny", "balanced"], default="balanced")

    parser.add_argument("--include-video", action="store_true", help="Also train video in this pipeline run")
    parser.add_argument("--video-max-clips", type=int, default=200000, help="High clip cap used when --include-video is enabled")

    parser.add_argument("--apply-prune", action="store_true", help="Delete all non-production report-backed runs under models/")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and run search in simulated mode")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    pretty = " ".join(str(p) for p in cmd)
    print(f"[*] {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def to_cli_args(params: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def find_latest_summary(search_root: Path, leaf: str) -> Path | None:
    candidates = sorted((search_root / leaf).glob("search_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_report(run_root: Path, report_name: str) -> Path | None:
    candidates = sorted(run_root.rglob(report_name), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def metric_from_report(report_path: Path, stage: str) -> float | None:
    payload = read_markdown_json_report(report_path)
    if not payload:
        return None
    key = "snr_db_metric" if stage == "audio" else "psnr_metric"
    value = payload.get("test_metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def extract_best_params(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    best = data.get("best_trial") if isinstance(data, dict) else None
    params = best.get("params") if isinstance(best, dict) else None
    if not isinstance(params, dict):
        raise RuntimeError(f"No best_trial.params in {summary_path}")
    return params


def copy_run_artifacts(src_dir: Path, dst_dir: Path) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in COPY_EXTS:
            continue
        shutil.copy2(p, dst_dir / p.name)
        copied.append(p.name)
    return copied


def collect_report_run_dirs(models_root: Path) -> list[Path]:
    runs: set[Path] = set()
    for p in models_root.rglob("*"):
        if p.is_file() and p.name in REPORT_NAMES:
            runs.add(p.parent)
    return sorted(runs)


def run_search_for_stage(
    args: argparse.Namespace,
    project_root: Path,
    search_root: Path,
    stage: str,
    trials: int,
    minutes: float,
    extra_args: list[str],
) -> dict[str, Any]:
    cmd = [
        args.python,
        "random_search_hyperparams.py",
        "--modality",
        "image" if stage in {"image_standard", "image_lossy"} else stage,
        "--resource-profile",
        args.resource_profile,
        "--trials",
        str(trials),
        "--max-minutes",
        str(minutes),
        "--search-config",
        args.search_config,
        "--output-root",
        str(search_root),
    ]
    if stage == "image_lossy":
        cmd.extend(["--image-trainer", "lossy"])

    for token in extra_args:
        cmd.append(f"--extra-arg={token}")

    if args.dry_run:
        cmd.append("--dry-run")

    run_cmd(cmd, cwd=project_root, dry_run=False)

    leaf = "image" if stage == "image_standard" else ("image_lossy" if stage == "image_lossy" else stage)
    summary_path = find_latest_summary(search_root, leaf)
    if summary_path is None:
        raise RuntimeError(f"No search summary found for {stage} in {search_root / leaf}")
    print(f"[+] {stage} summary: {summary_path}")
    return extract_best_params(summary_path)


def run_train_stage(
    args: argparse.Namespace,
    project_root: Path,
    train_root: Path,
    stage: str,
    script_name: str,
    params: dict[str, Any],
    fixed_args: dict[str, Any],
) -> StageResult:
    stage_out = train_root / stage
    merged = {**params, **fixed_args, "output_root": str(stage_out)}
    cmd = [args.python, script_name, *to_cli_args(merged)]

    if args.dry_run:
        print(f"[*] {' '.join(cmd)}")
        return StageResult(name=stage, command=cmd, run_dir=None, report_path=None, metric=None)

    run_cmd(cmd, cwd=project_root, dry_run=False)
    report_name = REPORT_BY_STAGE[stage]
    report = find_latest_report(stage_out, report_name)
    if report is None:
        raise RuntimeError(f"No report found for {stage} under {stage_out}")
    metric = metric_from_report(report, stage)
    return StageResult(name=stage, command=cmd, run_dir=report.parent, report_path=report, metric=metric)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    search_root = (project_root / args.search_output_root / ts).resolve()
    train_root = (project_root / args.train_output_root / ts).resolve()
    production_dir = (project_root / args.production_root / f"production_{ts}").resolve()

    search_root.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    cifar_train = str((project_root / "data" / "cifar10 2" / "train").resolve())
    cifar_val = str((project_root / "data" / "cifar10 2" / "val").resolve())
    cifar_test = str((project_root / "data" / "cifar10 2" / "test").resolve())
    audio_dir = str((project_root / "data" / "AudioData" / "ESC-50-master" / "audio").resolve())
    video_dir = str((project_root / "data" / "VIDEO DATA").resolve())

    print(f"[+] Search root: {search_root}")
    print(f"[+] Train root: {train_root}")
    print(f"[+] Production dir: {production_dir}")

    search_params: dict[str, dict[str, Any]] = {}

    # 1) Gather hyperparams
    search_params["image_standard"] = run_search_for_stage(
        args,
        project_root,
        search_root,
        "image_standard",
        trials=max(1, int(args.trials_image_standard)),
        minutes=max(1.0, float(args.minutes_image_standard)),
        extra_args=[
            "--train-dir",
            cifar_train,
            "--val-dir",
            cifar_val,
            "--test-dir",
            cifar_test,
            "--real-only",
            "--no-export-tflite",
        ],
    )

    search_params["image_lossy"] = run_search_for_stage(
        args,
        project_root,
        search_root,
        "image_lossy",
        trials=max(1, int(args.trials_image_lossy)),
        minutes=max(1.0, float(args.minutes_image_lossy)),
        extra_args=[
            "--train-dir",
            cifar_train,
            "--val-dir",
            cifar_val,
            "--test-dir",
            cifar_test,
            "--real-only",
            "--no-export-tflite",
        ],
    )

    search_params["audio"] = run_search_for_stage(
        args,
        project_root,
        search_root,
        "audio",
        trials=max(1, int(args.trials_audio)),
        minutes=max(1.0, float(args.minutes_audio)),
        extra_args=[
            "--data-dir",
            audio_dir,
            "--real-file-limit",
            "0",
            "--no-export-tflite",
        ],
    )

    if args.include_video:
        search_params["video"] = run_search_for_stage(
            args,
            project_root,
            search_root,
            "video",
            trials=max(1, int(args.trials_video)),
            minutes=max(1.0, float(args.minutes_video)),
            extra_args=[
                "--data-dir",
                video_dir,
                "--real-max-videos",
                "0",
                "--real-max-clips",
                str(max(1, int(args.video_max_clips))),
                "--real-clip-stride",
                "1",
                "--no-export-tflite",
            ],
        )

    params_path = search_root / "selected_hyperparams.json"
    params_path.write_text(json.dumps(search_params, indent=2), encoding="utf-8")
    print(f"[+] Selected params: {params_path}")

    # 2) Train in requested order
    stage_results: list[StageResult] = []
    stage_results.append(
        run_train_stage(
            args,
            project_root,
            train_root,
            "image_standard",
            "train_autoencoder_image_local.py",
            search_params["image_standard"],
            {
                "preset": "custom",
                "train_dir": cifar_train,
                "val_dir": cifar_val,
                "test_dir": cifar_test,
                "real_only": True,
                "no_export_tflite": True,
            },
        )
    )

    stage_results.append(
        run_train_stage(
            args,
            project_root,
            train_root,
            "image_lossy",
            "train_autoencoder_image_lossy_local.py",
            search_params["image_lossy"],
            {
                "preset": "custom",
                "train_dir": cifar_train,
                "val_dir": cifar_val,
                "test_dir": cifar_test,
                "real_only": True,
                "no_export_tflite": True,
            },
        )
    )

    stage_results.append(
        run_train_stage(
            args,
            project_root,
            train_root,
            "audio",
            "train_autoencoder_audio_local.py",
            search_params["audio"],
            {
                "preset": "custom",
                "data_dir": audio_dir,
                "real_file_limit": 0,
                "no_export_tflite": True,
            },
        )
    )

    if args.include_video:
        stage_results.append(
            run_train_stage(
                args,
                project_root,
                train_root,
                "video",
                "train_autoencoder_video_local.py",
                search_params["video"],
                {
                    "preset": "custom",
                    "data_dir": video_dir,
                    "real_max_videos": 0,
                    "real_max_clips": max(1, int(args.video_max_clips)),
                    "real_clip_stride": 1,
                    "no_export_tflite": True,
                },
            )
        )

    # 3) Build production folder from selected runs
    copied_summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "search_root": str(search_root),
        "train_root": str(train_root),
        "production_dir": str(production_dir),
        "stages": {},
        "video_defaults": {
            "data_dir": video_dir,
            "real_max_videos": 0,
            "real_max_clips": int(max(1, args.video_max_clips)),
            "real_clip_stride": 1,
        },
    }

    if not args.dry_run:
        production_dir.mkdir(parents=True, exist_ok=True)

    for result in stage_results:
        if result.run_dir is None:
            copied_summary["stages"][result.name] = {
                "command": result.command,
                "status": "dry_run",
            }
            continue

        dst = production_dir / result.name
        copied = []
        if not args.dry_run:
            copied = copy_run_artifacts(result.run_dir, dst)

        copied_summary["stages"][result.name] = {
            "command": result.command,
            "source_run_dir": str(result.run_dir),
            "source_report": str(result.report_path) if result.report_path else None,
            "metric": result.metric,
            "copied_files": copied,
        }

    manifest_path = production_dir / "pipeline_manifest.md"
    if args.dry_run:
        print("[dry-run] Skipping production copy write and prune.")
        print(json.dumps(copied_summary, indent=2))
        return 0

    write_markdown_json_report(copied_summary, manifest_path, title="Production Pipeline Manifest")

    # 4) Optional prune of all non-production report-backed runs
    prune_deleted = 0
    if args.apply_prune:
        models_root = (project_root / "models").resolve()
        report_runs = collect_report_run_dirs(models_root)
        for run_dir in report_runs:
            if production_dir in run_dir.parents or run_dir == production_dir:
                continue
            shutil.rmtree(run_dir, ignore_errors=True)
            prune_deleted += 1

    print("\n[+] Pipeline complete")
    print(f"[+] Production dir: {production_dir}")
    print(f"[+] Manifest: {manifest_path}")
    if args.apply_prune:
        print(f"[+] Pruned run directories: {prune_deleted}")
    else:
        print("[+] Prune not applied (use --apply-prune to delete other report-backed runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


