#!/usr/bin/env python3
"""Keep best model run per category and delete all other model artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_NAMES = {"evaluation_report.md", "audio_evaluation_report.md", "video_evaluation_report.md"}
KEEP_EXTS = {".md", ".json", ".keras", ".h5", ".tflite", ".png", ".wav", ".txt", ".csv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keep best run for image_upscaler/image_lossy/audio/video and delete the rest.")
    p.add_argument(
        "--models-root",
        type=Path,
        default=Path("/Users/tyejaedon/PycharmProjects/AI_Compressor/models"),
        help="Root models directory",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletion. Without this flag, performs a dry run.",
    )
    return p.parse_args()


def read_report_payload(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("```json")
    if start != -1:
        start = text.find("{", start)
        end = text.find("\n```", start)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def classify(report_name: str, payload: dict[str, Any]) -> str:
    if report_name == "audio_evaluation_report.md":
        return "audio"
    if report_name == "video_evaluation_report.md":
        return "video"
    if "lossy_size_stats" in payload or "matched_psnr_baseline_benchmark" in payload:
        return "image_lossy"
    return "image_upscaler"


def score_for_category(category: str, payload: dict[str, Any]) -> tuple[float | None, str | None]:
    test_metrics = payload.get("test_metrics", {})
    if isinstance(test_metrics, dict) and category == "audio":
        snr = test_metrics.get("snr_db_metric")
        if isinstance(snr, (int, float)):
            return float(snr), "snr_db_metric"
    if isinstance(test_metrics, dict):
        psnr = test_metrics.get("psnr_metric")
        if isinstance(psnr, (int, float)):
            return float(psnr), "psnr_metric"
    return None, None


def find_best_runs(models_root: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for report in models_root.rglob("*.md"):
        if report.name not in REPORT_NAMES:
            continue
        payload = read_report_payload(report)
        if payload is None:
            continue
        category = classify(report.name, payload)
        score, metric = score_for_category(category, payload)
        if score is None:
            continue
        current = best.get(category)
        if current is None or score > current["score"]:
            best[category] = {
                "category": category,
                "metric": metric,
                "score": score,
                "report_path": str(report),
                "run_dir": str(report.parent),
            }
    return best


def copy_selected_runs(best: dict[str, dict[str, Any]], bundle_dir: Path) -> dict[str, dict[str, Any]]:
    copied_info: dict[str, dict[str, Any]] = {}
    for category, info in sorted(best.items()):
        src = Path(info["run_dir"])
        dst = bundle_dir / category
        dst.mkdir(parents=True, exist_ok=True)
        copied_count = 0
        for f in src.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in KEEP_EXTS:
                continue
            rel = f.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            copied_count += 1
        enriched = dict(info)
        enriched["copied_file_count"] = copied_count
        copied_info[category] = enriched
    return copied_info


def prune_models_root(models_root: Path, keep_bundle_dir: Path) -> int:
    deleted = 0
    for child in sorted(models_root.iterdir()):
        if child.name == "production_bundle":
            for sub in sorted(child.iterdir()):
                if sub.resolve() == keep_bundle_dir.resolve():
                    continue
                if sub.is_dir():
                    shutil.rmtree(sub, ignore_errors=True)
                else:
                    sub.unlink(missing_ok=True)
                deleted += 1
            continue

        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        deleted += 1
    return deleted


def main() -> int:
    args = parse_args()
    models_root = args.models_root.resolve()
    if not models_root.exists():
        raise SystemExit(f"models root does not exist: {models_root}")

    best = find_best_runs(models_root)
    if not best:
        raise SystemExit("No ranked reports found; aborting prune.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = models_root / "production_bundle" / f"top_models_{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    selected = copy_selected_runs(best, bundle_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "models_root": str(models_root),
        "bundle_dir": str(bundle_dir),
        "selection_strategy": "best metric per category (audio prefers snr_db_metric; others use psnr_metric)",
        "selected": selected,
        "applied": bool(args.apply),
    }

    if args.apply:
        manifest["deleted_top_level_entries"] = prune_models_root(models_root, bundle_dir)

    manifest_path = bundle_dir / "metadata.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Bundle: {bundle_dir}")
    print(f"Metadata: {manifest_path}")
    for category, info in sorted(selected.items()):
        print(f"{category}: {info['metric']}={info['score']:.6f}")
    if args.apply:
        print(f"Deleted entries: {manifest['deleted_top_level_entries']}")
    else:
        print("Dry run only: no deletions applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

