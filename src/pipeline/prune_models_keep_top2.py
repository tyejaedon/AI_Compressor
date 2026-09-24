#!/usr/bin/env python3
"""Prune model artifacts and keep only top-N model runs by PSNR.

- Scans report files under models/
- Ranks run directories by test PSNR
- Keeps top N run dirs (default: 2)
- Deletes other ranked run dirs, including bundle copies
- In kept runs, keeps at most one primary model file and one report file
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reporting"))
from report_markdown import read_markdown_json_report


REPORT_NAMES = {"evaluation_report.md", "audio_evaluation_report.md", "video_evaluation_report.md"}
MODEL_EXTS = {".keras", ".tflite", ".h5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune models to keep only top runs by PSNR.")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--keep", type=int, default=2, help="How many top runs to keep")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=["models/optimal_m1_full"],
        help="Directory prefixes to never delete (can be passed multiple times)",
    )
    parser.add_argument("--apply", action="store_true", help="Apply deletions (default is dry-run)")
    return parser.parse_args()


def should_exclude(path: Path, excludes: list[Path]) -> bool:
    return any(path == ex or ex in path.parents for ex in excludes)


def collect_ranked_runs(models_root: Path, excludes: list[Path]) -> list[tuple[float, Path, Path]]:
    best_by_run: dict[Path, tuple[float, Path]] = {}
    for rp in models_root.rglob("*"):
        if rp.name not in REPORT_NAMES:
            continue
        run_dir = rp.parent
        if should_exclude(run_dir, excludes):
            continue
        payload = read_markdown_json_report(rp)
        if not payload:
            continue
        psnr = payload.get("test_metrics", {}).get("psnr_metric")
        if not isinstance(psnr, (int, float)):
            continue
        current = best_by_run.get(run_dir)
        if current is None or float(psnr) > current[0]:
            best_by_run[run_dir] = (float(psnr), rp)

    ranked = sorted(((v[0], k, v[1]) for k, v in best_by_run.items()), key=lambda t: t[0], reverse=True)
    return ranked


def choose_primary_model_file(run_dir: Path) -> Path | None:
    files = [p for p in run_dir.iterdir() if p.is_file() and p.suffix in MODEL_EXTS]
    if not files:
        return None
    pref = [p for p in files if p.suffix == ".keras"]
    if pref:
        return sorted(pref)[0]
    pref = [p for p in files if p.suffix == ".tflite"]
    if pref:
        return sorted(pref)[0]
    return sorted(files)[0]


def prune_kept_run(run_dir: Path, keep_report: Path, apply: bool) -> list[str]:
    actions: list[str] = []
    primary_model = choose_primary_model_file(run_dir)

    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in REPORT_NAMES and p != keep_report:
            actions.append(f"delete file {p}")
            if apply:
                p.unlink(missing_ok=True)
            continue
        if p.suffix in MODEL_EXTS and p != primary_model:
            actions.append(f"delete file {p}")
            if apply:
                p.unlink(missing_ok=True)
            continue

    # Remove common heavy artifact directories in kept runs.
    for sub in ["lossy_samples"]:
        d = run_dir / sub
        if d.exists() and d.is_dir():
            actions.append(f"delete dir {d}")
            if apply:
                shutil.rmtree(d, ignore_errors=True)
    return actions


def main() -> int:
    args = parse_args()
    models_root = args.models_root.resolve()
    excludes = [Path(p).resolve() for p in args.exclude_prefix]

    ranked = collect_ranked_runs(models_root, excludes)
    if not ranked:
        print("[!] No ranked runs found from report files; nothing to do.")
        return 0

    keep_n = max(1, int(args.keep))
    keep_entries = ranked[:keep_n]
    keep_runs = {run for _, run, _ in keep_entries}

    print("[+] Keeping top runs:")
    for i, (psnr, run, _) in enumerate(keep_entries, start=1):
        print(f"  {i}. {psnr:.3f} | {run}")

    delete_runs = [run for _, run, _ in ranked[keep_n:]]
    print(f"[+] Ranked runs to delete: {len(delete_runs)}")

    manifest = {
        "keep": [{"psnr": psnr, "run_dir": str(run), "report": str(rp)} for psnr, run, rp in keep_entries],
        "delete": [str(run) for run in delete_runs],
        "applied": bool(args.apply),
    }
    manifest_path = models_root / "prune_manifest_keep_top2.json"
    if args.apply:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.apply:
        print("[dry-run] No files deleted. Re-run with --apply to execute.")
        return 0

    # Delete lower-ranked run dirs.
    for run in delete_runs:
        if should_exclude(run, excludes):
            continue
        shutil.rmtree(run, ignore_errors=True)

    # Prune kept runs to a single primary model file + one report.
    for _, run, rp in keep_entries:
        prune_kept_run(run, rp, apply=True)

    # Remove old production bundle copies entirely to avoid redundant model duplicates.
    bundle_root = models_root / "production_bundle"
    if bundle_root.exists():
        for d in bundle_root.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

    print("[+] Prune complete.")
    print(f"[+] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

