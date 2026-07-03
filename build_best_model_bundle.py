#!/usr/bin/env python3
import datetime
import glob
import json
import os
import shutil

from report_markdown import read_markdown_json_report

BASE = "/Users/tyejaedon/PycharmProjects/AI_Compressor/models"


def best_report(pattern: str):
    pick = None
    for path in glob.glob(os.path.join(BASE, pattern), recursive=True):
        data = read_markdown_json_report(path)
        if data is None:
            continue
        psnr = data.get("test_metrics", {}).get("psnr_metric")
        if not isinstance(psnr, (int, float)):
            continue
        if pick is None or psnr > pick[0]:
            pick = (float(psnr), path, data)
    return pick


def copy_run_artifacts(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith((".json", ".md", ".tflite", ".h5", ".keras", ".png", ".wav")):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst_dir, name))
        copied.append(name)
    return copied


def main():
    out_root = os.path.join(BASE, "production_bundle")
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, datetime.datetime.now().strftime("best_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    patterns = {
        "image": "**/evaluation_report.md",
        "audio": "**/audio_evaluation_report.md",
        "video": "**/video_evaluation_report.md",
    }

    metadata = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "bundle_dir": out_dir,
        "selection_strategy": "highest test_metrics.psnr_metric per modality",
        "models": {},
    }

    for modality, pattern in patterns.items():
        best = best_report(pattern)
        if best is None:
            continue

        psnr, report_path, report_data = best
        run_dir = os.path.dirname(report_path)
        copied_files = copy_run_artifacts(run_dir, os.path.join(out_dir, modality))

        metadata["models"][modality] = {
            "psnr_metric": psnr,
            "source_report": os.path.relpath(report_path, BASE),
            "source_run_dir": os.path.relpath(run_dir, BASE),
            "test_metrics": report_data.get("test_metrics", {}),
            "copied_files": copied_files,
        }

    # If best audio run lacks TFLite, add best deployable fallback.
    audio_info = metadata["models"].get("audio")
    if audio_info and "audio_autoencoder.tflite" not in audio_info.get("copied_files", []):
        deploy_best = None
        for report in glob.glob(os.path.join(BASE, "**/audio_evaluation_report.md"), recursive=True):
            run_dir = os.path.dirname(report)
            tflite = os.path.join(run_dir, "audio_autoencoder.tflite")
            if not os.path.exists(tflite):
                continue
            data = read_markdown_json_report(report)
            if data is None:
                continue
            psnr = data.get("test_metrics", {}).get("psnr_metric")
            if not isinstance(psnr, (int, float)):
                continue
            if deploy_best is None or psnr > deploy_best[0]:
                deploy_best = (float(psnr), run_dir, report)

        if deploy_best is not None:
            psnr, run_dir, report = deploy_best
            deploy_dir = os.path.join(out_dir, "audio_deployable_tflite")
            os.makedirs(deploy_dir, exist_ok=True)
            for name in [
                "audio_autoencoder.tflite",
                "audio_model.weights.h5",
                "best_audio_model.keras",
                "audio_evaluation_report.md",
            ]:
                src = os.path.join(run_dir, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(deploy_dir, name))

            metadata["audio_deployable_fallback"] = {
                "reason": "best-by-PSNR audio run had no TFLite artifact",
                "source_run_dir": os.path.relpath(run_dir, BASE),
                "source_report": os.path.relpath(report, BASE),
                "psnr_metric": psnr,
                "copied_dir": os.path.relpath(deploy_dir, BASE),
            }

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(out_dir)
    print(metadata_path)


if __name__ == "__main__":
    main()

