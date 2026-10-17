#!/usr/bin/env python3
"""Generate a master markdown report of major runs and lossy image baseline metrics."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

from report_markdown import read_markdown_json_report


WORKSPACE = Path("/Users/tyejaedon/PycharmProjects/AI_Compressor")
MODELS_DIR = WORKSPACE / "models"


MAJOR_RUN_FILES = [
    ("Image", "local_run/20260626_005606/evaluation_report.md", "synthetic_best"),
    ("Image", "local_real_run/20260629_101835/evaluation_report.md", "real_prod"),
    ("Image", "local_run_tuned_fast/20260629_065112/evaluation_report.md", "tuned_fast"),
    ("Image", "image_realdata_psnr30/20260629_150224/evaluation_report.md", "realdata_psnr30"),
    ("Image", "image_realdata_psnr30_try2/20260629_150721/evaluation_report.md", "realdata_psnr30_try2"),
    ("Audio", "audio_local_run/20260627_161410/audio_evaluation_report.md", "baseline"),
    ("Audio", "audio_real_run_v2/20260629_125927/audio_evaluation_report.md", "real_run_v2"),
    ("Audio", "audio_realdata_opt_smoke/20260629_145612/audio_evaluation_report.md", "realdata_opt_smoke"),
    ("Video", "video_local_run_tuned_fast2/20260629_070559/video_evaluation_report.md", "tuned_fast2"),
    ("Video", "video_realdata_opt_smoke/20260629_145842/video_evaluation_report.md", "realdata_opt_smoke"),
    ("Bundle", "production_bundle/best_20260629_142034/metadata.json", "production_bundle"),
]


def read_json(path: Path) -> dict[str, Any] | None:
    return read_markdown_json_report(path)


def fmt(v: Any, ndigits: int = 4) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{ndigits}f}"
    return "-"


def collect_major_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality, rel, tag in MAJOR_RUN_FILES:
        path = MODELS_DIR / rel
        data = read_json(path)
        if data is None:
            rows.append(
                {
                    "modality": modality,
                    "tag": tag,
                    "path": str(path.relative_to(WORKSPACE)),
                    "exists": False,
                }
            )
            continue

        if rel.endswith("metadata.json"):
            rows.append(
                {
                    "modality": modality,
                    "tag": tag,
                    "path": str(path.relative_to(WORKSPACE)),
                    "exists": True,
                    "bundle": data,
                }
            )
            continue

        tm = data.get("test_metrics", {})
        rows.append(
            {
                "modality": modality,
                "tag": tag,
                "path": str(path.relative_to(WORKSPACE)),
                "exists": True,
                "loss": tm.get("loss"),
                "mse": tm.get("mse"),
                "psnr_metric": tm.get("psnr_metric"),
                "ssim_metric": tm.get("ssim_metric"),
                "snr_db_metric": tm.get("snr_db_metric"),
                "actual_compression_ratio": data.get("actual_compression_ratio"),
                "estimated_input_to_latent_ratio": data.get("estimated_input_to_latent_ratio"),
                "latent_bit_depth": data.get("config", {}).get("latent_bit_depth"),
                "loss_profile": data.get("config", {}).get("loss_profile", "balanced"),
            }
        )
    return rows


def collect_image_paths(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    ta = tf.convert_to_tensor(a[None, ...], dtype=tf.float32)
    tb = tf.convert_to_tensor(b[None, ...], dtype=tf.float32)
    return float(tf.image.ssim(ta, tb, max_val=1.0).numpy()[0])


def encode_decode(img: Image.Image, fmt_name: str, quality: int) -> tuple[np.ndarray, int]:
    import io

    buff = io.BytesIO()
    if fmt_name == "JPEG":
        img.save(buff, format="JPEG", quality=quality, optimize=True)
    elif fmt_name == "WEBP":
        img.save(buff, format="WEBP", quality=quality, method=6)
    else:
        raise ValueError(fmt_name)
    data = buff.getvalue()
    recon = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(recon).astype(np.float32) / 255.0
    return arr, len(data)


def benchmark_lossy_images() -> list[dict[str, Any]]:
    image_root = WORKSPACE / "data/ImageData/archive/data"
    paths = collect_image_paths(image_root)
    if not paths:
        return []

    rng = np.random.default_rng(42)
    sample_n = min(24, len(paths))
    indices = rng.choice(len(paths), size=sample_n, replace=False)
    chosen = [paths[int(i)] for i in indices]

    configs = [
        ("JPEG", 95),
        ("JPEG", 85),
        ("JPEG", 75),
        ("JPEG", 50),
        ("WEBP", 90),
        ("WEBP", 75),
        ("WEBP", 50),
    ]

    out: list[dict[str, Any]] = []
    for fmt_name, quality in configs:
        psnrs, ssims, ratios = [], [], []
        for p in chosen:
            img = Image.open(p).convert("RGB")
            # Keep runtime practical but preserve content.
            if max(img.size) > 1280:
                img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)

            orig = np.asarray(img).astype(np.float32) / 255.0
            recon, compressed_size = encode_decode(img, fmt_name, quality)

            psnrs.append(psnr_np(orig, recon))
            ssims.append(ssim_np(orig, recon))
            ratios.append(compressed_size / max(orig.size, 1))

        out.append(
            {
                "codec": fmt_name,
                "quality": quality,
                "samples": sample_n,
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "size_ratio_vs_raw": float(np.mean(ratios)),
            }
        )
    return out


def plot_audio_frontier(models_dir: Path, out_path: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio_runs = []
    for report_file in models_dir.rglob("audio_evaluation_report.md"):
        data = read_json(report_file)
        if not data or not isinstance(data, dict):
            continue
        tm = data.get("test_metrics", {})
        snr = tm.get("snr_db_metric")
        psnr = tm.get("psnr_metric")
        act_ratio = data.get("actual_compression_ratio")
        est_ratio = data.get("estimated_input_to_latent_ratio")
        cfg = data.get("config", {})
        bit_depth = cfg.get("latent_bit_depth", 0)
        profile = cfg.get("loss_profile", "balanced")
        rel_path = str(report_file.parent.relative_to(models_dir))

        ratio = act_ratio if act_ratio is not None else est_ratio
        if ratio is not None and (snr is not None or psnr is not None):
            audio_runs.append({
                "run": rel_path,
                "ratio": float(ratio),
                "actual_ratio": act_ratio,
                "estimated_ratio": est_ratio,
                "snr": float(snr) if snr is not None else None,
                "psnr": float(psnr) if psnr is not None else None,
                "bit_depth": bit_depth,
                "loss_profile": profile,
            })

    if not audio_runs:
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in audio_runs:
        y_val = r["snr"] if r["snr"] is not None else r["psnr"]
        lbl = f"{r['run'][:22]} ({r['bit_depth']}b)"
        ax.scatter(r["ratio"], y_val, s=100, alpha=0.8, zorder=3)
        ax.annotate(lbl, (r["ratio"], y_val), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_title("Audio Compression Ratio vs SNR Frontier", fontsize=12)
    ax.set_xlabel("Compression Ratio (higher is better)", fontsize=10)
    ax.set_ylabel("SNR dB / PSNR (higher is better)", fontsize=10)
    if any(r["ratio"] > 100 for r in audio_runs):
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return audio_runs


def write_markdown(rows: list[dict[str, Any]], lossy: list[dict[str, Any]], audio_frontier: list[dict[str, Any]], out_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    lines: list[str] = []
    lines.append("# Master Training & Compression Report")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append("## Major Runs")
    lines.append("")
    lines.append("| Modality | Run Tag | Report Path | PSNR | SSIM | MSE | Loss | SNR dB |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if not r.get("exists"):
            lines.append(f"| {r['modality']} | {r['tag']} | `{r['path']}` | - | - | - | - | - |")
            continue
        if r["modality"] == "Bundle":
            lines.append(f"| Bundle | {r['tag']} | `{r['path']}` | - | - | - | - | - |")
            continue
        lines.append(
            "| {mod} | {tag} | `{path}` | {psnr} | {ssim} | {mse} | {loss} | {snr} |".format(
                mod=r["modality"],
                tag=r["tag"],
                path=r["path"],
                psnr=fmt(r.get("psnr_metric"), 3),
                ssim=fmt(r.get("ssim_metric"), 4),
                mse=fmt(r.get("mse"), 6),
                loss=fmt(r.get("loss"), 6),
                snr=fmt(r.get("snr_db_metric"), 3),
            )
        )

    lines.append("")
    lines.append("## Production Bundle Selection")
    lines.append("")
    bundle = next((r for r in rows if r.get("modality") == "Bundle" and r.get("exists")), None)
    if bundle:
        b = bundle["bundle"]
        lines.append(f"- Bundle: `{bundle['path']}`")
        lines.append(f"- Strategy: `{b.get('selection_strategy', '-')}`")
        models = b.get("models", {})
        for k in ["image", "audio", "video"]:
            mk = models.get(k, {})
            lines.append(
                f"- {k.title()}: source `{mk.get('source_run_dir', '-')}`, PSNR `{fmt(mk.get('psnr_metric'), 3)}`"
            )
    else:
        lines.append("- No production bundle metadata found.")

    lines.append("")
    lines.append("## JPEG / Other Lossy Image Baseline (Real Data)")
    lines.append("")
    lines.append("Dataset sampled from: `data/ImageData/archive/data`")
    lines.append("")
    lines.append("| Codec | Quality | Samples | Mean PSNR | Mean SSIM | Mean Size Ratio vs Raw RGB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in lossy:
        lines.append(
            f"| {c['codec']} | {c['quality']} | {c['samples']} | {c['psnr_mean']:.3f} | {c['ssim_mean']:.4f} | {c['size_ratio_vs_raw']:.4f} |"
        )

    lines.append("")
    if audio_frontier:
        lines.append("")
        lines.append("## Audio Compression-Ratio vs SNR Frontier (Milestone A5)")
        lines.append("")
        lines.append("![Audio Frontier](audio_compression_snr_frontier.png)")
        lines.append("")
        lines.append("| Run | Bit Depth | Loss Profile | Actual Ratio | Est Ratio | SNR (dB) | PSNR |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for af in audio_frontier:
            snr_s = f"{af['snr']:.2f}" if af.get("snr") is not None else "-"
            psnr_s = f"{af['psnr']:.2f}" if af.get("psnr") is not None else "-"
            act_s = f"{af['actual_ratio']:.2f}x" if af.get("actual_ratio") is not None else "-"
            est_s = f"{af['estimated_ratio']:.2f}x" if af.get("estimated_ratio") is not None else "-"
            lines.append(f"| `{af['run']}` | {af.get('bit_depth', 0)}b | `{af.get('loss_profile', 'balanced')}` | {act_s} | {est_s} | {snr_s} | {psnr_s} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- JPEG/WebP baselines are classic lossy codecs on full-resolution images.")
    lines.append("- Model PSNR/SSIM values above come from run reports and may use resized model input dimensions.")
    lines.append("- Audio MP3 benchmark in `audio_real_run_v2` is marked skipped due to ffmpeg unavailability at that run time.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_major_runs()
    lossy = benchmark_lossy_images()

    out_dir = MODELS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "MASTER_RUN_REPORT.md"
    frontier_plot_path = out_dir / "audio_compression_snr_frontier.png"
    audio_frontier = plot_audio_frontier(MODELS_DIR, frontier_plot_path)

    write_markdown(rows, lossy, audio_frontier, out_path)

    print(str(out_path))


if __name__ == "__main__":
    main()

