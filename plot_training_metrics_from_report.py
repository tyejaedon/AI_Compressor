#!/usr/bin/env python3
"""Plot training/validation metrics from a markdown report with embedded JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_markdown import read_markdown_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training view metrics from a report file.")
    parser.add_argument("--report", required=True, help="Path to evaluation report (.md or .json)")
    parser.add_argument("--output", default="", help="Optional output PNG path")
    return parser.parse_args()


def _float_series(history: dict, key: str) -> list[float]:
    values = history.get(key, [])
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _plot_pair(ax, history: dict, train_key: str, val_key: str, title: str) -> None:
    train_vals = _float_series(history, train_key)
    val_vals = _float_series(history, val_key)

    if train_vals:
        ax.plot(range(1, len(train_vals) + 1), train_vals, label="train", linewidth=2)
    if val_vals:
        ax.plot(range(1, len(val_vals) + 1), val_vals, label="val", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    if train_vals or val_vals:
        ax.legend()


def _first_available_pair(history: dict, candidates: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    for train_key, val_key, title in candidates:
        if history.get(train_key) or history.get(val_key):
            return train_key, val_key, title
    return None


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    payload = read_markdown_json_report(report_path)
    if payload is None:
        raise RuntimeError(f"Could not parse report payload: {report_path}")

    history = payload.get("history", {})
    if not isinstance(history, dict) or not history:
        raise RuntimeError(f"No 'history' section found in report: {report_path}")

    out_path = Path(args.output) if args.output else report_path.with_name("video_training_view_metrics.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    _plot_pair(axes[0, 0], history, "loss", "val_loss", "Loss")
    _plot_pair(axes[0, 1], history, "mse", "val_mse", "MSE")
    _plot_pair(axes[0, 2], history, "psnr_metric", "val_psnr_metric", "PSNR")

    quality_pair = _first_available_pair(
        history,
        [
            ("ssim_metric", "val_ssim_metric", "SSIM"),
            ("snr_db_metric", "val_snr_db_metric", "SNR dB"),
        ],
    )
    if quality_pair is not None:
        train_key, val_key, title = quality_pair
        _plot_pair(axes[1, 0], history, train_key, val_key, title)
    else:
        axes[1, 0].set_title("Quality Metric")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)

    lr = _float_series(history, "learning_rate")
    if lr:
        axes[1, 1].plot(range(1, len(lr) + 1), lr, color="tab:purple", linewidth=2)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    tm = payload.get("test_metrics", {})
    quality_summary = "-"
    if "ssim_metric" in tm:
        quality_summary = tm.get("ssim_metric", "-")
        quality_name = "ssim_metric"
    elif "snr_db_metric" in tm:
        quality_summary = tm.get("snr_db_metric", "-")
        quality_name = "snr_db_metric"
    else:
        quality_name = "quality_metric"

    lines = [
        "Test Metrics",
        f"loss: {tm.get('loss', '-')}",
        f"mse: {tm.get('mse', '-')}",
        f"psnr_metric: {tm.get('psnr_metric', '-')}",
        f"{quality_name}: {quality_summary}",
    ]
    axes[1, 2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.suptitle("Training View Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(out_path)


if __name__ == "__main__":
    main()

