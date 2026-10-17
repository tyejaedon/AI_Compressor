#!/usr/bin/env python3
"""Quick smoke test for audio trainer covering baseline and Milestone A1 quantization."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add reporting to sys.path to read markdown reports
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / "src" / "reporting"))
from report_markdown import read_markdown_json_report


def run_audio_smoke(extra_args, output_dir):
    cmd = [
        sys.executable,
        str(root / "src" / "training" / "train_autoencoder_audio_local.py"),
        "--preset",
        "custom",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--latent-dim",
        "64",
        "--clip-seconds",
        "0.5",
        "--data-dir",
        str(root / "data" / "AudioData" / "ESC-50-master" / "audio"),
        "--real-file-limit",
        "64",
        "--no-export-tflite",
        "--output-root",
        str(output_dir),
    ] + extra_args
    subprocess.check_call(cmd)

    # Locate report in latest run subdirectory
    reports = list(output_dir.glob("*/audio_evaluation_report.md"))
    if not reports:
        raise FileNotFoundError(f"No audio_evaluation_report.md found under {output_dir}")
    report_file = sorted(reports)[-1]
    data = read_markdown_json_report(report_file)
    if not data:
        raise ValueError(f"Failed to read report JSON from {report_file}")
    return data


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Baseline unquantized run
        print("[*] Running audio smoke test (baseline float32)...")
        base_dir = tmp_path / "baseline"
        base_report = run_audio_smoke([], base_dir)

        assert "estimated_input_to_latent_ratio" in base_report, "Missing estimated_input_to_latent_ratio"
        assert "actual_latent_bytes_per_clip" in base_report, "Missing actual_latent_bytes_per_clip"
        assert "actual_compression_ratio" in base_report, "Missing actual_compression_ratio"
        assert "compression_estimate" in base_report, "Missing compression_estimate"
        assert base_report["compression_estimate"]["actual"]["quantized"] is False, "Expected unquantized baseline"
        print(
            f"[+] Baseline passed: estimated_ratio={base_report['estimated_input_to_latent_ratio']:.2f}x, "
            f"actual_bytes={base_report['actual_latent_bytes_per_clip']:.1f} bytes/clip"
        )

        # 2. Milestone A1 quantized run (8-bit quantization + noise annealing)
        print("[*] Running audio smoke test (Milestone A1 8-bit quantized + STE)...")
        quant_dir = tmp_path / "quantized"
        quant_report = run_audio_smoke(["--latent-bit-depth", "8", "--quant-noise-anneal"], quant_dir)

        assert "actual_latent_bytes_per_clip" in quant_report, "Missing actual_latent_bytes_per_clip in quant run"
        assert "actual_compression_ratio" in quant_report, "Missing actual_compression_ratio in quant run"
        actual_info = quant_report["compression_estimate"]["actual"]
        assert actual_info["quantized"] is True, "Expected quantized latent in quant run"
        assert actual_info["latent_bit_depth"] == 8, f"Expected latent_bit_depth=8, got {actual_info['latent_bit_depth']}"
        assert actual_info["actual_latent_bytes_per_clip"] > 0, "Invalid actual_latent_bytes_per_clip"
        assert actual_info["actual_compression_ratio"] > 0, "Invalid actual_compression_ratio"
        print(
            f"[+] Quantized passed: actual_bytes={actual_info['actual_latent_bytes_per_clip']:.1f} bytes/clip, "
            f"actual_ratio={actual_info['actual_compression_ratio']:.2f}x, "
            f"entropy={actual_info['entropy_bits_per_dim']:.2f} bits/dim"
        )

        # 3. Milestone A3 perceptual loss run (perceptual profile + psychoacoustic evaluation)
        print("[*] Running audio smoke test (Milestone A3 perceptual profile + evaluation)...")
        perceptual_dir = tmp_path / "perceptual"
        perceptual_report = run_audio_smoke(["--loss-profile", "perceptual", "--eval-perceptual-metrics"], perceptual_dir)

        assert perceptual_report["config"]["loss_profile"] == "perceptual", "Expected loss_profile=perceptual"
        assert perceptual_report["config"]["loss_mel_weight"] > 0, "Expected positive loss_mel_weight in perceptual profile"
        assert perceptual_report["config"]["loss_a_weight"] > 0, "Expected positive loss_a_weight in perceptual profile"
        assert "perceptual_metrics" in perceptual_report, "Missing perceptual_metrics in evaluation report"
        p_metrics = perceptual_report["perceptual_metrics"]
        assert "lsd_db" in p_metrics and p_metrics["lsd_db"] >= 0, "Invalid LSD metric"
        assert "a_weighted_lsd_db" in p_metrics and p_metrics["a_weighted_lsd_db"] >= 0, "Invalid A-LSD metric"
        assert "envelope_correlation_stoi_proxy" in p_metrics and 0.0 <= p_metrics["envelope_correlation_stoi_proxy"] <= 1.0, "Invalid STOI proxy metric"
        assert p_metrics["sample_count"] > 0, "Zero samples evaluated for perceptual metrics"
        print(
            f"[+] Perceptual passed: LSD={p_metrics['lsd_db']:.2f} dB, "
            f"A-LSD={p_metrics['a_weighted_lsd_db']:.2f} dB, "
            f"STOI proxy={p_metrics['envelope_correlation_stoi_proxy']:.3f}"
        )

    print("[+] All audio smoke tests passed successfully (Baseline, A1 Quantization, & A3 Perceptual Loss)")


if __name__ == "__main__":
    main()
