#!/usr/bin/env python3
"""Smoke test for real-time streaming audio compression pipeline (Milestone A2)."""

import sys
from pathlib import Path

import numpy as np

# Add src/audio to sys.path
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / "src" / "audio"))
from realtime_audio_compression_pipeline import RealtimeAudioCompressionPipeline


def main():
    sample_rate = 16000
    clip_len = 4000  # 0.25s
    overlap_ratio = 0.5
    stream_seconds = 2.0
    total_samples = int(sample_rate * stream_seconds)

    print(f"[*] Initializing pipeline for smoke test...")
    pipeline = RealtimeAudioCompressionPipeline(
        sample_rate=sample_rate,
        clip_len=clip_len,
        overlap_ratio=overlap_ratio,
        latent_dim=32,
        model_base_filters=16,
    )

    # Generate synthetic multi-tone signal
    t = np.arange(total_samples, dtype=np.float32) / sample_rate
    test_stream = 0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)

    # Stream through pipeline with variable block sizes (simulating varying callback sizes)
    chunk_sizes = [512, 1024, 768, 256, 1024]
    idx = 0
    collected_output = []

    print(f"[*] Streaming {total_samples} samples through pipeline in varying chunk sizes...")
    chunk_idx = 0
    while idx < total_samples:
        block_size = chunk_sizes[chunk_idx % len(chunk_sizes)]
        end = min(idx + block_size, total_samples)
        block = test_stream[idx:end]
        pipeline.push_samples(block)
        ready = pipeline.pop_samples()
        if len(ready) > 0:
            collected_output.append(ready)
        idx = end
        chunk_idx += 1

    # Flush final remaining samples
    tail = pipeline.flush()
    if len(tail) > 0:
        collected_output.append(tail)

    full_output = np.concatenate(collected_output)
    metrics = pipeline.get_metrics()

    # Validations & assertions
    assert len(full_output) == total_samples, (
        f"Output length mismatch: expected {total_samples}, got {len(full_output)}"
    )
    assert not np.isnan(full_output).any(), "Found NaNs in reconstructed output"
    assert not np.isinf(full_output).any(), "Found Infs in reconstructed output"
    assert np.max(np.abs(full_output)) <= 1.05, f"Output exceeded amplitude bounds: {np.max(np.abs(full_output))}"
    assert metrics.chunks_processed > 0, "No chunks were processed"
    assert metrics.algorithmic_latency_ms == (clip_len / sample_rate) * 1000.0, "Latency mismatch"

    print(f"[+] Reconstructed {len(full_output)} samples matching input stream length")
    print(f"[+] Algorithmic latency: {metrics.algorithmic_latency_ms:.1f} ms")
    print(f"[+] Average processing time: {metrics.avg_chunk_time_ms:.2f} ms/chunk")
    print(f"[+] Real-Time Factor (RTF): {metrics.real_time_factor:.4f}")
    print("[+] Real-time audio compression pipeline smoke test passed successfully")


if __name__ == "__main__":
    main()
