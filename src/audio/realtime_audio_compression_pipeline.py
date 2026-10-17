#!/usr/bin/env python3
"""
Real-time chunked/overlap-add streaming audio compression pipeline (Milestone A2).

Provides an actual streaming inference engine executing the trained neural
autoencoder over continuous PCM audio streams with bounded latency and click-free
overlap-add reconstruction.

Distinction from vortex demo:
- `src/audio/realtime_audio_vortex_pipeline.py` is a visualization-only feature
  extractor (FFT energies -> fluid vortex controls).
- `src/audio/realtime_audio_compression_pipeline.py` (this module) actually runs
  the neural autoencoder encoder/decoder over streaming PCM audio.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Ensure training module is discoverable for model building and custom layers
training_dir = Path(__file__).resolve().parent.parent / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

try:
    import tensorflow as tf
    from train_autoencoder_audio_local import (
        StraightThroughQuantize,
        build_audio_autoencoder,
    )
except ImportError:
    tf = None
    StraightThroughQuantize = None
    build_audio_autoencoder = None


@dataclass
class StreamMetrics:
    total_input_samples: int
    total_output_samples: int
    chunks_processed: int
    total_process_time_s: float
    avg_chunk_time_ms: float
    real_time_factor: float  # process_time / audio_duration (< 1.0 is faster than real time)
    algorithmic_latency_ms: float
    buffered_input_ms: float
    buffered_output_ms: float


class RealtimeAudioCompressionPipeline:
    """
    Streaming inference pipeline for neural audio autoencoders.

    Buffers incoming PCM samples of arbitrary sizes, slices them into `clip_len`
    chunks with `hop_size` advance, executes the neural autoencoder, and uses
    normalized overlap-add (OLA) windowing to eliminate chunk boundary clicks.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[str | Path] = None,
        sample_rate: int = 16000,
        clip_len: int = 8000,
        overlap_ratio: float = 0.5,
        latent_dim: int = 128,
        latent_bit_depth: int = 0,
        model_base_filters: int = 48,
        model_kernel_size: int = 5,
    ):
        self.sample_rate = int(sample_rate)
        self.clip_len = int(clip_len)
        if self.clip_len % 16 != 0:
            raise ValueError(f"clip_len ({self.clip_len}) must be divisible by 16")

        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError(f"overlap_ratio ({overlap_ratio}) must be in [0.0, 1.0)")
        self.overlap_ratio = float(overlap_ratio)

        # Hop size is the distance the window advances between inferences
        if self.overlap_ratio > 0.0:
            self.hop_size = max(1, int(round(self.clip_len * (1.0 - self.overlap_ratio))))
        else:
            self.hop_size = self.clip_len

        # Algorithmic delay in ms (buffer length to produce first chunk)
        self.algorithmic_latency_ms = (self.clip_len / float(self.sample_rate)) * 1000.0

        # Analysis/synthesis Hann window normalized for perfect overlap-add
        self.window = np.hanning(self.clip_len).astype(np.float32)

        # Model resolution
        self.latent_dim = int(latent_dim)
        self.latent_bit_depth = int(latent_bit_depth)
        self.model_base_filters = int(model_base_filters)
        self.model_kernel_size = int(model_kernel_size)

        if model is not None:
            self.model = model
        elif model_path is not None and str(model_path).strip():
            self.model = self._load_model_from_file(model_path)
        else:
            self.model = self._build_default_model()

        # Buffers
        self._input_buffer = np.zeros(0, dtype=np.float32)
        # Overlap-add buffers: audio accumulator and window normalization accumulator
        self._ola_audio = np.zeros(0, dtype=np.float32)
        self._ola_weights = np.zeros(0, dtype=np.float32)
        self._output_queue = np.zeros(0, dtype=np.float32)

        # Telemetry
        self.total_input_samples = 0
        self.total_output_samples = 0
        self.chunks_processed = 0
        self.total_process_time_s = 0.0

    def _load_model_from_file(self, model_path: str | Path) -> Any:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio model not found at {path}")

        custom_objects = {}
        if StraightThroughQuantize is not None:
            custom_objects["StraightThroughQuantize"] = StraightThroughQuantize

        if path.suffix == ".keras" or path.is_dir():
            return tf.keras.models.load_model(str(path), custom_objects=custom_objects)
        elif path.suffix in {".h5", ".weights"}:
            model = self._build_default_model()
            model.load_weights(str(path))
            return model
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")

    def _build_default_model(self) -> Any:
        if build_audio_autoencoder is None:
            raise RuntimeError("TensorFlow or train_autoencoder_audio_local is not available")
        return build_audio_autoencoder(
            clip_len=self.clip_len,
            latent_dim=self.latent_dim,
            latent_bit_depth=self.latent_bit_depth,
            model_base_filters=self.model_base_filters,
            model_kernel_size=self.model_kernel_size,
        )

    def push_samples(self, samples: np.ndarray) -> int:
        """
        Push streaming PCM samples into the pipeline.

        Accepts 1D mono or 2D (samples, channels) audio.
        Handles int16 or float32 representations in [-1.0, 1.0].
        Returns the number of audio chunks processed during this push.
        """
        arr = np.asarray(samples)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # downmix to mono
        arr = arr.astype(np.float32, copy=False)

        # Integer PCM to float32 normalization
        if arr.size and np.max(np.abs(arr)) > 1.5:
            arr = arr / 32768.0

        arr = np.clip(arr, -1.0, 1.0)
        self.total_input_samples += len(arr)
        self._input_buffer = np.concatenate((self._input_buffer, arr))

        return self._process_available_chunks()

    def _process_available_chunks(self) -> int:
        """Processes all complete `clip_len` windows available in `_input_buffer`."""
        chunks_this_round = 0

        while len(self._input_buffer) >= self.clip_len:
            # 1. Slice current window
            window_raw = self._input_buffer[: self.clip_len]
            # Advance input buffer by hop_size
            self._input_buffer = self._input_buffer[self.hop_size :]

            # 2. Execute Neural Autoencoder inference
            t0 = time.perf_counter()
            model_in = window_raw.reshape(1, self.clip_len, 1)
            pred = self.model(model_in, training=False)
            recon = pred.numpy().reshape(self.clip_len)
            proc_time = time.perf_counter() - t0

            self.total_process_time_s += proc_time
            self.chunks_processed += 1
            chunks_this_round += 1

            # 3. Overlap-add accumulation with window weighting
            windowed_recon = recon * self.window
            window_weights = self.window

            # Extend OLA buffers if needed to accommodate new window
            needed_len = max(len(self._ola_audio), self.clip_len)
            if len(self._ola_audio) < needed_len:
                pad = needed_len - len(self._ola_audio)
                self._ola_audio = np.pad(self._ola_audio, (0, pad), mode="constant")
                self._ola_weights = np.pad(self._ola_weights, (0, pad), mode="constant")

            self._ola_audio[: self.clip_len] += windowed_recon
            self._ola_weights[: self.clip_len] += window_weights

            # 4. Extract ready samples (hop_size samples where reconstruction is complete)
            if len(self._ola_audio) >= self.hop_size:
                ready_samples = self._ola_audio[: self.hop_size]
                ready_weights = self._ola_weights[: self.hop_size]

                # Normalize by accumulated window weights to prevent amplitude distortion
                safe_weights = np.where(ready_weights > 1e-6, ready_weights, 1.0)
                normalized = np.clip(ready_samples / safe_weights, -1.0, 1.0)

                self._output_queue = np.concatenate((self._output_queue, normalized))
                self.total_output_samples += len(normalized)

                # Shift OLA buffers left by hop_size
                self._ola_audio = self._ola_audio[self.hop_size :]
                self._ola_weights = self._ola_weights[self.hop_size :]

        return chunks_this_round

    def pop_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        Pop reconstructed audio samples from the output queue.

        If `num_samples` is None, returns all available reconstructed samples.
        """
        if len(self._output_queue) == 0:
            return np.zeros(0, dtype=np.float32)

        if num_samples is None or num_samples >= len(self._output_queue):
            out = self._output_queue
            self._output_queue = np.zeros(0, dtype=np.float32)
            return out

        n = int(num_samples)
        out = self._output_queue[:n]
        self._output_queue = self._output_queue[n:]
        return out

    def flush(self) -> np.ndarray:
        """Flushes remaining samples in the OLA accumulator to the output queue."""
        if len(self._ola_audio) > 0:
            safe_weights = np.where(self._ola_weights > 1e-6, self._ola_weights, 1.0)
            tail = np.clip(self._ola_audio / safe_weights, -1.0, 1.0)
            self._output_queue = np.concatenate((self._output_queue, tail))
            self.total_output_samples += len(tail)
            self._ola_audio = np.zeros(0, dtype=np.float32)
            self._ola_weights = np.zeros(0, dtype=np.float32)
        return self.pop_samples()

    def get_metrics(self) -> StreamMetrics:
        """Calculates live throughput, real-time factor, and latency metrics."""
        avg_ms = (
            (self.total_process_time_s / max(1, self.chunks_processed)) * 1000.0
            if self.chunks_processed > 0
            else 0.0
        )
        total_audio_duration_s = (
            (self.chunks_processed * self.hop_size) / float(self.sample_rate)
            if self.chunks_processed > 0
            else 0.0
        )
        rtf = (
            self.total_process_time_s / max(1e-6, total_audio_duration_s)
            if total_audio_duration_s > 0
            else 0.0
        )
        buffered_in_ms = (len(self._input_buffer) / float(self.sample_rate)) * 1000.0
        buffered_out_ms = (len(self._output_queue) / float(self.sample_rate)) * 1000.0

        return StreamMetrics(
            total_input_samples=self.total_input_samples,
            total_output_samples=self.total_output_samples,
            chunks_processed=self.chunks_processed,
            total_process_time_s=self.total_process_time_s,
            avg_chunk_time_ms=avg_ms,
            real_time_factor=rtf,
            algorithmic_latency_ms=self.algorithmic_latency_ms,
            buffered_input_ms=buffered_in_ms,
            buffered_output_ms=buffered_out_ms,
        )

    def reset(self) -> None:
        """Resets all buffers and metric counters."""
        self._input_buffer = np.zeros(0, dtype=np.float32)
        self._ola_audio = np.zeros(0, dtype=np.float32)
        self._ola_weights = np.zeros(0, dtype=np.float32)
        self._output_queue = np.zeros(0, dtype=np.float32)
        self.total_input_samples = 0
        self.total_output_samples = 0
        self.chunks_processed = 0
        self.total_process_time_s = 0.0


def main():
    parser = argparse.ArgumentParser(description="Real-time streaming audio compression pipeline.")
    parser.add_argument("--model-path", type=str, default="", help="Path to trained .keras or .weights.h5 audio model")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument("--clip-seconds", type=float, default=0.5, help="Autoencoder chunk duration (seconds)")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Overlap ratio between successive chunks (e.g. 0.5)")
    parser.add_argument("--stream-seconds", type=float, default=3.0, help="Duration of simulated synthetic stream")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Simulated streaming callback buffer size")
    args = parser.parse_args()

    clip_len = int(args.sample_rate * args.clip_seconds)
    # Align to 16-sample downsampling boundary
    clip_len = (clip_len // 16) * 16

    print(f"[*] Initializing RealtimeAudioCompressionPipeline...")
    print(f"    Sample rate: {args.sample_rate} Hz")
    print(f"    Chunk length: {clip_len} samples ({clip_len / args.sample_rate:.3f}s)")
    print(f"    Overlap ratio: {args.overlap_ratio:.1%}")

    pipeline = RealtimeAudioCompressionPipeline(
        model_path=args.model_path if args.model_path else None,
        sample_rate=args.sample_rate,
        clip_len=clip_len,
        overlap_ratio=args.overlap_ratio,
        latent_dim=64,
        model_base_filters=32,
    )

    print(f"[*] Simulating live audio stream ({args.stream_seconds}s in blocks of {args.chunk_size} samples)...")
    total_samples = int(args.sample_rate * args.stream_seconds)
    t = np.arange(total_samples, dtype=np.float32) / args.sample_rate
    # Multi-tone test signal
    signal = 0.6 * np.sin(2.0 * np.pi * 300.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1200.0 * t)

    idx = 0
    while idx < total_samples:
        end = min(idx + args.chunk_size, total_samples)
        block = signal[idx:end]
        pipeline.push_samples(block)
        ready = pipeline.pop_samples()
        metrics = pipeline.get_metrics()
        idx = end

    # Flush final tails
    tail = pipeline.flush()
    metrics = pipeline.get_metrics()

    print(f"\n[+] Streaming simulation complete:")
    print(f"    Input samples:  {metrics.total_input_samples}")
    print(f"    Output samples: {metrics.total_output_samples}")
    print(f"    Chunks computed: {metrics.chunks_processed}")
    print(f"    Algorithmic latency: {metrics.algorithmic_latency_ms:.1f} ms")
    print(f"    Average inference time: {metrics.avg_chunk_time_ms:.2f} ms/chunk")
    print(f"    Real-Time Factor (RTF): {metrics.real_time_factor:.4f} (RTF < 1.0 is real-time viable)")


if __name__ == "__main__":
    main()
