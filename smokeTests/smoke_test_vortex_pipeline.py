#!/usr/bin/env python3
"""Smoke test for realtime audio vortex pipeline."""

import numpy as np

from realtime_audio_vortex_pipeline import RealtimeAudioVortexPipeline


def main():
    sr = 48000
    n = 2048
    t = np.arange(n, dtype=np.float32) / sr
    tone = 0.7 * np.sin(2.0 * np.pi * 220.0 * t)

    pipeline = RealtimeAudioVortexPipeline(sample_rate=sr, window_size=4096, smoothing=0.8)
    pipeline.push_samples(tone)
    pipeline.push_samples(tone)

    state = pipeline.get_vortex_state()

    assert 0.1 <= state.spin <= 6.0
    assert 0.05 <= state.turbulence <= 8.0
    assert 0.15 <= state.radius <= 4.0
    assert 0.1 <= state.inward_pull <= 4.0
    assert 0.0 <= state.color_shift <= 360.0

    print("[+] Vortex pipeline smoke test passed")


if __name__ == "__main__":
    main()

