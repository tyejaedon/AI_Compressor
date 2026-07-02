#!/usr/bin/env python3
"""Tiny realtime demo for the audio->vortex pipeline.

Simulates audio playback chunks and prints vortex controls at render cadence.
"""

import time

import numpy as np

from realtime_audio_vortex_pipeline import RealtimeAudioVortexPipeline


def synth_chunk(sample_rate, start_sample, chunk_size):
    t = (np.arange(chunk_size, dtype=np.float32) + start_sample) / float(sample_rate)
    low = 0.65 * np.sin(2.0 * np.pi * 90.0 * t)
    mid = 0.35 * np.sin(2.0 * np.pi * 640.0 * t)
    high = 0.20 * np.sin(2.0 * np.pi * 3200.0 * t)
    beat = 0.7 + 0.3 * np.sin(2.0 * np.pi * 1.2 * t)
    return np.clip((low + mid + high) * beat, -1.0, 1.0).astype(np.float32)


def main():
    sample_rate = 48000
    chunk_size = 1024
    seconds = 5.0

    pipeline = RealtimeAudioVortexPipeline(sample_rate=sample_rate, window_size=4096, smoothing=0.82)

    total_chunks = int((seconds * sample_rate) // chunk_size)
    for i in range(total_chunks):
        chunk = synth_chunk(sample_rate, i * chunk_size, chunk_size)
        pipeline.push_samples(chunk)
        state = pipeline.get_vortex_state()

        print(
            f"spin={state.spin:.2f} turb={state.turbulence:.2f} "
            f"radius={state.radius:.2f} pull={state.inward_pull:.2f} hue={state.color_shift:.1f}"
        )

        # Simulate render frame pacing close to real-time chunk cadence.
        time.sleep(chunk_size / sample_rate)


if __name__ == "__main__":
    main()

