#!/usr/bin/env python3
"""Realtime waveform -> frequency features -> fluid vortex control mapping.

This module is designed to be called from an audio playback callback.
Feed PCM chunks using `RealtimeAudioVortexPipeline.push_samples(...)`, then call
`RealtimeAudioVortexPipeline.get_vortex_state()` each render tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FrequencyFeatures:
    rms: float
    peak: float
    centroid_hz: float
    bass_energy: float
    mid_energy: float
    treble_energy: float
    spectral_flux: float


@dataclass
class VortexState:
    spin: float
    turbulence: float
    radius: float
    inward_pull: float
    color_shift: float


class RealtimeAudioVortexPipeline:
    """Maintains a rolling waveform window and exposes smoothed vortex controls."""

    def __init__(self, sample_rate: int = 48000, window_size: int = 4096, smoothing: float = 0.82):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0.0, 1.0)")

        self.sample_rate = int(sample_rate)
        self.window_size = int(window_size)
        self.smoothing = float(smoothing)

        self._buffer = np.zeros(self.window_size, dtype=np.float32)
        self._write_idx = 0
        self._filled = 0

        self._prev_spectrum = np.zeros(self.window_size // 2 + 1, dtype=np.float32)
        self._smoothed = {
            "rms": 0.0,
            "peak": 0.0,
            "centroid_hz": 0.0,
            "bass_energy": 0.0,
            "mid_energy": 0.0,
            "treble_energy": 0.0,
            "spectral_flux": 0.0,
        }

    def push_samples(self, samples: np.ndarray) -> None:
        """Push PCM samples from the playback stream.

        Accepts mono shape (n,) or stereo shape (n, 2). Values should be float
        in [-1, 1], but int PCM is also accepted and normalized.
        """
        x = np.asarray(samples)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32, copy=False)

        # Normalize integer PCM input to [-1, 1].
        if x.size and np.max(np.abs(x)) > 1.5:
            x = x / 32768.0

        x = np.clip(x, -1.0, 1.0)
        for value in x:
            self._buffer[self._write_idx] = value
            self._write_idx = (self._write_idx + 1) % self.window_size
            self._filled = min(self._filled + 1, self.window_size)

    def _get_window(self) -> np.ndarray:
        if self._filled < self.window_size:
            out = np.zeros(self.window_size, dtype=np.float32)
            out[-self._filled :] = self._buffer[: self._filled]
            return out
        return np.concatenate((self._buffer[self._write_idx :], self._buffer[: self._write_idx]))

    def _band_energy(self, freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(np.mean(mag[mask]))

    def compute_features(self) -> FrequencyFeatures:
        window = self._get_window()
        hann = np.hanning(len(window)).astype(np.float32)
        win = window * hann

        spectrum = np.abs(np.fft.rfft(win)).astype(np.float32)
        freqs = np.fft.rfftfreq(len(win), d=1.0 / self.sample_rate)

        mag_sum = float(np.sum(spectrum) + 1e-8)
        centroid = float(np.sum(freqs * spectrum) / mag_sum)

        bass = self._band_energy(freqs, spectrum, 20.0, 250.0)
        mid = self._band_energy(freqs, spectrum, 250.0, 2000.0)
        treble = self._band_energy(freqs, spectrum, 2000.0, 12000.0)

        flux = float(np.mean(np.maximum(spectrum - self._prev_spectrum, 0.0)))
        self._prev_spectrum = spectrum

        rms = float(np.sqrt(np.mean(window * window) + 1e-8))
        peak = float(np.max(np.abs(window)) if window.size else 0.0)

        raw = {
            "rms": rms,
            "peak": peak,
            "centroid_hz": centroid,
            "bass_energy": bass,
            "mid_energy": mid,
            "treble_energy": treble,
            "spectral_flux": flux,
        }

        for key, value in raw.items():
            self._smoothed[key] = self.smoothing * self._smoothed[key] + (1.0 - self.smoothing) * value

        return FrequencyFeatures(**self._smoothed)

    def get_vortex_state(self) -> VortexState:
        f = self.compute_features()

        bass_mid = f.bass_energy + f.mid_energy + 1e-6
        bass_ratio = f.bass_energy / bass_mid
        treble_ratio = f.treble_energy / (f.mid_energy + f.treble_energy + 1e-6)

        spin = float(np.clip(0.3 + 3.8 * bass_ratio + 1.2 * f.rms, 0.1, 6.0))
        turbulence = float(np.clip(0.1 + 6.0 * treble_ratio + 2.0 * f.spectral_flux, 0.05, 8.0))
        radius = float(np.clip(0.2 + 2.8 * f.rms + 0.2 * bass_ratio, 0.15, 4.0))
        inward_pull = float(np.clip(0.2 + 2.5 * bass_ratio + 0.2 * f.peak, 0.1, 4.0))

        color_shift = float(np.clip((f.centroid_hz / 12000.0) * 360.0, 0.0, 360.0))

        return VortexState(
            spin=spin,
            turbulence=turbulence,
            radius=radius,
            inward_pull=inward_pull,
            color_shift=color_shift,
        )

    def get_feature_dict(self) -> Dict[str, float]:
        f = self.compute_features()
        return {
            "rms": f.rms,
            "peak": f.peak,
            "centroid_hz": f.centroid_hz,
            "bass_energy": f.bass_energy,
            "mid_energy": f.mid_energy,
            "treble_energy": f.treble_energy,
            "spectral_flux": f.spectral_flux,
        }

