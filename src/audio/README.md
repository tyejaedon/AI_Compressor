# Audio Processing & Streaming Modules (`src/audio/`)

This directory contains audio dataset canonicalization, **real-time neural streaming compression**, and a separate **FFT-to-vortex visualization** demo.

## At a Glance

| File | Role | Uses Trained Model? | Key Outputs / Behavior |
|---|---|---|---|
| `prepare_audio_dataset.py` | Dataset Canonicalizer | No | Converts mixed-codec audio (`.mp3/.flac/.ogg/.wav`) -> canonical mono PCM16 `.wav` or `.npy` + `*.metadata.json` + `preparation_report.md` |
| `realtime_audio_compression_pipeline.py` | Real-Time Neural Codec | **Yes** | Chunked Hann overlap-add streaming inference (`RealtimeAudioCompressionPipeline`) with bounded latency & SNR/throughput metrics |
| `realtime_audio_vortex_pipeline.py` | Audio-to-Vortex Mapper | No (FFT only) | Extracts sub-bass/bass/mid/high energy + spectral centroid -> `VortexState` fluid simulation controls |
| `vortex_realtime_demo.py` | Vortex CLI Demo | No (FFT only) | Feeds synthesized audio chunks into `RealtimeAudioVortexPipeline` and prints render-tick states |

---

## 1. `prepare_audio_dataset.py`
Converts heterogeneous audio folders into uniform training-ready files:
- Decodes `.wav` natively or other formats via `ffmpeg` / `ffprobe`.
- Supports `--resample-method {linear,polyphase}` (bandlimited Kaiser-windowed sinc), `--stereo-downmix {average,left,right,error}`, `--remove-dc`, and peak/RMS normalization (`--target-rms 0.12`, `--target-peak 0.95`).
- Writes per-file `*.metadata.json` sidecars and a summary `preparation_report.md`.

```zsh
python src/audio/prepare_audio_dataset.py \
  --input-dir data/AudioData --output-dir data/AudioData_canonical \
  --target-sample-rate 16000 --output-format wav --normalization rms_peak --remove-dc
```

## 2. `realtime_audio_compression_pipeline.py`
Runs **actual neural autoencoder inference** over continuous PCM audio streams (Milestone A2):
- `RealtimeAudioCompressionPipeline` buffers incoming samples, slices fixed-size windows (`--clip-seconds`), runs `model.predict(...)`, and stitches output chunks via **Hann-windowed overlap-add** (`--overlap-ratio 0.5`) to eliminate boundary clicks.
- Tracks real-time factor (RTF), latency, throughput, and stream SNR via `get_metrics()`.

```zsh
python src/audio/realtime_audio_compression_pipeline.py \
  --model-path models/audio_local_run/<run>/best_audio_model.keras \
  --sample-rate 16000 --clip-seconds 0.5 --overlap-ratio 0.5 --stream-seconds 5.0
```

## 3. `realtime_audio_vortex_pipeline.py` & `vortex_realtime_demo.py`
**Visualization-only** modules (do **not** run neural autoencoder inference):
- `RealtimeAudioVortexPipeline.push_samples(...)` computes smoothed FFT band energies (`sub_bass`, `bass`, `mid`, `high`), RMS, peak, and spectral centroid, returning a `VortexState` (`angular_velocity`, `turbulence`, `radius_pulse`, `color_shift`) for visualizers.
- `vortex_realtime_demo.py` runs a self-contained synthetic audio loop demonstrating `RealtimeAudioVortexPipeline`.
