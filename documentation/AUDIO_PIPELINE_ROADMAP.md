# Audio Compression Pipeline — Development Roadmap

Status: Draft v1 · Owner: audio modality · Companion to `AGENT.md` / `CONTRIBUTING.md`

This roadmap targets `train_autoencoder_audio_local.py`, `prepare_audio_dataset.py`,
`realtime_audio_vortex_pipeline.py` / `vortex_realtime_demo.py`, and the audio paths
in `random_search_hyperparams.py` / `param_overrides.py`.

## 1. Current state (as of this roadmap)

| Area | Today |
|---|---|
| Architecture | Conv1D encoder (4 stages, stride-2, 16× downsample) → linear Dense bottleneck (no ReLU, preserves signed/phase-continuous values) → Conv1DTranspose decoder (16× upsample, no skip connections) → tanh output. |
| Data pipeline | `prepare_audio_dataset.py` canonicalizes mp3/flac/ogg/m4a/aac/opus/wav → mono PCM16 WAV at a target sample rate with `none/peak/rms/rms_peak` normalization + JSON sidecar metadata. Trainer loads canonical WAVs, mono-downmixes, resamples via linear interpolation, pads/crops to a fixed `clip_len = sample_rate * clip_seconds`. |
| Compression accounting | **Theoretical only**: `estimate_audio_compression_ratio()` compares `clip_len*16 bits` vs `latent_dim*latent_bits` — the latent is **never actually quantized or entropy-coded**; it's dense float32 throughout training and export. |
| Loss/metrics | Weighted MSE + L1 + STFT-magnitude loss with a "hard residual" upweighting term; tracks SNR (dB, primary) and PSNR. `TargetSNRCallback` stops early at `--target-snr-db` (default 90). |
| Real-time path | `realtime_audio_vortex_pipeline.py` is a **feature-extraction-only** pipeline (FFT band energies → visual "vortex" control signals for a demo). It does **not** run the trained autoencoder — there is no real-time/streaming inference path for the compression model itself. |
| Robustness | Mono-only (stereo is downmixed), fixed training sample rate (16 kHz) vs. fixed vortex demo rate (48 kHz), no noise augmentation, silent fallback-to-silence on WAV read failure. |
| Search & docs | `random_search_hyperparams.py` samples latent_dim/filters/kernel/batch/epochs/lr/clip_seconds for audio; ranks by `snr_db_metric`. `--params-file` JSON `audio` section documented in `documentation/MODEL_PARAMS_INPUT.md`. |

## 2. Constraints (do not violate)

1. **Laptop-only (M1) compute** — every milestone needs a tiny/smoke-testable path.
2. **Frozen preset defaults** (`m1-air-fast/balanced/quality`) tied to `README.md` benchmark numbers — changes must be called out explicitly in their own PR.
3. **CI stays lightweight** — no training/data-dependent jobs in CI; new functionality must be smoke-testable via `smokeTests/smoke_test_audio_local.py` / `smoke_test_prepare_audio_dataset.py` / `smoke_test_vortex_pipeline.py` locally.
4. **No package refactor** — stay within the flat CLI-script structure.
5. **Backward-compatible CLI** — new flags default to reproducing current behavior; breaking renames require a `!`/`BREAKING CHANGE` commit and a `run_full_production_pipeline.py` sync.

## 3. Goals

- **G1 — Real compression accounting**: measure actual bits/sample, not a theoretical ratio.
- **G2 — Close the QAT gap**: train the model to tolerate the quantization it will actually be deployed with.
- **G3 — Real-time compression inference**: extend (or add alongside) the vortex pipeline a streaming path that actually runs the trained autoencoder in chunks.
- **G4 — Perceptual quality**: move beyond waveform/STFT-magnitude loss toward psychoacoustically-informed objectives.
- **G5 — Robustness & multi-channel/rate support**: stereo, variable sample rate, noisy real-world input.
- **G6 — Discoverability**: docs, search space, and reports stay in sync as the above land.

## 4. Milestones

### Milestone A1 — Real Compression Accounting & Quantization
**Objective:** Replace the theoretical bits/latent-dim estimate with a real measured bitrate, and make latent quantization actually happen (not just be assumed for the ratio calc).
**Why:** `--latent-bits` currently only feeds an arithmetic estimate; the latent is float32 end-to-end, so real deployed file size is unknown and likely far larger than advertised.
**Exit criteria:** Evaluation report includes an `actual_latent_bytes_per_clip` computed from a real quantization/entropy step, gated behind an opt-in flag that defaults to preserving current behavior.

Issues:
1. **[audio/compression] Add real bit-depth quantization of the latent vector** — `--latent-bit-depth` (default disabled/float32) that actually rounds/clips the bottleneck output to N bits before decode.
2. **[audio/compression] Add straight-through estimator for quantized latent gradients** — so the model can be trained with the quantization in the loop, not just evaluated post-hoc.
3. **[audio/compression] Add real vs theoretical compression metrics to evaluation report** — log both `estimated_input_to_latent_ratio` (existing) and `actual_latent_bytes_per_clip` / `actual_compression_ratio` side by side.
4. **[audio/docs] Document real vs theoretical audio compression accounting** — update `README.md` / `documentation/MODEL_PARAMS_INPUT.md`.

### Milestone A2 — Real-Time Streaming Compression Path
**Objective:** Give the trained autoencoder an actual streaming/chunked inference path, distinct from the existing vortex feature-extraction demo.
**Why:** `realtime_audio_vortex_pipeline.py` never invokes the trained model — there is currently no way to run the compressor in real time at all.
**Exit criteria:** A new streaming inference module can take arbitrary-length PCM input, process it in `clip_len`-sized (or overlap-add) chunks through a loaded model, and reconstruct output with bounded latency, with a smoke test proving no crash on a short synthetic stream.

Issues:
1. **[audio/streaming] Design a chunked/overlap-add inference wrapper for the trained autoencoder** — reuse existing fixed `clip_len` constraint; document latency implications.
2. **[audio/streaming] Add `realtime_audio_compression_pipeline.py`** — mirrors `realtime_audio_vortex_pipeline.py`'s streaming buffer conventions but decodes/encodes through the trained model instead of extracting FFT features.
3. **[audio/streaming] Add a smoke test for streaming inference** — `smokeTests/smoke_test_realtime_audio_compression.py`, tiny model + short synthetic stream, asserts output shape/latency bounds.
4. **[audio/docs] Document the real-time compression path and its relationship to the vortex demo** — clarify in README that the vortex pipeline is visualization-only.

### Milestone A3 — Perceptual Quality & Psychoacoustic Loss
**Objective:** Move the loss function beyond MSE/L1/STFT-magnitude toward perceptually-weighted objectives.
**Why:** Current loss treats all frequencies/samples equally (aside from the "hard residual" weighting); no mel-scale, A-weighting, or loudness-aware terms exist.
**Exit criteria:** New optional loss terms are available via `--loss-profile` (mirroring the image trainer's pattern), default profile reproduces current benchmark behavior exactly.

Issues:
1. **[audio/loss] Add a mel-scale/log-mel spectral loss option** — `--loss-mel-weight`, computed via `tf.signal.linear_to_mel_weight_matrix`.
2. **[audio/loss] Add an A-weighting (or simplified perceptual weighting) term** — approximate psychoacoustic frequency sensitivity curve applied to the STFT magnitude loss.
3. **[audio/loss] Introduce `--loss-profile {balanced,snr,perceptual}` presets** — mirrors the image trainer's `--loss-profile` convention; `balanced` matches current defaults exactly.
4. **[audio/eval] Add optional PESQ/STOI-style perceptual metric to evaluation report** — behind an opt-in flag if a suitable pure-Python/TF implementation is available; otherwise document as a future dependency decision.

### Milestone A4 — Data Pipeline Robustness (Multi-channel, Rate, Noise)
**Objective:** Remove the mono-only/fixed-sample-rate/no-augmentation assumptions.
**Why:** Real-world audio is often stereo, comes at varied sample rates, and training never sees additive noise — robustness is untested.
**Exit criteria:** Trainer supports an opt-in stereo mode (or documents the mono-only constraint explicitly as intentional), resampling method is configurable, and a noise-robustness smoke check exists.

Issues:
1. **[audio/data] Make stereo→mono downmixing explicit and opt-in vs. add a `--channels {mono,stereo}` mode** — decide and document the intended scope; if stereo is out of scope, add a clear `NotImplementedError`/docstring instead of silent averaging.
2. **[audio/data] Replace linear-interpolation resampling with a configurable method** — `--resample-method {linear,polyphase}`, default `linear` to preserve current behavior.
3. **[audio/data] Add optional additive noise augmentation during training** — `--augment-noise-std`, default 0 (off).
4. **[audio/data] Add a corrupted/noisy-input robustness smoke check** — feed a clipped/corrupted WAV through inference and assert graceful handling (no crash, bounded SNR drop) instead of silent fallback-to-silence.

### Milestone A5 — Reporting, Search & Docs Alignment
**Objective:** Keep `random_search_hyperparams.py`, `documentation/*.md`, and evaluation reports consistent with A1–A4 as they land.
**Why:** Every milestone above adds flags/metrics; without this milestone docs and search spaces drift (a known repo risk per `AGENT.md`).
**Exit criteria:** Random search can sweep any new axis (bit-depth, loss-profile, resample-method) behind explicit toggles; `documentation/MODEL_PARAMS_INPUT.md` and `README.md` enumerate every new flag.

Issues:
1. **[audio/search] Extend audio search space for new flags** — bit-depth, loss-profile, resample-method added to `random_search_hyperparams.py` and `documentation/random_search_profile.json`.
2. **[audio/docs] Full CLI flag audit for the audio trainer** — cross-check `--help` output vs `documentation/MODEL_PARAMS_INPUT.md`.
3. **[audio/reporting] Add a compression-ratio/SNR frontier plot** — extend `plot_training_metrics_from_report.py` / `generate_master_report.py`.
4. **[audio/pipeline] Sync `run_full_production_pipeline.py` with any renamed/added audio flags.**

## 5. Suggested sequencing

A1 (real accounting) should land before A3 (perceptual loss) so quality tradeoffs are measured against real bitrate, not the theoretical estimate. A2 (streaming) and A4 (robustness) are independent and can run in parallel. A5 is continuous/trailing.

## 6. Out of scope (for this roadmap)

- Image/video pipelines (tracked separately).
- Cloud/multi-GPU training support.
- Changing existing preset defaults without an explicit, separately-reviewed PR.
- Making the vortex visualization demo itself more sophisticated (it's out of scope for the *compression* roadmap).

