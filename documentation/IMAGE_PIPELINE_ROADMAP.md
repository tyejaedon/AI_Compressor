# Image Compression Pipeline — Development Roadmap

Status: Draft v1 · Owner: image modality · Companion to `AGENT.md` / `CONTRIBUTING.md`

This roadmap targets `train_autoencoder_image_local.py` (lossless/edge-preserving
trainer), `train_autoencoder_image_lossy_local.py` (rate-distortion trainer),
`upscale_reconstructed_images.py`, `generate_real_image_comparisons.py`,
`random_search_hyperparams.py` (image paths), and `param_overrides.py`.

## 1. Current state (as of this roadmap)

| Area                   | Today                                                                                                                                                                                                                                                                        |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lossless trainer       | Residual, pooling-free CNN (4 enc conv + 1×1 bottleneck + 4 dec conv), edge/SSIM/L1/MSE composite loss, dynamic-resolution inference, `--upscale-factor` + `--degrade-interp` simulate low-res input, `TargetPSNRCallback` + `TargetOutputJpegRatioCallback` for early stop. |
| Lossy trainer          | Strided conv encoder/decoder (16× spatial reduction), `StraightThroughQuantize` (naive round-to-8-bit) + `RatePenalty` (mean                                                                                                                                                 |latent| heuristic, not real bitrate), JPEG/WebP matched-PSNR baseline benchmark. |
| Upscaling              | `upscale_reconstructed_images.py` is **classic BICUBIC only** — no learned super-resolution model.                                                                                                                                                                           |
| Compression accounting | "Compression ratio" is measured indirectly via re-encoded JPEG/WebP byte size of the *reconstruction*, not the actual latent bitstream. No entropy coder exists for the latent tensor itself.                                                                                |
| Data pipeline          | Recursive file discovery, patch/center-crop extraction, CIFAR-10 fallback, explicit train/val/test dirs, `--real-only` no-leakage mode. Single-threaded Python-side decode + `tf.data.AUTOTUNE`, no TFRecord caching.                                                        |
| Search & docs          | `random_search_hyperparams.py` samples both trainers' hyperparameters; `--params-file` JSON convention is shared and documented in `documentation/MODEL_PARAMS_INPUT.md`.                                                                                                    |

## 2. Constraints (do not violate)

1. **Laptop-only (M1) compute.** No assumption of GPU cluster or multi-day training runs; every milestone must have a "tiny"/smoke-testable path.
2. **Frozen preset defaults.** `m1-air-fast/balanced/quality` presets and lossy defaults are tied to the benchmark numbers in `README.md` — any change to defaults must be called out explicitly in its PR, not silently bundled into a feature.
3. **CI stays lightweight.** No training/data-dependent jobs in CI (see `.github/workflows/ci.yml`); all new functionality must remain smoke-testable via `smokeTests/*.py` with tiny data/epoch counts, run locally by contributors.
4. **No package refactor.** Stay within the "flat CLI scripts" structure — do not turn this into an installable package as part of this roadmap.
5. **Backward-compatible CLI.** New flags must have safe defaults that reproduce current behavior; breaking flag renames require a `!`/`BREAKING CHANGE` commit and a `run_full_production_pipeline.py` sync (per `AGENT.md`).

## 3. Goals

- **G1 — Real compression accounting**: know actual latent bits/pixel, not a JPEG-size proxy.
- **G2 — Better rate-distortion tradeoff**: quantization the model is actually trained for (QAT), tunable/learned rate control.
- **G3 — Real upscaling, not just interpolation**: an optional learned super-resolution stage that composes with the existing degrade/upscale-factor training signal.
- **G4 — Robustness & throughput**: pipeline holds up on corrupted/varied real images and loads data faster on repeated runs.
- **G5 — Discoverability**: docs, reports, and random search stay in sync as the above land.

## 4. Milestones

### Milestone 1 — Real Rate-Distortion Accounting
**Objective:** Replace the heuristic `RatePenalty` magnitude proxy with an actual measurable bitrate, and report true bits-per-pixel / compression ratio in evaluation reports.
**Why:** Right now "compression ratio" is inferred from re-encoding reconstructions as JPEG — it never measures the size of the model's own latent representation. Contributors can't compare runs on real compression efficiency.
**Exit criteria:** Evaluation report includes a `latent_bits_per_pixel` / `estimated_bitstream_bytes` field computed from actual quantized latent entropy (or a real coder), for the lossy trainer, without changing default CLI behavior.

Issues:
1. **[image/lossy] Add real bits-per-pixel measurement to evaluation report** — compute empirical entropy of the quantized latent tensor per test batch and log `latent_bits_per_pixel` alongside existing PSNR/SSIM/rate-penalty metrics.
2. **[image/lossy] Prototype a lightweight entropy coder (range/arithmetic) for the latent tensor** — behind a `--enable-entropy-coding` flag (default off), exports actual compressed byte size for the test set.
3. **[image/lossy] Replace `RatePenalty` magnitude heuristic with an entropy-estimate-based rate loss** — opt-in via `--rate-loss-mode {magnitude,entropy}` (default `magnitude` to preserve current benchmarks).
4. **[image/docs] Document real vs proxy compression metrics** — update `README.md`/`documentation/MODEL_PARAMS_INPUT.md` to clearly distinguish "JPEG-baseline comparison" from "true latent bitrate".

### Milestone 2 — Quantization-Aware Training & Latent Quality
**Objective:** Close the train/deploy gap in the lossy trainer's `StraightThroughQuantize` step and give it configurable bit-depth.
**Why:** Current quantization always rounds to a fixed pseudo-8-bit level with plain straight-through gradients — no annealing, no configurable precision, so quantization error isn't learned around.
**Exit criteria:** Lossy trainer supports a `--latent-bit-depth` flag (default 8, preserving current behavior) and a `--quant-noise-anneal` option; smoke test covers both.

Issues:
1. **[image/lossy] Make latent quantization bit-depth configurable** — parameterize `StraightThroughQuantize` with `--latent-bit-depth` (4/6/8/10), default 8 to match current behavior.
2. **[image/lossy] Add additive quantization noise during early training (QAT-style annealing)** — optional `--quant-noise-anneal` flag that fades simulated quantization noise from high to zero over training.
3. **[image/lossy] Add a random-search sweep profile for bit-depth vs PSNR/rate tradeoff** — extend `documentation/random_search_profile.json` with a `latent_bit_depth` axis for the lossy modality.
4. **[image/lossy] Extend smoke test to assert quantized latents are read back losslessly at chosen bit-depth** — add assertion to `smokeTests/smoke_test_image_lossy_local.py`.

### Milestone 3 — Learned Upscaling
**Objective:** Add an optional learned super-resolution model to replace/augment the BICUBIC-only path in `upscale_reconstructed_images.py`.
**Why:** The lossless trainer already trains against a simulated `--upscale-factor` degradation, but the standalone upscaler script only does classic interpolation — the encoder's learned detail recovery isn't leveraged at arbitrary upscale targets.
**Exit criteria:** A new `--upscaler-mode {bicubic,learned}` flag on `upscale_reconstructed_images.py`; a small dedicated SR trainer or reuse of the existing decoder at inference-time with tiling for larger outputs; comparison panels show both.

Issues:
1. **[image/upscale] Design a minimal learned super-resolution head** — small residual CNN trained on the same degrade/upscale-factor pairs already produced by `train_autoencoder_image_local.py`'s data pipeline.
2. **[image/upscale] Add `train_upscaler_image_local.py` trainer script** — mirrors existing trainer conventions (`--preset`, `--params-file`, smoke-testable), outputs its own `evaluation_report.md`.
3. **[image/upscale] Wire `--upscaler-mode learned` into `upscale_reconstructed_images.py`** — falls back to BICUBIC if no learned model path is given (`--upscaler-mode bicubic` stays default).
4. **[image/upscale] Add tiling support for large-image inference** — avoid OOM on M1 by tiling with overlap + blending at model input resolution.
5. **[image/docs] Document the learned upscaling workflow end-to-end** — README section + smoke test (`smokeTests/smoke_test_upscaler_local.py`).

### Milestone 4 — Data Pipeline Robustness & Throughput
**Objective:** Make the data pipeline resilient to messy real-world images and faster to iterate with on repeated runs.
**Why:** Corrupted/truncated files are silently dropped with minimal logging; there's no caching of decoded/patched tensors across runs, which slows local iteration on M1.
**Exit criteria:** Corrupt files are logged with reasons in the run's evaluation report; repeated runs on an unchanged dataset reuse a local cache instead of re-decoding from scratch.

Issues:
1. **[image/data] Add structured logging for skipped/corrupt images** — extend `filter_paths_by_min_size` (and add a decode-error guard) to record counts + sample reasons into `split_info` in the evaluation report.
2. **[image/data] Add optional on-disk TFRecord/tensor cache keyed by (data-dir, block-size, split)** — `--cache-dir` flag, default off, invalidated automatically when inputs change (hash-based).
3. **[image/data] Add a corrupted/degraded-input robustness smoke check** — feed a few intentionally corrupted/blurred/noisy images through inference and assert no crash + bounded PSNR drop.
4. **[image/data] Make degradation noise (σ=0.015) and interpolation choice fully documented/config-surfaced** — confirm `--degrade-interp` and add `--degrade-noise-std` (default 0.015, preserving current behavior).

### Milestone 5 — Reporting, Search & Docs Alignment
**Objective:** Keep `random_search_hyperparams.py`, `documentation/*.md`, and evaluation reports consistent with milestones 1–4 as they land.
**Why:** Every prior milestone adds flags/metrics; without this milestone, docs and search spaces silently drift out of sync (a known repo risk per `AGENT.md`).
**Exit criteria:** Random search can sweep any newly added axis (bit-depth, entropy coding, learned upscaler) behind explicit CLI toggles; `documentation/MODEL_PARAMS_INPUT.md` and `README.md` enumerate every new flag.

Issues:
1. **[image/search] Extend image + image_lossy search spaces for new flags** — bit-depth, rate-loss-mode, quant-noise-anneal, cache-dir toggles added to `random_search_hyperparams.py` and `documentation/random_search_profile.json`.
2. **[image/docs] Full CLI flag audit for both image trainers** — cross-check `--help` output vs `documentation/MODEL_PARAMS_INPUT.md`, fill any gaps.
3. **[image/reporting] Add a compression-ratio/PSNR frontier plot** — extend `plot_training_metrics_from_report.py` / `generate_master_report.py` to chart rate-distortion tradeoff across trials.
4. **[image/pipeline] Sync `run_full_production_pipeline.py` with any renamed/added flags** — verify end-to-end pipeline still runs after milestones 1–4 land.

## 5. Suggested sequencing

Milestone 1 → 2 are prerequisites for meaningful compression benchmarking; Milestone 3
(learned upscaling) is independent and can run in parallel; Milestone 4 (data pipeline)
can start anytime; Milestone 5 is continuous/trailing cleanup after each of the others.

## 6. Out of scope (for this roadmap)

- Audio/video pipelines (tracked separately).
- Cloud/multi-GPU training support.
- Changing existing preset defaults without an explicit, separately-reviewed PR.

