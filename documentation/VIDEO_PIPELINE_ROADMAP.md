# Video Compression Pipeline — Development Roadmap

Status: Draft v1 · Owner: video modality · Companion to `AGENT.md` / `CONTRIBUTING.md`

This roadmap targets `src/training/train_autoencoder_video_local.py`, video-related paths in
`src/reporting/generate_real_av_comparisons.py`, and the video paths in
`src/training/random_search_hyperparams.py` / `src/training/param_overrides.py`.

## 1. Current state (as of this roadmap)

| Area | Today |
|---|---|
| Architecture | Conv3D encoder (3 stages, strides `(1,2,2)→(2,2,2)→(1,2,2)`: 2× temporal / 8× spatial downsample) → `GlobalAveragePooling3D` → Dense bottleneck (L1/L2-regularizable) → Dense-reconstructed decoder with symmetric `UpSampling3D` + skip connections → sigmoid output. |
| Temporal modeling | **None** — each clip's frames are flattened before SSIM/PSNR computation and the loss treats frames independently; no motion compensation, optical flow, or inter-frame prediction. Video is effectively compressed as an independent frame stack. |
| Data pipeline | `ffmpeg` (system or `imageio-ffmpeg` fallback) decodes `.mp4/.mov/.mkv/.avi/.webm/.m4v` to raw RGB frames at a fixed `fps`/`height`/`width`; clips extracted via a sliding window (`--real-clip-stride`, default 4) capped by `--real-max-clips` (default 1800). Single-threaded, one video at a time. |
| Compression accounting | **Theoretical only**, same pattern as audio/image-lossy: `estimate_video_compression_ratio()` compares raw frame bits vs. `latent_dim * latent_bits` — latent is dense float32, never actually quantized. |
| Loss/metrics | Hardcoded `0.75*MSE + 0.15*L1 + 0.10*SSIM` (not configurable via CLI, unlike the image trainer's `--loss-profile`); tracks MSE/PSNR/SSIM per-frame. `TargetPSNRCallback` early-stops at `--target-psnr` (default 25.0). |
| Cost profile | Explicitly called out in `README.md` as "the most expensive path and highly sensitive to clip shape" — smallest presets use `batch_size=3-4`, `epochs<=36`, tiny frame/resolution smoke test (6 frames, 48×48). |
| Robustness | No corrupted/variable-framerate video handling beyond a broad try/except that silently skips a file; no dynamic resolution (height/width must be divisible by 8, frames by 2); no real data fallback (raises immediately if no videos found). |
| Search & docs | `src/training/random_search_hyperparams.py` samples latent_dim/filters/kernel/batch/epochs/lr/frames/height/width for video; ranks by `psnr_metric`. `--params-file` JSON `video` section documented in `documentation/MODEL_PARAMS_INPUT.md`. |

## 2. Constraints (do not violate)

1. **Laptop-only (M1) compute** — video is already the most compute/thermal-constrained modality; every milestone needs a tiny/smoke-testable path (few clips, small resolution, 1 epoch).
2. **Frozen preset defaults** (`m1-air-fast/balanced/quality`) tied to `README.md` benchmark numbers — changes must be called out explicitly.
3. **CI stays lightweight** — no training/data-dependent jobs in CI; validate via `tests/smoke/smoke_test_video_local.py` locally.
4. **No installable-package refactor** — scripts live under `src/{training,pipeline,reporting,audio}/` for discoverability but remain plain CLI scripts (`python src/.../script.py`), not an installable package.
5. **Backward-compatible CLI** — new flags default to reproducing current behavior.
6. **Clip-shape sensitivity** — any new feature must not silently change the frames/8-divisibility or height/width/8-divisibility constraints without explicit opt-in.

## 3. Goals

- **G1 — Real compression accounting**: measure actual bits/frame, not a theoretical estimate.
- **G2 — Exploit temporal redundancy**: move from independent-frame compression toward inter-frame prediction, since video's main compression opportunity (motion redundancy) is currently unused.
- **G3 — Data pipeline robustness & throughput**: parallel decode, corrupted/VFR video handling, dynamic resolution.
- **G4 — Configurable loss & broader codec benchmarking**: bring video's loss configurability up to parity with the image trainer, and benchmark against more than just H.264 CRF sweep.
- **G5 — Discoverability**: docs, search space, and reports stay in sync as the above land.

## 4. Milestones

### Milestone V1 — Real Compression Accounting & Quantization
**Objective:** Replace the theoretical bits/latent-dim estimate with a real measured bitrate for video clips.
**Why:** Same gap as audio/image-lossy — `--latent-bits` only feeds an arithmetic estimate; the latent is float32 end-to-end.
**Exit criteria:** Evaluation report includes `actual_latent_bytes_per_clip` from a real quantization step, opt-in and defaulting to current behavior.

Issues:
1. **[video/compression] Add real bit-depth quantization of the video latent vector** — `--latent-bit-depth` opt-in flag, default preserves float32 behavior.
2. **[video/compression] Add straight-through estimator for quantized latent gradients (video)** — train-time awareness of the quantization step.
3. **[video/compression] Add real vs theoretical compression metrics to the video evaluation report.**
4. **[video/docs] Document real vs theoretical video compression accounting.**

### Milestone V2 — Temporal Redundancy & Motion-Aware Compression
**Objective:** Start exploiting inter-frame redundancy instead of treating each clip as an independent frame stack.
**Why:** This is the single biggest untapped compression opportunity for video specifically (vs. image/audio) — currently zero motion compensation or temporal prediction exists.
**Exit criteria:** A prototype mode (behind a flag) that predicts frame *t* from frame *t-1* (or a shared temporal latent) and reports a measurable PSNR/bitrate improvement over the baseline independent-frame mode on the same smoke-test data, without changing default behavior.

Issues:
1. **[video/temporal] Prototype a frame-delta/residual coding mode** — encode first frame directly, subsequent frames as residuals against a (learned or simple) prediction of the previous frame; `--temporal-mode {independent,residual}`, default `independent`.
2. **[video/temporal] Add a lightweight learned inter-frame predictor (e.g. ConvLSTM or simple warping head)** — optional component feeding the residual-coding mode above.
3. **[video/temporal] Add PSNR/bitrate comparison between independent vs residual temporal modes to the evaluation report.**
4. **[video/docs] Document the temporal-mode tradeoffs and current limitations (no true motion compensation/optical flow yet).**

### Milestone V3 — Data Pipeline Robustness & Throughput
**Objective:** Make video decoding resilient and faster to iterate with locally.
**Why:** Decoding is single-threaded (one video at a time via subprocess `ffmpeg` call), corrupted/VFR videos are silently skipped with minimal logging, and there's no caching of extracted clips across runs.
**Exit criteria:** Corrupt/unreadable videos are logged with reasons in the evaluation report; repeated runs on an unchanged dataset can reuse a local clip cache; multiple videos can be decoded in parallel.

Issues:
1. **[video/data] Add structured logging for skipped/corrupt/unreadable video files** — record counts + sample reasons into `split_info` in the evaluation report instead of silent skip.
2. **[video/data] Parallelize video decoding across files** — use a process/thread pool for the per-video `ffmpeg` subprocess calls, bounded by `--decode-workers` (default 1, preserving current behavior).
3. **[video/data] Add optional on-disk clip cache** — `--cache-dir` flag keyed by (data-dir, frames, height, width, fps, stride), default off, hash-invalidated.
4. **[video/data] Add dynamic-resolution / non-divisible-by-8 input handling (auto-pad or auto-crop)** — currently a hard `raise` if height/width aren't divisible by 8.

### Milestone V4 — Loss Configurability & Modern Codec Benchmarking
**Objective:** Bring the video trainer's loss function up to parity with the image trainer's `--loss-profile` pattern, and broaden the benchmark beyond H.264 CRF sweep.
**Why:** `video_loss` weights (0.75/0.15/0.10) are hardcoded, unlike the image and (proposed) audio trainers; the benchmark only compares against H.264 at 3 CRF values.
**Exit criteria:** `--loss-profile {balanced,psnr,perceptual}` mirrors the image trainer, `balanced` reproduces current default weights exactly; benchmark can optionally include at least one additional codec (e.g. VP9/AV1 via ffmpeg) behind a flag.

Issues:
1. **[video/loss] Introduce `--loss-mse-weight`/`--loss-l1-weight`/`--loss-ssim-weight` CLI flags and `--loss-profile` presets** — `balanced` profile matches current hardcoded 0.75/0.15/0.10 exactly.
2. **[video/eval] Add an edge/motion-aware loss term option** — e.g. penalize frame-to-frame gradient mismatches, complementing Milestone V2's temporal work.
3. **[video/benchmark] Add an optional VP9 or AV1 comparison to the MP4 benchmark** — `--benchmark-codecs h264,vp9`, default `h264` only (current behavior).
4. **[video/docs] Document the new loss/benchmark flags in README and MODEL_PARAMS_INPUT.md.**

### Milestone V5 — Reporting, Search & Docs Alignment
**Objective:** Keep `src/training/random_search_hyperparams.py`, `documentation/*.md`, and evaluation reports consistent with V1–V4 as they land.
**Why:** Every milestone above adds flags/metrics; without this milestone docs and search spaces drift.
**Exit criteria:** Random search can sweep any new axis (bit-depth, temporal-mode, loss-profile, decode-workers) behind explicit toggles; docs enumerate every new flag.

Issues:
1. **[video/search] Extend video search space for new flags** — bit-depth, temporal-mode, loss-profile added to `src/training/random_search_hyperparams.py` and `documentation/random_search_profile.json`.
2. **[video/docs] Full CLI flag audit for the video trainer.**
3. **[video/reporting] Add a compression-ratio/PSNR frontier plot for video trials** — extend `src/reporting/plot_training_metrics_from_report.py` / `src/reporting/generate_master_report.py`.
4. **[video/pipeline] Sync `src/pipeline/run_full_production_pipeline.py` with any renamed/added video flags.**

## 5. Suggested sequencing

V1 (real accounting) should land before V4's benchmark work so tradeoffs are measured against real bitrate. V2 (temporal redundancy) is the highest-value, highest-risk milestone and should be prototyped in isolation behind a flag before any default changes are considered. V3 (data pipeline) can proceed independently at any time given it's the most compute/time-constrained modality. V5 is continuous/trailing.

## 6. Out of scope (for this roadmap)

- Image/audio pipelines (tracked separately).
- Cloud/multi-GPU training support.
- Changing existing preset defaults without an explicit, separately-reviewed PR.
- Full optical-flow-based motion compensation (V2 only prototypes a lightweight residual-coding step as a first move in that direction).

