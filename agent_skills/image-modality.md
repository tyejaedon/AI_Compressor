# Skill: Image Modality

Use this playbook for image trainer, lossy image, and learned upscaler tasks.

## Primary code surface

- `src/training/train_autoencoder_image_local.py`
- `src/training/train_autoencoder_image_lossy_local.py`
- `src/training/train_upscaler_image_local.py`
- `src/reporting/upscale_reconstructed_images.py`
- `src/reporting/generate_real_image_comparisons.py`
- `src/training/random_search_hyperparams.py` (image paths)
- `tests/smoke/smoke_test_local.py`
- `tests/smoke/smoke_test_image_lossy_local.py`
- `tests/smoke/smoke_test_upscaler_local.py`

## Working context

The image area has three distinct lanes:
1. **lossless-style image autoencoder** work,
2. **lossy/rate-distortion image autoencoder** work,
3. **learned upscaler** work.

Do not assume a change in one lane belongs in the others.

## Roadmap snapshot

From `documentation/IMAGE_PIPELINE_ROADMAP.md`:
- **M1 — Real Rate-Distortion Accounting**: implemented.
- **M2 — Quantization-Aware Training & Latent Quality**: implemented.
- **M3 — Learned Upscaling**: implemented.
- **M4 — Data Pipeline Robustness & Throughput**: open.
- **M5 — Reporting, Search & Docs Alignment**: continuous/open.

## Likely file map by task type

| Task type | Likely files |
|---|---|
| Lossless trainer flag/loss/data change | `train_autoencoder_image_local.py`, docs, `smoke_test_local.py` |
| Lossy compression/rate/quantization change | `train_autoencoder_image_lossy_local.py`, reporting, docs, `smoke_test_image_lossy_local.py` |
| Upscaler behavior/model change | `train_upscaler_image_local.py`, `upscale_reconstructed_images.py`, docs, `smoke_test_upscaler_local.py` |
| Search-space alignment | `random_search_hyperparams.py`, `documentation/random_search_profile.json`, `documentation/RANDOM_SEARCH_PROFILE.md` |
| Comparison/report output | `generate_real_image_comparisons.py`, `generate_master_report.py`, `plot_training_metrics_from_report.py` |

## Image-specific cautions

- Image presets and defaults are benchmark-linked; avoid silent changes.
- Lossy tasks often have both proxy and true compression metrics; label them clearly.
- Upscaler work must respect M1 memory limits; tiling/fallback behavior matters.
- If a flag changes reconstruction or degradation behavior, audit any qualitative-comparison scripts.

## Validation expectations

Minimum local validation by area:
- lossless trainer work → `python tests/smoke/smoke_test_local.py`
- lossy trainer work → `python tests/smoke/smoke_test_image_lossy_local.py`
- upscaler work → `python tests/smoke/smoke_test_upscaler_local.py`

If search or production orchestration is touched, also run:
- `python tests/smoke/smoke_test_random_search.py`
- `python tests/smoke/smoke_test_full_production_pipeline.py`

## Fast task triage

- **Compression metric or latent bit-depth?** Open `agent_skills/compression-models.md`.
- **Roadmap milestone M4 or M5?** Open `agent_skills/milestone-delivery.md` too.
- **Any flag rename/addition?** Inspect `src/pipeline/run_full_production_pipeline.py` and docs before finishing.

