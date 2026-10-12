# Skill: Video Modality

Use this playbook for video trainer, temporal-mode, decode, and video-comparison tasks.

## Primary code surface

- `src/training/train_autoencoder_video_local.py`
- `src/reporting/generate_real_av_comparisons.py`
- `src/training/random_search_hyperparams.py` (video paths)
- `tests/smoke/smoke_test_video_local.py`

## Working context

Video is the most resource-constrained modality in this repo. Small changes in clip shape,
frame count, or decode behavior can have large runtime and memory effects on M1 hardware.

The roadmap also notes that temporal redundancy is largely unused today; many current paths
behave more like independent-frame-stack compression than a true video codec.

## Roadmap snapshot

From `documentation/VIDEO_PIPELINE_ROADMAP.md`:
- **V1 — Real Compression Accounting & Quantization**
- **V2 — Temporal Redundancy & Motion-Aware Compression**
- **V3 — Data Pipeline Robustness & Throughput**
- **V4 — Loss Configurability & Modern Codec Benchmarking**
- **V5 — Reporting, Search & Docs Alignment**

## Likely file map by task type

| Task type | Likely files |
|---|---|
| Trainer metric/loss/quantization change | `train_autoencoder_video_local.py`, docs, `smoke_test_video_local.py` |
| Temporal-mode prototype | `train_autoencoder_video_local.py`, docs, report output, `smoke_test_video_local.py` |
| Decode/cache/robustness work | `train_autoencoder_video_local.py`, docs, possibly reporting if new split info is surfaced |
| Search-space alignment | `random_search_hyperparams.py`, `documentation/random_search_profile.json`, `documentation/RANDOM_SEARCH_PROFILE.md` |
| Video comparison/report changes | `generate_real_av_comparisons.py`, `generate_master_report.py`, `plot_training_metrics_from_report.py` |

## Video-specific cautions

- Preserve divisibility/shape assumptions unless the task explicitly changes them.
- Keep new temporal modes opt-in first; benchmark parity matters.
- Distinguish theoretical compression estimates from actual measured latent sizes.
- If adding codec comparisons, keep current H.264-only behavior as the default unless a task explicitly changes it.

## Validation expectations

Minimum local validation:
- `python tests/smoke/smoke_test_video_local.py`

If search or orchestration is touched, also run:
- `python tests/smoke/smoke_test_random_search.py`
- `python tests/smoke/smoke_test_full_production_pipeline.py`

## Fast task triage

- **Real bitrate or quantization?** Open `agent_skills/compression-models.md`.
- **Temporal work?** Re-read milestone V2 and keep the new path behind an explicit flag.
- **Decode or clip-cache work?** Re-read milestone V3 and check report logging expectations.

