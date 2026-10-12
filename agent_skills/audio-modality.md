# Skill: Audio Modality

Use this playbook for audio trainer, dataset preparation, and real-time/streaming-adjacent tasks.

## Primary code surface

- `src/training/train_autoencoder_audio_local.py`
- `src/audio/prepare_audio_dataset.py`
- `src/audio/realtime_audio_vortex_pipeline.py`
- `src/audio/vortex_realtime_demo.py`
- `src/training/random_search_hyperparams.py` (audio paths)
- `tests/smoke/smoke_test_audio_local.py`
- `tests/smoke/smoke_test_prepare_audio_dataset.py`
- `tests/smoke/smoke_test_vortex_pipeline.py`

## Working context

The current audio stack separates three concerns:
1. dataset canonicalization,
2. offline autoencoder training/evaluation,
3. the existing vortex visualization/demo path.

Important: the roadmap explicitly says the current vortex pipeline is **not** the real-time compression path for the trained model.

## Roadmap snapshot

From `documentation/AUDIO_PIPELINE_ROADMAP.md`:
- **A1 — Real Compression Accounting & Quantization**
- **A2 — Real-Time Streaming Compression Path**
- **A3 — Perceptual Quality & Psychoacoustic Loss**
- **A4 — Data Pipeline Robustness (Multi-channel, Rate, Noise)**
- **A5 — Reporting, Search & Docs Alignment**

## Likely file map by task type

| Task type | Likely files |
|---|---|
| Trainer metric/loss/quantization change | `train_autoencoder_audio_local.py`, docs, `smoke_test_audio_local.py` |
| Dataset prep / canonicalization change | `prepare_audio_dataset.py`, docs, `smoke_test_prepare_audio_dataset.py` |
| Streaming or real-time compression work | new or existing audio runtime script(s), docs, likely a new smoke test plus `smoke_test_vortex_pipeline.py` if shared code is touched |
| Search-space alignment | `random_search_hyperparams.py`, `documentation/random_search_profile.json`, `documentation/RANDOM_SEARCH_PROFILE.md` |
| Report/plot changes | `generate_master_report.py`, `plot_training_metrics_from_report.py`, `report_markdown.py` |

## Audio-specific cautions

- Distinguish clearly between **theoretical** and **actual** compression accounting.
- Preserve the current vortex demo behavior unless the task explicitly changes it.
- Be explicit about mono/stereo and sample-rate assumptions; avoid silent channel averaging changes.
- If a change touches streaming inference, document latency/chunking behavior.

## Validation expectations

Minimum local validation by area:
- trainer work → `python tests/smoke/smoke_test_audio_local.py`
- dataset prep work → `python tests/smoke/smoke_test_prepare_audio_dataset.py`
- vortex/shared runtime work → `python tests/smoke/smoke_test_vortex_pipeline.py`

If search or orchestration is touched, also run:
- `python tests/smoke/smoke_test_random_search.py`
- `python tests/smoke/smoke_test_full_production_pipeline.py`

## Fast task triage

- **Quantization or real bitrate work?** Open `agent_skills/compression-models.md`.
- **Streaming path work?** Re-read milestone A2 before editing to avoid conflating the compressor with the vortex visualization path.
- **Resampling/channels/noise changes?** Re-read milestone A4 and ensure defaults preserve current behavior.

