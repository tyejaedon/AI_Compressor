# Milestone Template: M4 / A4 / V4 — Robustness, Configurability, and Benchmark Breadth

Use this template when the task maps to:
- **Image M4** — data pipeline robustness and throughput
- **Audio A4** — multi-channel/rate/noise robustness
- **Video V4** — loss configurability and modern codec benchmarking

## Why these are grouped

These fourth-wave milestones broaden the repo's real-world usefulness:
- image improves data hygiene and iteration speed,
- audio broadens supported input assumptions,
- video gains more configurable optimization and stronger baselines.

The shared rule is to expose broader behavior carefully while keeping current defaults intact.

## Default-safe implementation shape

- Preserve current decode/resample/loss/benchmark defaults.
- Use explicit opt-in flags for broader robustness or benchmarking modes.
- Make skipped/corrupt-input behavior visible in logs or reports instead of silently changing outcomes.

## Typical deliverables by modality

| Modality | Likely deliverables |
|---|---|
| Image (`M4`) | corrupt-file logging, cache-dir support, degradation-parameter surfacing |
| Audio (`A4`) | channel-mode decisions, configurable resampling, noise augmentation, robustness checks |
| Video (`V4`) | loss-profile or explicit weights, optional additional codecs, improved benchmark documentation |

## File-impact map

### Image M4
- `src/training/train_autoencoder_image_local.py`
- possibly shared image data helpers inside that trainer
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_local.py`

### Audio A4
- `src/training/train_autoencoder_audio_local.py`
- `src/audio/prepare_audio_dataset.py` if ingestion assumptions change
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_audio_local.py`
- `tests/smoke/smoke_test_prepare_audio_dataset.py`

### Video V4
- `src/training/train_autoencoder_video_local.py`
- `src/reporting/generate_real_av_comparisons.py` if codec comparison output changes
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_video_local.py`

## Implementation checklist

- [ ] Keep prior defaults benchmark-compatible.
- [ ] Document every new CLI surface.
- [ ] If robustness improves error handling, record enough detail for debugging/reporting.
- [ ] If benchmark breadth expands, keep the old benchmark path as the default.
- [ ] Add smoke coverage for new input assumptions or config flags when feasible.

## Validation commands

```zsh
python tests/smoke/smoke_test_local.py
python tests/smoke/smoke_test_audio_local.py
python tests/smoke/smoke_test_prepare_audio_dataset.py
python tests/smoke/smoke_test_video_local.py
```

Run only the relevant subset for the touched modality.

## PR notes template

- **Milestone:** `M4` / `A4` / `V4`
- **Expanded capability:** robustness / configurability / codec comparison
- **Default behavior impact:** unchanged / opt-in
- **Observability added:** logging/report fields, if any
- **Smoke tests:** list exact commands run

