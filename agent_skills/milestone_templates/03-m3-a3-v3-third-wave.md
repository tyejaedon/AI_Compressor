# Milestone Template: M3 / A3 / V3 — Quality Expansion and Robustness Scaling

Use this template when the task maps to:
- **Image M3** — learned upscaling
- **Audio A3** — perceptual quality and psychoacoustic loss
- **Video V3** — data pipeline robustness and throughput

## Why these are grouped

These third-wave milestones expand the system beyond the initial core compressor:
- image adds a learned post-process stage,
- audio deepens perceptual quality modeling,
- video improves the practicality of real-data iteration.

The shared rule is to add capability without destabilizing the existing baseline workflow.

## Default-safe implementation shape

- Make the new path optional unless the roadmap explicitly says otherwise.
- Keep small-device constraints visible, especially for M1 laptops.
- Prefer narrow extensions that reuse existing data flow, output layout, and argparse patterns.

## Typical deliverables by modality

| Modality | Likely deliverables |
|---|---|
| Image (`M3`) | learned upscaler mode, trainer, inference tiling/fallback, updated comparison outputs |
| Audio (`A3`) | optional perceptual loss terms or `loss-profile`, additional evaluation metrics |
| Video (`V3`) | structured decode-error logging, cache/worker controls, clip reuse, more robust ingest |

## File-impact map

### Image M3
- `src/training/train_upscaler_image_local.py`
- `src/reporting/upscale_reconstructed_images.py`
- `src/reporting/generate_real_image_comparisons.py`
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_upscaler_local.py`

### Audio A3
- `src/training/train_autoencoder_audio_local.py`
- `src/reporting/report_markdown.py` if new metrics/loss components are surfaced
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_audio_local.py`

### Video V3
- `src/training/train_autoencoder_video_local.py`
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_video_local.py`
- possibly reporting if skipped-file reasons or cache info are surfaced to reports

## Implementation checklist

- [ ] Reuse existing trainer/report/output conventions where possible.
- [ ] Keep new compute-heavy functionality bounded for laptop use.
- [ ] Surface any new mode/weights/workers/cache dirs as explicit flags.
- [ ] Update comparisons or reporting if the new path changes what users inspect visually or numerically.
- [ ] Add smoke coverage that exercises the opt-in path.

## Validation commands

```zsh
python tests/smoke/smoke_test_upscaler_local.py
python tests/smoke/smoke_test_audio_local.py
python tests/smoke/smoke_test_video_local.py
```

Run the command(s) for the touched modality, plus orchestration checks if random search or pipeline wiring changes.

## PR notes template

- **Milestone:** `M3` / `A3` / `V3`
- **Capability added:** learned upscaler / perceptual loss / decode robustness feature
- **Default behavior impact:** unchanged / opt-in
- **Resource notes:** memory, cache, worker, or loss-cost notes
- **Smoke tests:** list exact commands run

