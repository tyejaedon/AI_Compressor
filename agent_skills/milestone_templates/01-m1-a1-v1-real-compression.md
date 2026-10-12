# Milestone Template: M1 / A1 / V1 — Real Compression Accounting

Use this template when the task maps to:
- **Image M1** — real rate-distortion accounting
- **Audio A1** — real compression accounting and quantization
- **Video V1** — real compression accounting and quantization

## Core objective

Replace purely theoretical or proxy compression reporting with a measurable representation
of the model's own latent size, while preserving current benchmark defaults unless the
roadmap item explicitly changes them.

## Default-safe implementation shape

- Prefer **opt-in flags** for new quantization or entropy-coding paths.
- Keep existing proxy/theoretical metrics available for comparison.
- Name new metrics explicitly, e.g. `actual_*`, `estimated_*`, `proxy_*`, `latent_*`.
- Ensure report output makes the distinction obvious.

## Typical deliverables by modality

| Modality | Likely deliverables |
|---|---|
| Image (`M1`) | empirical latent entropy, optional entropy coder, `latent_bits_per_pixel`, clear real-vs-proxy reporting |
| Audio (`A1`) | actual latent quantization, real bytes/clip, side-by-side actual vs theoretical ratios |
| Video (`V1`) | actual latent quantization, real bytes/clip, side-by-side actual vs theoretical ratios |

## File-impact map

### Image M1
- `src/training/train_autoencoder_image_lossy_local.py`
- `src/reporting/report_markdown.py`
- `src/reporting/generate_master_report.py`
- `src/reporting/plot_training_metrics_from_report.py` if frontier plots use the metric
- `documentation/MODEL_PARAMS_INPUT.md`
- `documentation/IMAGE_PIPELINE_ROADMAP.md` if milestone notes need status sync
- `tests/smoke/smoke_test_image_lossy_local.py`

### Audio A1
- `src/training/train_autoencoder_audio_local.py`
- `src/reporting/report_markdown.py`
- `src/reporting/generate_master_report.py`
- `documentation/MODEL_PARAMS_INPUT.md`
- `documentation/AUDIO_PIPELINE_ROADMAP.md` if status wording changes
- `tests/smoke/smoke_test_audio_local.py`

### Video V1
- `src/training/train_autoencoder_video_local.py`
- `src/reporting/report_markdown.py`
- `src/reporting/generate_master_report.py`
- `documentation/MODEL_PARAMS_INPUT.md`
- `documentation/VIDEO_PIPELINE_ROADMAP.md` if status wording changes
- `tests/smoke/smoke_test_video_local.py`

## Implementation checklist

- [ ] Identify where the current compression ratio is estimated or proxied.
- [ ] Add the real-measurement path behind a safe flag if current behavior must remain stable.
- [ ] Thread new metrics into evaluation output.
- [ ] Keep old metrics for before/after comparison unless the roadmap explicitly removes them.
- [ ] Update docs to explain real vs theoretical/proxy reporting.
- [ ] Add or update smoke-test assertions for the new metrics if cheap and deterministic.

## Validation commands

```zsh
python tests/smoke/smoke_test_image_lossy_local.py
python tests/smoke/smoke_test_audio_local.py
python tests/smoke/smoke_test_video_local.py
```

Run only the modality you touched, plus cross-cutting validation if you modified shared reporting or search.

## PR notes template

- **Milestone:** `M1` / `A1` / `V1`
- **Default behavior impact:** preserved / opt-in
- **New metrics:** list exact field names
- **Compression accounting:** explain real vs theoretical/proxy distinction
- **Smoke tests:** list exact commands run

