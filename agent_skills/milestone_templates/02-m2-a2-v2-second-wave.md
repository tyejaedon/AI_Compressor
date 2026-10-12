# Milestone Template: M2 / A2 / V2 — Quantization, Streaming, and Temporal Modes

Use this template when the task maps to:
- **Image M2** — quantization-aware training and latent quality
- **Audio A2** — real-time streaming compression path
- **Video V2** — temporal redundancy and motion-aware compression

## Why these are grouped

These second-wave milestones all move beyond the initial baseline into a more deployment-
or modality-aware operating mode:
- image learns around quantization,
- audio adds chunked real-time inference,
- video begins exploiting temporal redundancy.

The shared rule is the same: keep the advanced path **explicit, testable, and benchmark-safe**.

## Default-safe implementation shape

- Add new behavior behind a **flag or explicit mode selector**.
- Preserve existing offline/independent/default execution unless the roadmap says otherwise.
- Favor small, smoke-testable prototypes over broad refactors.
- Surface operational limits clearly: latency, memory, bit depth, frame dependencies, etc.

## Typical deliverables by modality

| Modality | Likely deliverables |
|---|---|
| Image (`M2`) | configurable latent bit depth, quantization-noise anneal, improved latent-quality reporting |
| Audio (`A2`) | streaming/chunked inference wrapper, new real-time compression script, latency notes |
| Video (`V2`) | `temporal-mode` selector, residual/inter-frame prototype, comparison metrics vs independent mode |

## File-impact map

### Image M2
- `src/training/train_autoencoder_image_lossy_local.py`
- `src/training/random_search_hyperparams.py`
- `documentation/MODEL_PARAMS_INPUT.md`
- `documentation/random_search_profile.json`
- `documentation/RANDOM_SEARCH_PROFILE.md`
- `tests/smoke/smoke_test_image_lossy_local.py`

### Audio A2
- likely a new runtime file under `src/audio/` and/or shared logic reused from existing audio scripts
- `src/training/train_autoencoder_audio_local.py` if export/loading assumptions change
- `documentation/AUDIO_PIPELINE_ROADMAP.md` and README/docs for the streaming path
- `tests/smoke/` with a streaming-specific smoke test if introduced
- `tests/smoke/smoke_test_vortex_pipeline.py` if shared runtime code changes

### Video V2
- `src/training/train_autoencoder_video_local.py`
- `src/reporting/report_markdown.py` and/or aggregate reporting if mode comparison is surfaced
- `documentation/MODEL_PARAMS_INPUT.md`
- `tests/smoke/smoke_test_video_local.py`

## Implementation checklist

- [ ] Add a mode/flag that clearly selects the new path.
- [ ] Preserve the old default path unchanged.
- [ ] Document operational tradeoffs in user-facing docs.
- [ ] Ensure evaluation output distinguishes baseline vs new mode.
- [ ] Add the smallest viable smoke-test coverage for the new mode.

## Validation commands

```zsh
python tests/smoke/smoke_test_image_lossy_local.py
python tests/smoke/smoke_test_vortex_pipeline.py
python tests/smoke/smoke_test_video_local.py
```

Use the modality-specific command(s) that match the change. If you add a new streaming smoke test, run it too.

## PR notes template

- **Milestone:** `M2` / `A2` / `V2`
- **New mode/flag:** name it exactly
- **Default behavior impact:** unchanged / opt-in
- **Operational constraints:** latency, memory, frame dependency, or bit-depth notes
- **Smoke tests:** list exact commands run

