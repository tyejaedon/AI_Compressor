# AGENT.md — Context & Workflow Guide for AI Coding Agents

This file gives any AI agent (GitHub Copilot, Claude, etc.) the context needed to work
safely and productively in this repository. Read this before making changes.

## Project summary

AI_Compressor is a **local-first neural compression** project. It trains autoencoders
for image, audio, and video data (encoder/decoder pairs) and produces reproducible
evaluation reports, benchmarks, and a "best model" production bundle. It is optimized
for iteration on a single laptop (Apple Silicon / M1), not for cloud-scale training.

## Repo map (what lives where)

| Path | Purpose |
|---|---|
| `train_autoencoder_image_local.py` | Image autoencoder trainer (lossless-style). |
| `train_autoencoder_image_lossy_local.py` | Image autoencoder trainer, lossy/rate-distortion focused. |
| `train_autoencoder_audio_local.py` | Audio autoencoder trainer. |
| `train_autoencoder_video_local.py` | Video autoencoder trainer. |
| `train_all_optimal_m1.py` | Orchestrates training across modalities with tuned M1 presets. |
| `random_search_hyperparams.py` | Constrained random hyperparameter search per modality. |
| `run_full_production_pipeline.py` | End-to-end pipeline: search → train → bundle → prune. |
| `build_best_model_bundle.py`, `prune_keep_best_models.py`, `prune_models_keep_top2.py` | Select/package/prune best trained models. |
| `generate_master_report.py`, `report_markdown.py`, `plot_training_metrics_from_report.py` | Reporting and metric visualization. |
| `generate_real_av_comparisons.py`, `generate_real_image_comparisons.py`, `upscale_reconstructed_images.py` | Qualitative comparison/preview generation. |
| `prepare_audio_dataset.py` | Canonicalizes raw audio into a training-ready dataset. |
| `realtime_audio_vortex_pipeline.py`, `vortex_realtime_demo.py` | Real-time inference/demo pipeline. |
| `smokeTests/` | Fast sanity-check scripts (small epoch counts, tiny data subsets). |
| `documentation/` | Requirements, params guides, random-search profile docs. |
| `param_overrides.py` | Shared param/override plumbing used by trainers. |
| `data/`, `models/` (gitignored) | Local datasets and training artifacts — **never commit these**. |

## Ground rules for agents

1. **Never commit large binaries or generated artifacts**: no `data/`, `models/`,
   `*.h5`, `*.keras`, `*.tflite`, `__pycache__/`, `.DS_Store`, `.idea/`. These are
   gitignored — do not force-add them.
2. **Prefer editing existing scripts over duplicating logic.** Most trainers share
   patterns (presets, `--params-file`, output-root conventions) — follow existing
   conventions in files like `train_autoencoder_image_local.py` when adding options.
3. **Keep changes scoped.** This repo has independent modalities (image/audio/video);
   avoid cross-modality edits unless the task requires it.
4. **Validate before proposing a PR**: run the relevant smoke test(s) under
   `smokeTests/` for any modality you touched (see README "Smoke test" section).
   Full smoke tests need local data under `data/`, which is not checked into git —
   CI only runs a lightweight syntax/import check (see `.github/workflows/ci.yml`).
5. **Follow branch/commit conventions** in `CONTRIBUTING.md` (Conventional Commits,
   `feat/*`, `fix/*`, `chore/*`, `docs/*` branch prefixes). Never commit to `master`
   directly — always open a PR.
6. **Update docs alongside behavior changes.** If you add/change a CLI flag, update
   the relevant section of `README.md` and/or `documentation/*.md`.
7. **Don't silently change default hyperparameters** for existing presets — these
   are tied to the benchmark numbers in `README.md`. Call out any such change
   explicitly in the PR description.

## Typical workflows

- **Add/modify a training flag**: edit the trainer's argparse section, thread the
  value through to the training loop/report, update `documentation/MODEL_PARAMS_INPUT.md`
  if it's exposed via `--params-file`, run the matching smoke test.
- **Add a new report/plot**: extend `report_markdown.py` / `plot_training_metrics_from_report.py`,
  keep output paths consistent with existing `models/.../<timestamp>/` layout.
- **Pipeline changes**: `run_full_production_pipeline.py` composes the other scripts
  as subprocess/CLI calls — keep argument names in sync when renaming flags elsewhere.

## Before opening a PR

- [ ] Ran relevant smoke test(s) locally.
- [ ] No files under `data/`, `models/`, or other gitignored paths staged.
- [ ] Branch name matches `feat/…`, `fix/…`, `chore/…`, or `docs/…`.
- [ ] Commit messages follow Conventional Commits (see `CONTRIBUTING.md`).
- [ ] README/docs updated if user-facing behavior changed.
- [ ] PR description filled out using `.github/PULL_REQUEST_TEMPLATE.md`.


