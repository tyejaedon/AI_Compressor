# Copilot instructions for AI_Compressor

Condensed context for AI coding agents. See `AGENT.md` at the repo root for full detail.

## What this repo is
Local-first neural compression (autoencoders) for image/audio/video, tuned for
iteration on a single laptop (M1). Entry points are top-level CLI scripts, not a
package/module — everything is run via `python <script>.py --flags`.

## Setup & run
```zsh
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r documentation/requirements.txt
```
Train image: `python train_autoencoder_image_local.py --preset m1-air-balanced --data-dir data --output-root models/local_run`

Smoke tests (fast, tiny data/epochs): `python smokeTests/smoke_test_*.py`

## Conventions to follow
- Branch names: `feat/*`, `fix/*`, `chore/*`, `docs/*`.
- Commits: Conventional Commits (`feat: …`, `fix: …`, `chore: …`).
- Never commit `data/`, `models/`, `*.h5`, `*.keras`, `*.tflite`, `__pycache__/`.
- No direct commits to `master` — always via PR (branch protection enforced).
- Trainers share a `--params-file` (JSON) convention — see `documentation/MODEL_PARAMS_INPUT.md`.
- Don't change existing preset defaults without flagging it explicitly in the PR — they
  are tied to the benchmark numbers published in `README.md`.

## Validation expectations
CI only runs a lightweight syntax/import check (no GPU/data in CI). Before proposing
changes, run the smoke test matching the modality you touched, e.g.:
```zsh
python smokeTests/smoke_test_image_lossy_local.py
python smokeTests/smoke_test_audio_local.py
python smokeTests/smoke_test_video_local.py
```

## When editing
- Mirror existing argparse/preset patterns in trainer scripts rather than inventing new ones.
- Update `README.md` / `documentation/*.md` when adding or renaming CLI flags.
- Keep pipeline scripts (`run_full_production_pipeline.py`) in sync if you rename flags
  in the scripts they call.

