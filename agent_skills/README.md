# Agent Skills for AI_Compressor

Use this folder as the fast path for repo-specific task execution.

## What these skills are for

These playbooks help agents and contributors work efficiently on:
- roadmap milestones,
- compression-model changes,
- trainer/reporting/pipeline sync work,
- modality-specific smoke-test validation.

They complement, not replace:
- `AGENT.md` for repo-wide working rules,
- `CONTRIBUTING.md` for branch/commit/PR conventions,
- `documentation/*_PIPELINE_ROADMAP.md` for milestone intent and sequencing.

## Quick-start checklist

- [ ] Check `git status --branch` and switch/create an issue-scoped topic branch before editing (`feat/*`, `fix/*`, `docs/*`, etc.); if edits already began on `master`, branch immediately before continuing.
- [ ] Identify the modality: image, audio, video, or cross-cutting pipeline/reporting.
- [ ] Identify the roadmap milestone or issue family, if any.
- [ ] Read the matching skill file below before editing code.
- [ ] Trace all affected CLI surfaces: script flags, `--params-file`, random search, pipeline callers, docs, smoke tests.
- [ ] Keep preset defaults stable unless the task explicitly changes benchmarked behavior.
- [ ] Run the narrowest relevant smoke test(s) locally after changes.

## Skill map

| Skill file | Use when | Primary files |
|---|---|---|
| `agent_skills/milestone-delivery.md` | Turning roadmap items into scoped implementation plans | `documentation/*_PIPELINE_ROADMAP.md`, `.github/PULL_REQUEST_TEMPLATE.md` |
| `agent_skills/milestone_templates/*.md` | Starting from a numbered milestone family (`M1/A1/V1`, `M2/A2/V2`, etc.) | `documentation/*_PIPELINE_ROADMAP.md`, `tests/smoke/*.py` |
| `agent_skills/compression-models.md` | Modifying model architecture, quantization, losses, metrics, or trainer flags | `src/training/*.py`, `src/training/param_overrides.py` |
| `agent_skills/image-modality.md` | Working on image/lossy/upscaler tasks | `src/training/train_autoencoder_image*.py`, `src/reporting/upscale_reconstructed_images.py` |
| `agent_skills/audio-modality.md` | Working on audio training, dataset prep, or streaming | `src/training/train_autoencoder_audio_local.py`, `src/audio/*.py` |
| `agent_skills/video-modality.md` | Working on video training or temporal/decode tasks | `src/training/train_autoencoder_video_local.py` |
| `agent_skills/pipeline-reporting.md` | Working on search, production pipeline, bundling, pruning, reports, and plots | `src/pipeline/*.py`, `src/reporting/*.py`, `src/training/random_search_hyperparams.py` |

## Task routing

### If the task is milestone-driven
1. Read the relevant roadmap in `documentation/`.
2. Open `agent_skills/milestone-delivery.md`.
3. Open the numbered template under `agent_skills/milestone_templates/` that matches the milestone family.
4. Open the modality skill.
5. If the milestone adds or renames flags, also open `agent_skills/pipeline-reporting.md`.

### Milestone template map

- `agent_skills/milestone_templates/01-m1-a1-v1-real-compression.md`
- `agent_skills/milestone_templates/02-m2-a2-v2-second-wave.md`
- `agent_skills/milestone_templates/03-m3-a3-v3-third-wave.md`
- `agent_skills/milestone_templates/04-m4-a4-v4-fourth-wave.md`
- `agent_skills/milestone_templates/05-m5-a5-v5-alignment.md`

### If the task changes compression behavior
Open:
- `agent_skills/compression-models.md`
- plus the matching modality file.

### If the task changes reports, comparisons, or production orchestration
Open:
- `agent_skills/pipeline-reporting.md`
- plus the modality file if model metrics or flags are involved.

## Default done criteria

A task is not really done until all applicable items are true:
- [ ] Code changes are scoped to the intended modality/area.
- [ ] Any new or changed flag is wired through all affected call sites.
- [ ] `README.md` and/or `documentation/*.md` are updated for user-facing changes.
- [ ] Relevant smoke tests were run.
- [ ] No `data/`, `models/`, weights, or generated artifacts are staged.
- [ ] PR notes mention milestone impact and any benchmark-affecting default changes.

