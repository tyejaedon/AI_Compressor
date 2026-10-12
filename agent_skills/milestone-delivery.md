# Skill: Milestone Delivery

Use this playbook when a task comes from one of the roadmap files in `documentation/`.

## Goals

Translate a roadmap milestone or issue into a small, reviewable change that:
- preserves current defaults unless explicitly intended,
- keeps docs/search/pipeline surfaces aligned,
- is validated with the right smoke tests.

## Companion templates

Use this file for the general process, then open the numbered template in
`agent_skills/milestone_templates/` that matches the roadmap family:

- `01-m1-a1-v1-real-compression.md`
- `02-m2-a2-v2-second-wave.md`
- `03-m3-a3-v3-third-wave.md`
- `04-m4-a4-v4-fourth-wave.md`
- `05-m5-a5-v5-alignment.md`

Those templates give milestone-family-specific implementation prompts, likely files,
validation targets, and PR-note guidance.

## Inputs to collect before editing

- Modality: image / audio / video / cross-cutting.
- Milestone name and issue title from the roadmap.
- Whether the work is:
  - a new metric,
  - a new training flag,
  - a loss/model change,
  - a data-pipeline change,
  - a reporting/pipeline follow-up.
- Whether published benchmark defaults must stay frozen.

## Execution pattern

### 1) Pin the milestone scope
Summarize the task in this format:
- **Milestone:** `M*`, `A*`, or `V*`
- **Objective:** one sentence from the roadmap
- **Default behavior impact:** none / opt-in / breaking
- **Validation target:** exact smoke test(s)

### 2) Build the file-impact map
For most roadmap tasks, inspect this chain in order:
1. trainer or runtime script,
2. shared params plumbing (`src/training/param_overrides.py`) if `--params-file` support is needed,
3. search/orchestration (`src/training/random_search_hyperparams.py`, `src/pipeline/run_full_production_pipeline.py`),
4. reporting or comparison scripts,
5. `README.md` and `documentation/*.md`,
6. smoke tests.

### 3) Preserve repo invariants
- Keep scripts runnable as `python src/.../script.py`.
- Do not silently change preset defaults tied to `README.md` benchmarks.
- Prefer opt-in flags for milestone experiments.
- Avoid cross-modality edits unless the milestone truly spans modalities.

### 4) Use milestone-aware implementation patterns

| Task type | Preferred pattern |
|---|---|
| New metric | Add metric calculation, report field, and smoke-test assertion if cheap |
| New training flag | Add argparse flag, thread into config/use site, document it, test it |
| Search-space addition | Gate via explicit search toggle/profile entry; do not silently widen defaults |
| Pipeline rename/addition | Keep subprocess callers synchronized with callee flags |
| Robustness feature | Preserve existing behavior by default; make enhanced path opt-in where possible |

## Milestone completion checklist

- [ ] Work maps cleanly to one roadmap issue or a tightly related pair.
- [ ] Existing default CLI behavior still reproduces the previous path unless the task explicitly changes it.
- [ ] User-facing docs mention the new flag/metric/behavior.
- [ ] Pipeline/reporting/search callers are synchronized.
- [ ] Relevant smoke tests ran locally.
- [ ] PR notes mention milestone name and any benchmark impact.

## Validation matrix

| Area touched | Minimum validation |
|---|---|
| Image trainer / lossy / upscaler | `tests/smoke/smoke_test_local.py`, `smoke_test_image_lossy_local.py`, or `smoke_test_upscaler_local.py` as applicable |
| Audio trainer / dataset prep / streaming-adjacent work | `tests/smoke/smoke_test_audio_local.py`, `smoke_test_prepare_audio_dataset.py`, `smoke_test_vortex_pipeline.py` as applicable |
| Video trainer | `tests/smoke/smoke_test_video_local.py` |
| Random search / pipeline orchestration | `tests/smoke/smoke_test_random_search.py`, `smoke_test_full_production_pipeline.py` |
| Reporting-only changes | Run the nearest modality smoke test if report fields depend on generated training output; otherwise verify CLI/import path |

## Common failure modes to avoid

- Adding a flag to a trainer but not to `--params-file` documentation.
- Renaming a flag without updating `src/pipeline/run_full_production_pipeline.py`.
- Changing defaults that affect benchmarked presets without calling it out.
- Updating roadmap-facing functionality without updating the smoke test.
- Mixing unrelated milestone work into one PR.

