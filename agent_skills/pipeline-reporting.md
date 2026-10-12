# Skill: Pipeline, Search, and Reporting

Use this playbook for cross-cutting changes that connect trainers to search, reports, bundling, or production orchestration.

## Primary code surface

- `src/pipeline/run_full_production_pipeline.py`
- `src/pipeline/build_best_model_bundle.py`
- `src/pipeline/prune_keep_best_models.py`
- `src/pipeline/prune_models_keep_top2.py`
- `src/training/random_search_hyperparams.py`
- `src/reporting/generate_master_report.py`
- `src/reporting/report_markdown.py`
- `src/reporting/plot_training_metrics_from_report.py`
- `src/reporting/generate_real_av_comparisons.py`
- `src/reporting/generate_real_image_comparisons.py`
- `tests/smoke/smoke_test_random_search.py`
- `tests/smoke/smoke_test_full_production_pipeline.py`

## When to use this skill

Open this file if the task involves any of the following:
- adding or renaming trainer flags that are called by another script,
- changing report fields or metric names,
- expanding random-search dimensions,
- keeping bundle/prune logic aligned with new artifact layouts,
- adding new comparison plots or milestone-facing report summaries.

## Core rule: trace flags and metrics end-to-end

If a trainer flag or report field changes, check the whole chain:
1. trainer CLI,
2. params-file support and docs,
3. random-search sampling,
4. production pipeline subprocess calls,
5. report rendering / plots / comparisons,
6. smoke tests.

## Common change patterns

### 1) New search dimension
Update:
- `src/training/random_search_hyperparams.py`
- `documentation/random_search_profile.json`
- `documentation/RANDOM_SEARCH_PROFILE.md`
- relevant modality docs if user-facing

### 2) New trainer flag used by the production pipeline
Update:
- trainer argparse and behavior,
- `src/pipeline/run_full_production_pipeline.py`,
- related roadmap/docs,
- matching smoke test(s).

### 3) New evaluation metric
Update:
- trainer evaluation output,
- `src/reporting/report_markdown.py`,
- any master-report aggregation or plotting code,
- comparison scripts if they display that metric.

### 4) Artifact selection/bundling logic
Update:
- bundling/pruning scripts,
- assumptions about report names, metrics, and output directories,
- smoke test if artifact flow changes.

## Cross-cutting cautions

- Random-search defaults should remain conservative and explicit.
- Keep timestamped output-root conventions aligned with existing scripts.
- Do not break CLI compatibility for subprocess callers by accident.
- If metric naming changes, prefer additive aliases or migration notes over silent replacement.

## Validation expectations

Minimum local validation for cross-cutting work:
- `python tests/smoke/smoke_test_random_search.py`
- `python tests/smoke/smoke_test_full_production_pipeline.py`

If a modality-specific metric or flag is involved, also run that modality's smoke test.

## Done criteria

- [ ] Search profiles and docs reflect new axes.
- [ ] Pipeline callers still invoke downstream scripts correctly.
- [ ] Reports/plots render the expected fields.
- [ ] Smoke tests cover the touched orchestration path.

