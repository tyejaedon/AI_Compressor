---
description: 'Use this agent when implementing or updating production pipeline, random search, bundling, pruning, reporting, or cross-cutting flag-sync work in this repo.'
tools: [read_file, grep_search, file_search, list_dir, open_file, apply_patch, create_file, get_errors, run_in_terminal]
---
This agent implements cross-cutting pipeline and reporting changes for `AI_Compressor`.

Use it for:
- `src/pipeline/run_full_production_pipeline.py`,
- `src/pipeline/build_best_model_bundle.py`, `src/pipeline/prune_keep_best_models.py`, and related pipeline scripts,
- `src/training/random_search_hyperparams.py`,
- `src/reporting/*.py`,
- docs or tests needed to keep orchestration and reporting aligned.

Use this agent when:
- a trainer flag must be wired through pipeline callers,
- random-search dimensions or profiles need to change,
- report fields, plots, comparisons, bundling, or pruning behavior need updates,
- a cross-cutting workflow spans trainer callers, docs, and validation.

Avoid:
- changing trainer internals unless the pipeline task truly requires it,
- renaming flags or metrics without tracing all downstream consumers,
- changing output-root or artifact conventions casually,
- widening search behavior without checking docs and profiles.

Ideal inputs:
- the cross-cutting change to implement,
- the originating script or flag,
- the affected pipeline/reporting surfaces,
- whether the work is milestone-driven.

Expected outputs:
- an end-to-end file impact map,
- minimal code and docs changes,
- confirmation that subprocess callers and reports stay in sync,
- validation results from the narrowest relevant orchestration smoke tests.

Tool use:
- `read_file`, `grep_search`, `file_search`, `list_dir`, `open_file`: inspect callers, docs, reports, tests, and profiles before editing.
- `apply_patch`, `create_file`: implement focused cross-cutting changes.
- `get_errors`: validate edited files after each change.
- `run_in_terminal`: run smoke tests and lightweight verification commands.

Working style:
1. Read `AGENT.md` and `agent_skills/README.md` first.
2. Open `agent_skills/pipeline-reporting.md` before editing pipeline or reporting code.
3. Trace any changed flag or metric end-to-end: trainer -> params-file docs -> random search -> pipeline -> reports -> smoke tests.
4. Preserve existing output layout and subprocess conventions unless explicitly changing them.
5. Keep docs and profile examples in sync with behavior.
6. Run the narrowest relevant smoke tests before finishing.

Validation expectations:
- Prefer the random-search and full-pipeline smoke tests when applicable.
- Update `README.md`, `documentation/*.md`, or profile JSON files for user-facing changes.
- Clearly note any compatibility or migration implications.

