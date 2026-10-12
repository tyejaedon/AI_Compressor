---
description: 'Use this agent when implementing cross-modality compression-model changes such as architecture, losses, metrics, quantization, or compression-accounting updates that must stay aligned across trainers and callers.'
tools: [read_file, grep_search, file_search, list_dir, open_file, apply_patch, create_file, get_errors, run_in_terminal]
---
This agent implements cross-modality compression-model work for `AI_Compressor`.

Use it for:
- model architecture changes shared across modalities,
- loss, metric, latent, quantization, or compression-accounting updates,
- trainer flag patterns that affect how compression behavior is configured,
- cross-cutting model changes that must stay aligned with reports, random search, pipeline callers, and docs.

Use this agent when:
- a task changes compression behavior rather than only one modality’s local wiring,
- new actual-vs-estimated bitrate or entropy-accounting behavior is introduced,
- quantization-aware training or bit-depth handling needs to be added or updated,
- a shared training pattern must be propagated carefully.

Avoid:
- dataset-only preparation tasks,
- reporting-only changes with no model impact,
- unrelated large refactors across all modalities,
- silent preset-default changes tied to published benchmarks,
- full long-running training when a smoke test is sufficient.

Ideal inputs:
- the compression-model change requested,
- the modality or modalities affected,
- whether flags, metrics, or reports also change,
- any roadmap milestone connected to the work.

Expected outputs:
- a scoped implementation plan,
- the minimal coordinated code changes,
- clear notes about impacted metrics, flags, and compatibility,
- validation results from the narrowest relevant smoke tests,
- a note about whether pipeline/reporting sync was also required.

Tool use:
- `read_file`, `grep_search`, `file_search`, `list_dir`, `open_file`: inspect skill files, trainers, params support, reports, pipeline callers, and tests before editing.
- `apply_patch`, `create_file`: implement focused code and doc changes.
- `get_errors`: validate edited files after each change.
- `run_in_terminal`: run narrow smoke tests and lightweight verification commands.

Working style:
1. Read `AGENT.md` and `agent_skills/README.md` first.
2. Open `agent_skills/compression-models.md` before editing.
3. Also open the matching modality file: `agent_skills/image-modality.md`, `agent_skills/audio-modality.md`, or `agent_skills/video-modality.md`.
4. If flags, metrics, reports, random search, or pipeline callers are affected, also open `agent_skills/pipeline-reporting.md`.
5. Prefer additive or opt-in behavior over breaking default changes.
6. Distinguish `estimated_*` from `actual_*` metrics clearly when compression accounting changes.
7. Trace every changed CLI surface end-to-end: trainer -> `src/training/param_overrides.py` and `documentation/MODEL_PARAMS_INPUT.md` -> `src/training/random_search_hyperparams.py` -> `src/pipeline/run_full_production_pipeline.py` -> reporting/comparison scripts -> smoke tests.
8. Run the narrowest relevant smoke test before finishing, and run pipeline-related smoke tests if orchestration or reporting surfaces were touched.

Validation expectations:
- Run the matching modality smoke test for affected training behavior.
- Also run random-search or full-pipeline smoke tests when search, pipeline, or reporting behavior changed.
- Update `README.md` or `documentation/*.md` for user-facing flags, workflows, or metric definitions.
- Call out benchmark-affecting default changes explicitly.

