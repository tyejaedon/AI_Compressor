# Milestone Template: M5 / A5 / V5 — Reporting, Search, and Docs Alignment

Use this template when the task maps to:
- **Image M5** — reporting, search, and docs alignment
- **Audio A5** — reporting, search, and docs alignment
- **Video V5** — reporting, search, and docs alignment

## Core objective

Keep roadmap progress discoverable and usable by making trainer flags, random-search axes,
pipeline orchestration, reports, and documentation stay in sync.

## Shared rule

For these milestones, the biggest risk is silent drift. Assume that any new flag or metric
must be traced end-to-end before the task is done.

## Standard impact map

- `src/training/random_search_hyperparams.py`
- `documentation/random_search_profile.json`
- `documentation/RANDOM_SEARCH_PROFILE.md`
- `documentation/MODEL_PARAMS_INPUT.md`
- `README.md` if user-facing behavior or workflows changed
- `src/pipeline/run_full_production_pipeline.py`
- `src/reporting/report_markdown.py`
- `src/reporting/generate_master_report.py`
- `src/reporting/plot_training_metrics_from_report.py`
- relevant modality smoke test(s)
- `tests/smoke/smoke_test_random_search.py`
- `tests/smoke/smoke_test_full_production_pipeline.py`

## Typical deliverables

| Area | Likely deliverables |
|---|---|
| Search | new sweep axes, explicit toggles, updated profiles/docs |
| Reporting | new metric fields, frontier plots, summary tables |
| Pipeline | synchronized subprocess flags and artifact assumptions |
| Docs | audited `--help` parity with roadmap/user docs |

## Implementation checklist

- [ ] Audit trainer `--help` output against docs.
- [ ] Update random-search profiles for any new meaningful search axis.
- [ ] Keep production pipeline subprocess invocations in sync with current flag names.
- [ ] Ensure report rendering/plots handle new metric names or fields.
- [ ] Run both orchestration smoke tests if search or pipeline code changed.

## Validation commands

```zsh
python tests/smoke/smoke_test_random_search.py
python tests/smoke/smoke_test_full_production_pipeline.py
```

Also run the modality smoke test whose trainer or metrics changed.

## PR notes template

- **Milestone:** `M5` / `A5` / `V5`
- **Alignment scope:** search / pipeline / reporting / docs
- **Flag or metric audit:** summarize what was synchronized
- **Default behavior impact:** unchanged unless stated otherwise
- **Smoke tests:** list exact commands run

