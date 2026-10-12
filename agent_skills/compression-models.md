# Skill: Compression Model Tasks

Use this playbook for architecture, quantization, loss, metric, or compression-accounting work across any modality.

## Repo-specific model conventions

- Trainers are plain CLI scripts under `src/training/`, not package modules.
- New behavior should mirror existing argparse/preset conventions.
- Shared JSON override behavior follows the `--params-file` pattern documented in `documentation/MODEL_PARAMS_INPUT.md`.
- Published preset defaults are benchmark-sensitive. Avoid silent changes.
- Evaluation outputs usually feed downstream reporting and production-bundle workflows.

## Standard impact map for model changes

A non-trivial model change often touches several layers:
1. the trainer script,
2. config/override plumbing,
3. evaluation report contents,
4. random search support,
5. production pipeline or comparison scripts,
6. docs and smoke tests.

If a task changes CLI flags, metrics, or artifact expectations, assume at least 3-6 are relevant until proven otherwise.

## Common task families

### 1) Real compression accounting
Typical examples:
- actual latent bytes,
- bits per pixel/sample/frame,
- entropy-coded size,
- real vs theoretical ratio reporting.

Check:
- Where current ratio is estimated.
- Whether evaluation reports, comparison scripts, or frontier plots need new fields.
- Whether the feature should be opt-in to preserve benchmark continuity.

### 2) Quantization-aware training
Typical examples:
- configurable bit depth,
- quantization noise annealing,
- straight-through estimators,
- post-training quantized evaluation.

Check:
- Default should usually preserve prior float32 or existing quantization behavior.
- Smoke tests should cover at least one non-default setting if fast enough.
- Docs should distinguish train-time and eval-time quantization behavior.

### 3) Loss or objective changes
Typical examples:
- new loss weights,
- new `--loss-profile`,
- perceptual terms,
- temporal or residual penalties.

Check:
- Whether the modality already has a preset/profile pattern to mirror.
- Whether default weights must remain identical to prior benchmarks.
- Whether report output should expose component losses.

### 4) Architecture or runtime-path changes
Typical examples:
- new decoder heads,
- learned upscalers,
- streaming inference wrappers,
- temporal modes for video.

Check:
- CLI surfacing,
- artifact loading/saving,
- memory constraints on M1 hardware,
- smoke-test path with tiny inputs.

## Cross-cutting file checklist

| Concern | Files to inspect |
|---|---|
| CLI flags | trainer script, `README.md`, `documentation/MODEL_PARAMS_INPUT.md` |
| JSON overrides | `src/training/param_overrides.py`, example JSON docs/files |
| Search-space coverage | `src/training/random_search_hyperparams.py`, `documentation/random_search_profile.json`, `documentation/RANDOM_SEARCH_PROFILE.md` |
| Pipeline compatibility | `src/pipeline/run_full_production_pipeline.py` |
| Reports/plots | `src/reporting/generate_master_report.py`, `src/reporting/plot_training_metrics_from_report.py`, `src/reporting/report_markdown.py` |
| Qualitative comparisons | `src/reporting/generate_real_image_comparisons.py`, `src/reporting/generate_real_av_comparisons.py`, `src/reporting/upscale_reconstructed_images.py` |
| Validation | matching file under `tests/smoke/` |

## Decision heuristics

- Prefer additive flags over behavior-changing rewrites.
- Use milestone names and roadmap wording in comments/docs to keep context discoverable.
- Keep metrics naming explicit: `estimated_*` vs `actual_*`, `proxy_*` vs `latent_*`.
- If a change introduces a new artifact type, make its output-root behavior match existing timestamped run layouts.

## Done criteria for model tasks

- [ ] Behavior is reachable from the intended CLI entry point.
- [ ] Any new config surface is documented.
- [ ] Any new metric is visible in reports where users expect it.
- [ ] Pipeline/search callers remain compatible.
- [ ] Relevant smoke tests pass locally.

