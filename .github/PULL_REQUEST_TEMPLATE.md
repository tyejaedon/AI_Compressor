## Summary

<!-- What does this PR do and why? -->

## Linked issues

<!--
Use a GitHub closing keyword (Closes/Fixes/Resolves) for issues this PR fully
resolves — they'll auto-close on merge. Use "Refs"/"Part of" for issues this
PR only relates to or partially addresses (won't auto-close).
One per line, e.g.:
  Closes #123
  Refs #45
-->

Closes #

## Milestone

<!--
Which milestone (if any) does this belong to? Match the naming used in
documentation/{IMAGE,AUDIO,VIDEO}_PIPELINE_ROADMAP.md (e.g. "M3: Learned
Upscaling", "A2: Streaming Inference") or a release milestone (e.g. "v0.2 -
audio quality"). The assignee/reviewer sets this on the PR sidebar to match.
-->

- Milestone:

## Type of change

- [ ] feat — new feature
- [ ] fix — bug fix
- [ ] chore — tooling/maintenance
- [ ] docs — documentation only
- [ ] refactor — no behavior change
- [ ] perf — performance improvement
- [ ] test — smoke/test changes only

## Modality / area affected

- [ ] Image
- [ ] Audio
- [ ] Video
- [ ] Pipeline / orchestration (`run_full_production_pipeline.py`, bundling, pruning)
- [ ] Reporting / plotting
- [ ] Docs / repo infra only

## Labels

<!--
`.github/labeler.yml` auto-applies modality/area labels (image, audio, video,
pipeline, docs, infra) based on changed file paths once this PR is opened —
double check they landed correctly in the sidebar. Type labels below aren't
automated; check the ones that apply so a reviewer can add them:
-->

- [ ] `bug`
- [ ] `enhancement`
- [ ] `chore`

## Checklist

- [ ] Branch name follows `feat/*`, `fix/*`, `chore/*`, `docs/*`, `refactor/*`, `perf/*`, or `test/*`.
- [ ] Commit messages follow Conventional Commits.
- [ ] Ran the relevant smoke test(s) locally and they pass (list below).
- [ ] No files under `data/`, `models/`, or other gitignored paths are staged.
- [ ] Updated `README.md` / `documentation/*.md` if user-facing CLI flags or behavior changed.
- [ ] Default hyperparameters for existing presets are unchanged, or the change is
      called out explicitly below with justification.

## Smoke tests run

```
<!-- e.g. python tests/smoke/smoke_test_image_lossy_local.py -->
```

## Metric / quality impact (if training/model code changed)

<!-- PSNR/SSIM/SNR before vs after, or "N/A" -->

## Notes for reviewers

<!-- Anything else reviewers should know -->

