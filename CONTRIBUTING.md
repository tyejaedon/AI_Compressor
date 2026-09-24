# Contributing to AI_Compressor

Thanks for contributing! This document defines the branch, commit, PR, issue, and
release conventions for this repo.

## Branching model

- `master` is the protected, always-releasable branch. **No direct commits or pushes.**
- All work happens on a topic branch, opened as a PR into `master`.
- Branch name prefixes (required):

  | Prefix | Use for |
  |---|---|
  | `feat/…` | New features (e.g. `feat/video-augmentations`) |
  | `fix/…` | Bug fixes (e.g. `fix/audio-nan-loss`) |
  | `chore/…` | Tooling, deps, repo maintenance, cleanup |
  | `docs/…` | Documentation-only changes |
  | `refactor/…` | Non-behavioral code restructuring |
  | `perf/…` | Performance improvements |
  | `test/…` | Test/smoke-test only changes |

## Commit messages — Conventional Commits

Format: `<type>(optional-scope): <short summary>`

Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`.

Examples:
```
feat(image-trainer): add mixup augmentation option
fix(audio): guard against zero-length clips
chore: bump numpy pin, tidy .gitignore
docs: document --params-file schema
```

Use `!` after the type (or a `BREAKING CHANGE:` footer) for changes that break
existing CLI flags/behavior, e.g. `feat(video)!: rename --clip-len to --frames`.

## Pull requests

- Open a PR from your topic branch into `master` using the PR template
  (`.github/PULL_REQUEST_TEMPLATE.md`) — fill it out completely.
- Link issues using GitHub closing keywords (`Closes #123`, `Fixes #123`) for
  issues the PR fully resolves (auto-closes on merge), or `Refs #123` for
  issues it only relates to.
- Set the PR's **Milestone** in the sidebar to match the roadmap milestone it
  contributes to (see `documentation/*_PIPELINE_ROADMAP.md`) or a release
  milestone, and note it in the PR body.
- Modality/area **labels** (`image`, `audio`, `video`, `pipeline`, `docs`,
  `infra`) are auto-applied by `.github/workflows/labeler.yml` based on changed
  file paths (config in `.github/labeler.yml`) — double check they landed, and
  add a type label (`bug`/`enhancement`/`chore`) manually since that isn't
  path-derivable.
- Keep PRs scoped to one modality/concern when possible.
- **Enforcement is via the PR template checklist + human review** — there is no
  automated commit-linter. Reviewers should block merge if branch/commit
  conventions aren't followed.
- At least **1 approving review** is required before merge.
- The lightweight CI check (`.github/workflows/ci.yml`) must pass.
- Squash-merge (or rebase-merge) is preferred to keep `master` history readable;
  the squashed commit message should itself follow Conventional Commits.

## Branch protection (applied on GitHub, `master` branch)

Repo admins should configure, under Settings → Branches → Branch protection rules
for `master`:

- [x] Require a pull request before merging
- [x] Require approvals: **1**
- [x] Require status checks to pass before merging → select the CI smoke/lint check
- [x] Require branches to be up to date before merging
- [x] Do not allow bypassing the above settings (include administrators)
- [x] Block force pushes
- [x] Block branch deletion

## Issues & milestones

- Use the issue templates under `.github/ISSUE_TEMPLATE/` (bug report / feature request).
- Label issues by modality (`image`, `audio`, `video`, `pipeline`, `docs`, `infra`)
  and type (`bug`, `enhancement`, `chore`).
- Group related work into milestones per release/benchmark cycle (e.g. `v0.2 - audio quality`).

## Testing expectations before opening a PR

Run the smoke test(s) relevant to what you changed:

```zsh
python tests/smoke/smoke_test_local.py
python tests/smoke/smoke_test_image_lossy_local.py
python tests/smoke/smoke_test_upscaler_local.py
python tests/smoke/smoke_test_audio_local.py
python tests/smoke/smoke_test_video_local.py
python tests/smoke/smoke_test_random_search.py
python tests/smoke/smoke_test_full_production_pipeline.py
python tests/smoke/smoke_test_prepare_audio_dataset.py
python tests/smoke/smoke_test_vortex_pipeline.py
```

CI does **not** run these (they require local datasets that aren't checked into
git); CI runs a fast syntax/import check plus a no-data dry-run smoke job (random
search `--dry-run`, `--help` sanity on every CLI entry point, and the vortex
pipeline smoke test). Manual smoke-test runs + reviewer sign-off remain the real
quality gate for data-dependent behavior.

## CI/CD

- `.github/workflows/ci.yml` — runs on every PR/push to `master`:
  - `lint-and-compile`: byte-compiles all Python sources + non-blocking `ruff` lint.
  - `smoke-dry-run`: installs `documentation/requirements.txt`, runs `--help` on
    every trainer/pipeline/reporting CLI entry point (import + argparse sanity,
    no data/training), plus the random-search dry-run and vortex pipeline smoke
    tests (both data-free).
- `.github/workflows/release.yml` — runs on `v*` tags: packages `README.md`,
  `AGENT.md`, `CONTRIBUTING.md`, and `documentation/` into a docs bundle and cuts
  a GitHub Release with auto-generated notes. There is no cloud training/deploy
  target for this repo (see `AGENT.md`), so this is the extent of "CD" here.

## Repo hygiene

- Never commit `data/`, `models/`, `__pycache__/`, `.DS_Store`, `.idea/`, or model
  weight files (`*.h5`, `*.keras`, `*.tflite`) — these are gitignored.
- Keep generated reports/plots out of version control unless explicitly intended
  as documentation assets (e.g. images referenced from `README.md`).

See also: `AGENT.md` for AI-agent-specific working context.

