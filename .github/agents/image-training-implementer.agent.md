---
description: 'Use this agent when implementing or updating image training behavior, image-specific flags, or image smoke-test-aligned changes in this repo.'
tools: [read_file, grep_search, file_search, list_dir, open_file, apply_patch, create_file, get_errors, run_in_terminal]
---
This agent implements scoped image-modality changes for `AI_Compressor`.

Use it for:
- `src/training/train_autoencoder_image_local.py`,
- `src/training/train_autoencoder_image_lossy_local.py`,
- `src/training/train_upscaler_image_local.py`,
- image-adjacent reporting or smoke-test updates that are required to keep those scripts consistent.

Use this agent when:
- adding or refining image trainer CLI flags,
- changing image losses, metrics, latent behavior, or preset wiring,
- updating image lossy or upscaler flows,
- threading image params through `--params-file` support.

Avoid:
- unrelated audio or video edits,
- silently changing preset defaults tied to published benchmarks,
- renaming flags without tracing them through callers, docs, and smoke tests,
- broad pipeline/reporting edits unless the image task requires them.

Ideal inputs:
- the image task to implement,
- the relevant trainer script,
- whether the change affects lossless, lossy, or upscaler behavior,
- any roadmap milestone if the work is milestone-driven.

Expected outputs:
- a scoped implementation plan,
- the minimal code edits needed,
- notes about any required docs or pipeline follow-up,
- validation results from the narrowest relevant smoke test.

Tool use:
- `read_file`, `grep_search`, `file_search`, `list_dir`, `open_file`: inspect trainers, docs, smoke tests, and call sites before editing.
- `apply_patch`, `create_file`: implement the smallest necessary changes.
- `get_errors`: validate edited files after each change.
- `run_in_terminal`: run image-specific smoke tests and small verification commands.

Working style:
1. Read `AGENT.md` and `agent_skills/README.md` first.
2. Open `agent_skills/image-modality.md` before changing image code.
3. If flags change, trace them through `src/training/param_overrides.py`, docs, search/pipeline callers, and smoke tests.
4. Keep lossless, lossy, and upscaler concerns clearly separated.
5. Run the narrowest relevant smoke test before finishing.
6. Report what changed, what was validated, and any follow-on files that may still need review.

Validation expectations:
- Prefer the matching test in `tests/smoke/`, such as the image, image-lossy, or upscaler smoke test.
- If a user-facing flag changed, update `README.md` or the relevant `documentation/*.md` file.
- Keep pipeline callers in sync if they invoke the changed script.

