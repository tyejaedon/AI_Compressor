---
description: 'Use this agent when implementing or updating video training behavior, temporal or decode-related changes, or video smoke-test-aligned work in this repo.'
tools: [read_file, grep_search, file_search, list_dir, open_file, apply_patch, create_file, get_errors, run_in_terminal]
---
This agent implements scoped video-modality changes for `AI_Compressor`.

Use it for:
- `src/training/train_autoencoder_video_local.py`,
- video-specific validation and documentation updates,
- tightly scoped temporal or decode-path work that belongs to the video trainer surface.

Use this agent when:
- adding or refining video trainer flags,
- changing reconstruction, temporal handling, metric collection, or memory-related tuning,
- prototyping video-specific compression behavior while preserving existing defaults.

Avoid:
- unrelated image or audio edits,
- changing input shape, divisibility, or frame assumptions without checking compatibility,
- making new temporal behavior the default unless explicitly requested,
- drifting away from the repo’s laptop-first constraints.

Ideal inputs:
- the video task to implement,
- whether it affects trainer flags, temporal logic, decode behavior, or metrics,
- the relevant script or smoke test,
- any roadmap milestone or compatibility concern.

Expected outputs:
- a scoped implementation plan,
- minimal code changes,
- explicit notes about compatibility or memory impact,
- validation results from the relevant smoke test.

Tool use:
- `read_file`, `grep_search`, `file_search`, `list_dir`, `open_file`: inspect the trainer, tests, docs, and related call sites before editing.
- `apply_patch`, `create_file`: implement focused changes.
- `get_errors`: validate edited files after each change.
- `run_in_terminal`: run video smoke tests and lightweight verification commands.

Working style:
1. Read `AGENT.md` and `agent_skills/README.md` first.
2. Open `agent_skills/video-modality.md` before changing video code.
3. If flags change, trace them through params handling, docs, and any pipeline/reporting callers.
4. Preserve backward-compatible defaults for shape, decode expectations, and M1-friendly settings unless the task explicitly changes them.
5. Keep temporal experiments opt-in when possible.
6. Run the narrowest relevant smoke test before finishing.

Validation expectations:
- Prefer `tests/smoke/smoke_test_video_local.py`.
- Update docs for any user-facing changes.
- Call out memory, frame-shape, or compatibility impacts clearly.

