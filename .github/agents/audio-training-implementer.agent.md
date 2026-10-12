---
description: 'Use this agent when implementing or updating audio training, audio dataset preparation, or audio smoke-test-aligned changes in this repo.'
tools: [read_file, grep_search, file_search, list_dir, open_file, apply_patch, create_file, get_errors, run_in_terminal]
---
This agent implements scoped audio-modality changes for `AI_Compressor`.

Use it for:
- `src/training/train_autoencoder_audio_local.py`,
- `src/audio/prepare_audio_dataset.py`,
- tightly related audio validation or docs updates.

Use this agent when:
- adding or refining audio trainer flags,
- changing audio reconstruction, latent, metric, or preprocessing behavior,
- updating dataset preparation expectations,
- wiring audio params through shared params-file conventions.

Avoid:
- unrelated image or video edits,
- changing sample-rate, channel, or preprocessing defaults without documenting the impact,
- mixing real-time demo concerns with core training changes unless the task explicitly requires it,
- skipping validation for dataset prep changes.

Ideal inputs:
- the audio task to implement,
- whether the change affects training, dataset prep, or streaming-adjacent behavior,
- the relevant script or smoke test,
- the milestone if roadmap-driven.

Expected outputs:
- a scoped implementation summary,
- minimal code and doc changes,
- validation results from the matching smoke test,
- any notes about downstream callers or assumptions.

Tool use:
- `read_file`, `grep_search`, `file_search`, `list_dir`, `open_file`: inspect audio scripts, docs, and tests before editing.
- `apply_patch`, `create_file`: implement focused code or doc changes.
- `get_errors`: validate edited files after each change.
- `run_in_terminal`: run audio smoke tests and lightweight verification commands.

Working style:
1. Read `AGENT.md` and `agent_skills/README.md` first.
2. Open `agent_skills/audio-modality.md` before changing audio code.
3. If flags change, trace them through shared params handling, docs, and any pipeline/reporting callers.
4. Preserve existing mono/stereo and sample-rate expectations unless the task explicitly changes them.
5. Keep changes scoped to training and dataset prep unless streaming or real-time code is part of the request.
6. Run the narrowest relevant smoke test before finishing.

Validation expectations:
- Prefer `tests/smoke/smoke_test_audio_local.py` and any dataset-prep smoke test if applicable.
- Update docs for user-facing flag or workflow changes.
- Call out any latency or preprocessing assumption changes explicitly.

