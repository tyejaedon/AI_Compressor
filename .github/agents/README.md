# Custom Agents for `AI_Compressor`

This folder contains repo-scoped custom agents for routing, planning, and implementation work in `AI_Compressor`.

All agent files in this folder use a consistent frontmatter description pattern: `Use this agent when ...`.
If a task is unclear, start with `repo-task-router.agent.md` and hand off from there.

## Recommended order of use

### 1. Start here if the task is not yet scoped
- `repo-task-router.agent.md`
  - First stop for unclear, mixed-scope, or multi-surface requests.
  - Chooses the proper specialized agent before implementation begins.

### 2. Use this first for roadmap or milestone work
- `milestone-router.agent.md`
  - Turns roadmap items into a scoped implementation plan.
  - Identifies the likely files, docs, and smoke tests involved.

### 3. Use this for JetBrains/Copilot mode questions
- `jetbrains-agent-navigator.agent.md`
  - Explains Agent mode vs Inline Agent mode behavior.
  - Helps locate the right repo playbooks and confirms what is or is not likely exposed in the IDE UI.

### 4. Use these for implementation
- `image-training-implementer.agent.md`
  - Image, image-lossy, and upscaler trainer work.
- `audio-training-implementer.agent.md`
  - Audio training, dataset preparation, and audio-specific validation work.
- `video-training-implementer.agent.md`
  - Video training, temporal/decode changes, and video-specific validation work.
- `compression-models-implementer.agent.md`
  - Cross-modality architecture, losses, metrics, quantization, and compression-accounting changes.
- `pipeline-reporting-implementer.agent.md`
  - Random search, production pipeline, bundling, pruning, reporting, and cross-cutting flag-sync work.

## Quick routing guide

Use:
- `repo-task-router.agent.md` if you are not sure which agent should handle the task.
- `milestone-router.agent.md` if the task starts from `documentation/*_PIPELINE_ROADMAP.md`.
- `compression-models-implementer.agent.md` if the task changes compression behavior across one or more modalities.
- `pipeline-reporting-implementer.agent.md` if the task changes reports, plots, bundling, pruning, random search, or pipeline callers.
- A modality implementer if the task is clearly image, audio, or video specific.
- `jetbrains-agent-navigator.agent.md` if the question is about IDE behavior, Agent mode, Inline Agent mode, or whether specialized subagents should appear in the UI.

## Repo rules these agents are designed to follow

These agents are aligned with:
- `AGENT.md`
- `.github/copilot-instructions.md`
- `agent_skills/README.md`
- the matching modality or workflow playbook under `agent_skills/`

Across the lineup, the common expectations are:
- identify the modality and milestone first when relevant,
- trace changed flags end-to-end,
- keep docs in sync with user-facing behavior,
- preserve benchmark-linked defaults unless a task explicitly changes them,
- run the narrowest relevant smoke test before considering the work done.

## Suggested workflow

1. Start with `repo-task-router.agent.md` if the task is not already obvious.
2. If the request is roadmap-driven, route through `milestone-router.agent.md`.
3. Hand off to the appropriate implementation agent.
4. Run the relevant smoke test(s) for the touched area.
5. Update `README.md` or `documentation/*.md` if user-facing behavior changed.

