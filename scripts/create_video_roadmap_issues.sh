#!/usr/bin/env bash
# Creates GitHub milestones + issues for the Video Pipeline Roadmap
# (documentation/VIDEO_PIPELINE_ROADMAP.md).
#
# Prereqs:
#   1. gh auth login     (interactive; run once)
#   2. Run from repo root: ./scripts/create_video_roadmap_issues.sh
#
# Safe to re-run: milestones are looked up by title before creating; issues are
# always created new, so avoid re-running after issues already exist unless you
# want duplicates.

set -euo pipefail

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "Target repo: ${REPO}"

# Ensures a milestone with the given title exists (idempotent). Only logs to
# stderr; issues reference milestones by TITLE (gh issue create --milestone
# expects a name, not a number).
create_milestone() {
  local title="$1"
  local description="$2"
  local existing
  existing="$(gh api "repos/${REPO}/milestones?state=all" --jq ".[] | select(.title==\"${title}\") | .number" || true)"
  if [[ -n "${existing}" ]]; then
    echo "Milestone exists: ${title} (#${existing})" >&2
    return
  fi
  local number
  number="$(gh api "repos/${REPO}/milestones" -f title="${title}" -f description="${description}" --jq .number)"
  echo "Created milestone: ${title} (#${number})" >&2
}

create_issue() {
  local milestone_title="$1"
  local title="$2"
  local body="$3"
  local labels="$4"
  gh issue create \
    --repo "${REPO}" \
    --title "${title}" \
    --body "${body}" \
    --label "${labels}" \
    --milestone "${milestone_title}" \
    >/dev/null
  echo "  + issue: ${title}"
}

# --- Ensure labels exist (idempotent) ---
for label in "video" "compression" "temporal" "data-pipeline" "docs" "search" "loss" "benchmark"; do
  gh label create "${label}" --repo "${REPO}" --color "b60205" --force >/dev/null 2>&1 || true
done

echo "== Milestone V1: Real Compression Accounting & Quantization =="
V1="V1: Real Compression Accounting & Quantization"
create_milestone "$V1" \
"Replace theoretical bitrate estimate with real measured video latent bitrate/quantization. See documentation/VIDEO_PIPELINE_ROADMAP.md#milestone-v1"
create_issue "$V1" "[video/compression] Add real bit-depth quantization of the video latent vector" \
"--latent-bit-depth opt-in flag, default preserves float32 behavior.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V1." "video,compression"
create_issue "$V1" "[video/compression] Add straight-through estimator for quantized latent gradients (video)" \
"Train-time awareness of the quantization step.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V1." "video,compression"
create_issue "$V1" "[video/compression] Add real vs theoretical compression metrics to the video evaluation report" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V1." "video,compression"
create_issue "$V1" "[video/docs] Document real vs theoretical video compression accounting" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V1." "video,docs"

echo "== Milestone V2: Temporal Redundancy & Motion-Aware Compression =="
V2="V2: Temporal Redundancy & Motion-Aware Compression"
create_milestone "$V2" \
"Start exploiting inter-frame redundancy instead of treating each clip as an independent frame stack. See documentation/VIDEO_PIPELINE_ROADMAP.md#milestone-v2"
create_issue "$V2" "[video/temporal] Prototype a frame-delta/residual coding mode" \
"Encode first frame directly, subsequent frames as residuals against a (learned or simple) prediction of the previous frame; --temporal-mode {independent,residual}, default independent.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V2." "video,temporal"
create_issue "$V2" "[video/temporal] Add a lightweight learned inter-frame predictor (e.g. ConvLSTM or simple warping head)" \
"Optional component feeding the residual-coding mode above.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V2." "video,temporal"
create_issue "$V2" "[video/temporal] Add PSNR/bitrate comparison between independent vs residual temporal modes to the evaluation report" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V2." "video,temporal"
create_issue "$V2" "[video/docs] Document the temporal-mode tradeoffs and current limitations (no true motion compensation/optical flow yet)" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V2." "video,docs"

echo "== Milestone V3: Data Pipeline Robustness & Throughput =="
V3="V3: Data Pipeline Robustness & Throughput"
create_milestone "$V3" \
"Make video decoding resilient and faster to iterate with locally. See documentation/VIDEO_PIPELINE_ROADMAP.md#milestone-v3"
create_issue "$V3" "[video/data] Add structured logging for skipped/corrupt/unreadable video files" \
"Record counts + sample reasons into split_info in the evaluation report instead of silent skip.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V3." "video,data-pipeline"
create_issue "$V3" "[video/data] Parallelize video decoding across files" \
"Use a process/thread pool for the per-video ffmpeg subprocess calls, bounded by --decode-workers (default 1, preserving current behavior).

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V3." "video,data-pipeline"
create_issue "$V3" "[video/data] Add optional on-disk clip cache" \
"--cache-dir flag keyed by (data-dir, frames, height, width, fps, stride), default off, hash-invalidated.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V3." "video,data-pipeline"
create_issue "$V3" "[video/data] Add dynamic-resolution / non-divisible-by-8 input handling (auto-pad or auto-crop)" \
"Currently a hard raise if height/width aren't divisible by 8.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V3." "video,data-pipeline"

echo "== Milestone V4: Loss Configurability & Modern Codec Benchmarking =="
V4="V4: Loss Configurability & Modern Codec Benchmarking"
create_milestone "$V4" \
"Bring video's loss configurability to parity with the image trainer, and broaden the benchmark. See documentation/VIDEO_PIPELINE_ROADMAP.md#milestone-v4"
create_issue "$V4" "[video/loss] Introduce --loss-mse-weight/--loss-l1-weight/--loss-ssim-weight CLI flags and --loss-profile presets" \
"balanced profile matches current hardcoded 0.75/0.15/0.10 exactly.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V4." "video,loss"
create_issue "$V4" "[video/eval] Add an edge/motion-aware loss term option" \
"Penalize frame-to-frame gradient mismatches, complementing Milestone V2's temporal work.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V4." "video,loss"
create_issue "$V4" "[video/benchmark] Add an optional VP9 or AV1 comparison to the MP4 benchmark" \
"--benchmark-codecs h264,vp9, default h264 only (current behavior).

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V4." "video,benchmark"
create_issue "$V4" "[video/docs] Document the new loss/benchmark flags in README and MODEL_PARAMS_INPUT.md" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V4." "video,docs"

echo "== Milestone V5: Reporting, Search & Docs Alignment =="
V5="V5: Reporting, Search & Docs Alignment"
create_milestone "$V5" \
"Keep random search, docs, and reports in sync with V1-V4. See documentation/VIDEO_PIPELINE_ROADMAP.md#milestone-v5"
create_issue "$V5" "[video/search] Extend video search space for new flags" \
"bit-depth, temporal-mode, loss-profile added to src/training/random_search_hyperparams.py and documentation/random_search_profile.json.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V5." "video,search"
create_issue "$V5" "[video/docs] Full CLI flag audit for the video trainer" \
"Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V5." "video,docs"
create_issue "$V5" "[video/reporting] Add a compression-ratio/PSNR frontier plot for video trials" \
"Extend src/reporting/plot_training_metrics_from_report.py / src/reporting/generate_master_report.py.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V5." "video"
create_issue "$V5" "[video/pipeline] Sync src/pipeline/run_full_production_pipeline.py with any renamed/added video flags" \
"Verify end-to-end pipeline still runs after Milestones V1-V4 land.

Ref: documentation/VIDEO_PIPELINE_ROADMAP.md, Milestone V5." "video"

echo "Done. Review created milestones/issues at: https://github.com/${REPO}/issues"

