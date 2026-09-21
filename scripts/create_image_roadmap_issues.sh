#!/usr/bin/env bash
# Creates GitHub milestones + issues for the Image Pipeline Roadmap
# (documentation/IMAGE_PIPELINE_ROADMAP.md).
#
# Prereqs:
#   1. brew install gh   (already installed if you're reading this after setup)
#   2. gh auth login     (interactive; run once)
#   3. Run this script from the repo root: ./scripts/create_image_roadmap_issues.sh
#
# Safe to re-run: milestones are looked up by title before creating; issues are
# always created new (gh has no built-in idempotent issue-create), so avoid
# re-running after issues already exist unless you want duplicates.

set -euo pipefail

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "Target repo: ${REPO}"

# Ensures a milestone with the given title exists (idempotent). Does NOT print
# anything to stdout other than status; issues reference milestones by TITLE
# (gh issue create --milestone expects a name, not a number).
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
for label in "image" "compression" "upscaling" "data-pipeline" "docs" "search"; do
  gh label create "${label}" --repo "${REPO}" --color "0366d6" --force >/dev/null 2>&1 || true
done

echo "== Milestone 1: Real Rate-Distortion Accounting =="
M1="M1: Real Rate-Distortion Accounting"
create_milestone "$M1" \
"Replace heuristic rate proxy with real measurable bitrate/compression accounting. See documentation/IMAGE_PIPELINE_ROADMAP.md#milestone-1"
create_issue "$M1" "[image/lossy] Add real bits-per-pixel measurement to evaluation report" \
"Compute empirical entropy of the quantized latent tensor per test batch and log \`latent_bits_per_pixel\` alongside existing PSNR/SSIM/rate-penalty metrics.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 1." "image,compression"
create_issue "$M1" "[image/lossy] Prototype a lightweight entropy coder for the latent tensor" \
"Add an opt-in \`--enable-entropy-coding\` flag (default off) that range/arithmetic-codes the quantized latent and reports actual compressed byte size for the test set.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 1." "image,compression"
create_issue "$M1" "[image/lossy] Replace RatePenalty magnitude heuristic with entropy-estimate-based rate loss" \
"Add \`--rate-loss-mode {magnitude,entropy}\` (default \`magnitude\` to preserve current benchmark defaults).

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 1." "image,compression"
create_issue "$M1" "[image/docs] Document real vs proxy compression metrics" \
"Update README.md / documentation/MODEL_PARAMS_INPUT.md to distinguish JPEG-baseline comparison from true latent bitrate.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 1." "image,docs"

echo "== Milestone 2: Quantization-Aware Training & Latent Quality =="
M2="M2: Quantization-Aware Training & Latent Quality"
create_milestone "$M2" \
"Close the train/deploy gap in lossy quantization; configurable bit-depth. See documentation/IMAGE_PIPELINE_ROADMAP.md#milestone-2"
create_issue "$M2" "[image/lossy] Make latent quantization bit-depth configurable" \
"Parameterize StraightThroughQuantize with --latent-bit-depth (4/6/8/10), default 8 to match current behavior.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 2." "image,compression"
create_issue "$M2" "[image/lossy] Add quantization noise annealing (QAT-style)" \
"Optional --quant-noise-anneal flag that fades simulated quantization noise from high to zero over training.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 2." "image,compression"
create_issue "$M2" "[image/lossy] Add random-search sweep profile for bit-depth vs PSNR/rate tradeoff" \
"Extend documentation/random_search_profile.json with a latent_bit_depth axis for the lossy modality.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 2." "image,search"
create_issue "$M2" "[image/lossy] Extend smoke test for bit-depth round-trip" \
"Add assertion to smokeTests/smoke_test_image_lossy_local.py that quantized latents read back losslessly at the chosen bit-depth.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 2." "image"

echo "== Milestone 3: Learned Upscaling =="
M3="M3: Learned Upscaling"
create_milestone "$M3" \
"Add optional learned super-resolution to replace/augment BICUBIC-only upscaling. See documentation/IMAGE_PIPELINE_ROADMAP.md#milestone-3"
create_issue "$M3" "[image/upscale] Design a minimal learned super-resolution head" \
"Small residual CNN trained on the same degrade/upscale-factor pairs already produced by train_autoencoder_image_local.py's data pipeline.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 3." "image,upscaling"
create_issue "$M3" "[image/upscale] Add train_upscaler_image_local.py trainer script" \
"Mirror existing trainer conventions (--preset, --params-file, smoke-testable), output its own evaluation_report.md.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 3." "image,upscaling"
create_issue "$M3" "[image/upscale] Wire --upscaler-mode learned into upscale_reconstructed_images.py" \
"Falls back to BICUBIC if no learned model path is given; --upscaler-mode bicubic stays default.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 3." "image,upscaling"
create_issue "$M3" "[image/upscale] Add tiling support for large-image inference" \
"Avoid OOM on M1 by tiling with overlap + blending at model input resolution.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 3." "image,upscaling"
create_issue "$M3" "[image/docs] Document the learned upscaling workflow end-to-end" \
"README section + smokeTests/smoke_test_upscaler_local.py.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 3." "image,docs"

echo "== Milestone 4: Data Pipeline Robustness & Throughput =="
M4="M4: Data Pipeline Robustness & Throughput"
create_milestone "$M4" \
"Resilient to messy real-world images; faster repeated local iteration. See documentation/IMAGE_PIPELINE_ROADMAP.md#milestone-4"
create_issue "$M4" "[image/data] Add structured logging for skipped/corrupt images" \
"Extend filter_paths_by_min_size (and add a decode-error guard) to record counts + sample reasons into split_info in the evaluation report.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 4." "image,data-pipeline"
create_issue "$M4" "[image/data] Add optional on-disk TFRecord/tensor cache" \
"--cache-dir flag keyed by (data-dir, block-size, split), default off, invalidated automatically via input hash.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 4." "image,data-pipeline"
create_issue "$M4" "[image/data] Add corrupted/degraded-input robustness smoke check" \
"Feed intentionally corrupted/blurred/noisy images through inference and assert no crash + bounded PSNR drop.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 4." "image,data-pipeline"
create_issue "$M4" "[image/data] Surface degradation noise/interp config fully" \
"Add --degrade-noise-std (default 0.015, preserving current behavior) alongside existing --degrade-interp.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 4." "image,data-pipeline"

echo "== Milestone 5: Reporting, Search & Docs Alignment =="
M5="M5: Reporting, Search & Docs Alignment"
create_milestone "$M5" \
"Keep random search, docs, and reports in sync with M1-M4. See documentation/IMAGE_PIPELINE_ROADMAP.md#milestone-5"
create_issue "$M5" "[image/search] Extend image + image_lossy search spaces for new flags" \
"Add bit-depth, rate-loss-mode, quant-noise-anneal, cache-dir toggles to random_search_hyperparams.py and documentation/random_search_profile.json.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 5." "image,search"
create_issue "$M5" "[image/docs] Full CLI flag audit for both image trainers" \
"Cross-check --help output vs documentation/MODEL_PARAMS_INPUT.md, fill any gaps.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 5." "image,docs"
create_issue "$M5" "[image/reporting] Add a compression-ratio/PSNR frontier plot" \
"Extend plot_training_metrics_from_report.py / generate_master_report.py to chart rate-distortion tradeoff across trials.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 5." "image"
create_issue "$M5" "[image/pipeline] Sync run_full_production_pipeline.py with renamed/added flags" \
"Verify end-to-end pipeline still runs after Milestones 1-4 land.

Ref: documentation/IMAGE_PIPELINE_ROADMAP.md, Milestone 5." "image"

echo "Done. Review created milestones/issues at: https://github.com/${REPO}/issues"

