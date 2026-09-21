#!/usr/bin/env bash
# Creates GitHub milestones + issues for the Audio Pipeline Roadmap
# (documentation/AUDIO_PIPELINE_ROADMAP.md).
#
# Prereqs:
#   1. gh auth login     (interactive; run once)
#   2. Run from repo root: ./scripts/create_audio_roadmap_issues.sh
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
for label in "audio" "compression" "streaming" "data-pipeline" "docs" "search" "loss"; do
  gh label create "${label}" --repo "${REPO}" --color "5319e7" --force >/dev/null 2>&1 || true
done

echo "== Milestone A1: Real Compression Accounting & Quantization =="
A1="A1: Real Compression Accounting & Quantization"
create_milestone "$A1" \
"Replace theoretical bitrate estimate with real measured audio latent bitrate/quantization. See documentation/AUDIO_PIPELINE_ROADMAP.md#milestone-a1"
create_issue "$A1" "[audio/compression] Add real bit-depth quantization of the latent vector" \
"--latent-bit-depth (default disabled/float32) that actually rounds/clips the bottleneck output to N bits before decode.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A1." "audio,compression"
create_issue "$A1" "[audio/compression] Add straight-through estimator for quantized latent gradients" \
"Train the model with the quantization step in the loop, not just evaluated post-hoc.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A1." "audio,compression"
create_issue "$A1" "[audio/compression] Add real vs theoretical compression metrics to evaluation report" \
"Log both estimated_input_to_latent_ratio (existing) and actual_latent_bytes_per_clip / actual_compression_ratio side by side.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A1." "audio,compression"
create_issue "$A1" "[audio/docs] Document real vs theoretical audio compression accounting" \
"Update README.md / documentation/MODEL_PARAMS_INPUT.md.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A1." "audio,docs"

echo "== Milestone A2: Real-Time Streaming Compression Path =="
A2="A2: Real-Time Streaming Compression Path"
create_milestone "$A2" \
"Give the trained autoencoder an actual streaming/chunked inference path, distinct from the vortex feature-extraction demo. See documentation/AUDIO_PIPELINE_ROADMAP.md#milestone-a2"
create_issue "$A2" "[audio/streaming] Design a chunked/overlap-add inference wrapper for the trained autoencoder" \
"Reuse existing fixed clip_len constraint; document latency implications.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A2." "audio,streaming"
create_issue "$A2" "[audio/streaming] Add realtime_audio_compression_pipeline.py" \
"Mirrors realtime_audio_vortex_pipeline.py's streaming buffer conventions but decodes/encodes through the trained model instead of extracting FFT features.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A2." "audio,streaming"
create_issue "$A2" "[audio/streaming] Add a smoke test for streaming inference" \
"smokeTests/smoke_test_realtime_audio_compression.py: tiny model + short synthetic stream, assert output shape/latency bounds.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A2." "audio,streaming"
create_issue "$A2" "[audio/docs] Document the real-time compression path and its relationship to the vortex demo" \
"Clarify in README that the vortex pipeline is visualization-only.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A2." "audio,docs"

echo "== Milestone A3: Perceptual Quality & Psychoacoustic Loss =="
A3="A3: Perceptual Quality & Psychoacoustic Loss"
create_milestone "$A3" \
"Move the loss function beyond MSE/L1/STFT-magnitude toward perceptually-weighted objectives. See documentation/AUDIO_PIPELINE_ROADMAP.md#milestone-a3"
create_issue "$A3" "[audio/loss] Add a mel-scale/log-mel spectral loss option" \
"--loss-mel-weight, computed via tf.signal.linear_to_mel_weight_matrix.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A3." "audio,loss"
create_issue "$A3" "[audio/loss] Add an A-weighting (or simplified perceptual weighting) term" \
"Approximate psychoacoustic frequency sensitivity curve applied to the STFT magnitude loss.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A3." "audio,loss"
create_issue "$A3" "[audio/loss] Introduce --loss-profile {balanced,snr,perceptual} presets" \
"Mirrors the image trainer's --loss-profile convention; balanced matches current defaults exactly.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A3." "audio,loss"
create_issue "$A3" "[audio/eval] Add optional PESQ/STOI-style perceptual metric to evaluation report" \
"Behind an opt-in flag if a suitable pure-Python/TF implementation is available; otherwise document as a future dependency decision.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A3." "audio"

echo "== Milestone A4: Data Pipeline Robustness (Multi-channel, Rate, Noise) =="
A4="A4: Data Pipeline Robustness (Multi-channel, Rate, Noise)"
create_milestone "$A4" \
"Remove mono-only/fixed-sample-rate/no-augmentation assumptions. See documentation/AUDIO_PIPELINE_ROADMAP.md#milestone-a4"
create_issue "$A4" "[audio/data] Make stereo downmixing explicit and opt-in, or add a --channels mode" \
"Decide and document the intended scope; if stereo is out of scope, add a clear NotImplementedError/docstring instead of silent averaging.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A4." "audio,data-pipeline"
create_issue "$A4" "[audio/data] Replace linear-interpolation resampling with a configurable method" \
"--resample-method {linear,polyphase}, default linear to preserve current behavior.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A4." "audio,data-pipeline"
create_issue "$A4" "[audio/data] Add optional additive noise augmentation during training" \
"--augment-noise-std, default 0 (off).

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A4." "audio,data-pipeline"
create_issue "$A4" "[audio/data] Add a corrupted/noisy-input robustness smoke check" \
"Feed a clipped/corrupted WAV through inference and assert graceful handling (no crash, bounded SNR drop) instead of silent fallback-to-silence.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A4." "audio,data-pipeline"

echo "== Milestone A5: Reporting, Search & Docs Alignment =="
A5="A5: Reporting, Search & Docs Alignment"
create_milestone "$A5" \
"Keep random search, docs, and reports in sync with A1-A4. See documentation/AUDIO_PIPELINE_ROADMAP.md#milestone-a5"
create_issue "$A5" "[audio/search] Extend audio search space for new flags" \
"bit-depth, loss-profile, resample-method added to random_search_hyperparams.py and documentation/random_search_profile.json.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A5." "audio,search"
create_issue "$A5" "[audio/docs] Full CLI flag audit for the audio trainer" \
"Cross-check --help output vs documentation/MODEL_PARAMS_INPUT.md.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A5." "audio,docs"
create_issue "$A5" "[audio/reporting] Add a compression-ratio/SNR frontier plot" \
"Extend plot_training_metrics_from_report.py / generate_master_report.py.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A5." "audio"
create_issue "$A5" "[audio/pipeline] Sync run_full_production_pipeline.py with any renamed/added audio flags" \
"Verify end-to-end pipeline still runs after Milestones A1-A4 land.

Ref: documentation/AUDIO_PIPELINE_ROADMAP.md, Milestone A5." "audio"

echo "Done. Review created milestones/issues at: https://github.com/${REPO}/issues"

