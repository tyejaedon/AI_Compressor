# Production Pipeline & Artifact Management (`src/pipeline/`)

This directory orchestrates end-to-end search -> training -> bundling -> pruning workflows.

## At a Glance

| File | Role | Destructive? | Key Behavior |
|---|---|---|---|
| `run_full_production_pipeline.py` | End-to-End Orchestrator | Only with `--apply-prune` | Runs random search -> sequential training (`image_standard`, `image_lossy`, `audio`, optional `--include-video`) -> bundles best models -> optional prune |
| `build_best_model_bundle.py` | Best-Run Collector | No | Scans `models/` reports, picks highest `test_metrics.psnr_metric` per modality, copies artifacts + writes `metadata.json` |
| `prune_keep_best_models.py` | Aggressive Pruner | Yes (deletes non-best runs) | Keeps only the single best run per category inside a bundle and deletes other report-backed run folders |
| `prune_models_keep_top2.py` | Top-N Pruner | Dry-run by default (`--apply` to delete) | Ranks runs by test PSNR, keeps top `--keep` (default `2`) runs per modality, strips redundant checkpoints |

---

## Usage Examples

### Full Production Pipeline
```zsh
# Search + train image_standard, image_lossy, and audio, then bundle
python src/pipeline/run_full_production_pipeline.py \
  --trials-image-standard 10 --trials-image-lossy 10 --trials-audio 8

# Include video and prune non-production runs afterward
python src/pipeline/run_full_production_pipeline.py \
  --trials-image-standard 10 --trials-image-lossy 10 --trials-audio 8 \
  --include-video --apply-prune
```

### Standalone Bundling & Pruning
```zsh
# Collect best runs into models/production_bundle/best_<timestamp>
python src/pipeline/build_best_model_bundle.py

# Preview top-2 pruning without deleting anything
python src/pipeline/prune_models_keep_top2.py --keep 2
```
