# Reporting, Inference & Visual Comparisons (`src/reporting/`)

This directory handles structured Markdown/JSON reporting, master benchmark aggregation, training curve plotting, learned/bicubic upscaling, and side-by-side visual comparisons.

## At a Glance

| File | Role | Key Inputs / Outputs |
|---|---|---|
| `report_markdown.py` | Report I/O Helper | `write_markdown_json_report` / `read_markdown_json_report` — embeds machine-readable JSON inside human-readable `.md` reports |
| `generate_master_report.py` | Master Benchmark Aggregator | Scans run reports, benchmarks JPEG/WebP baselines on real data, plots audio SNR frontier -> writes `models/reports/MASTER_RUN_REPORT.md` |
| `upscale_reconstructed_images.py` | Image Reconstruction & Upscaler | Runs autoencoder inference and upscales via `--upscaler-mode {bicubic,learned}` with tiled overlap blending (`apply_learned_upscaler_tiled`) |
| `generate_real_image_comparisons.py` | Image Comparison Panels | Generates side-by-side original vs. reconstructed PNG grids on real images |
| `generate_real_av_comparisons.py` | Audio/Video Comparison Panels | Generates side-by-side waveform/spectrogram and video frame comparison PNGs |
| `plot_training_metrics_from_report.py` | Post-Hoc Metric Plotter | Reads any `*.md` evaluation report and regenerates loss/PSNR/SSIM/SNR training curves |

---

## Common Commands

```zsh
# Regenerate the master benchmark report & audio frontier plot
python src/reporting/generate_master_report.py

# Upscale reconstructed images using a trained learned super-resolution model
python src/reporting/upscale_reconstructed_images.py \
  --model models/local_run/<run>/best_model.keras \
  --upscaler-mode learned \
  --learned-upscaler-model models/upscaler_local_run/<run>/best_model.keras \
  --data-dir data/ImageData

# Generate real audio/video comparison panels
python src/reporting/generate_real_av_comparisons.py \
  --audio-dir data/AudioData/ESC-50-master/audio \
  --video-dir "data/VIDEO DATA"
```
