# Master Training & Compression Report

Generated: `2026-07-03T14:09:49.797683Z`

## Major Runs

| Modality | Run Tag | Report Path | PSNR | SSIM | MSE | Loss | SNR dB |
|---|---|---|---:|---:|---:|---:|---:|
| Image | synthetic_best | `models/local_run/20260626_005606/evaluation_report.md` | - | - | - | - | - |
| Image | real_prod | `models/local_real_run/20260629_101835/evaluation_report.md` | - | - | - | - | - |
| Image | tuned_fast | `models/local_run_tuned_fast/20260629_065112/evaluation_report.md` | - | - | - | - | - |
| Image | realdata_psnr30 | `models/image_realdata_psnr30/20260629_150224/evaluation_report.md` | 20.988 | 0.8488 | 0.008343 | 0.020870 | - |
| Image | realdata_psnr30_try2 | `models/image_realdata_psnr30_try2/20260629_150721/evaluation_report.md` | - | - | - | - | - |
| Audio | baseline | `models/audio_local_run/20260627_161410/audio_evaluation_report.md` | - | - | - | - | - |
| Audio | real_run_v2 | `models/audio_real_run_v2/20260629_125927/audio_evaluation_report.md` | - | - | - | - | - |
| Audio | realdata_opt_smoke | `models/audio_realdata_opt_smoke/20260629_145612/audio_evaluation_report.md` | 20.945 | - | 0.008565 | 0.070366 | 0.075 |
| Video | tuned_fast2 | `models/video_local_run_tuned_fast2/20260629_070559/video_evaluation_report.md` | - | - | - | - | - |
| Video | realdata_opt_smoke | `models/video_realdata_opt_smoke/20260629_145842/video_evaluation_report.md` | - | - | - | - | - |
| Bundle | production_bundle | `models/production_bundle/best_20260629_142034/metadata.json` | - | - | - | - | - |

## Production Bundle Selection

- No production bundle metadata found.

## JPEG / Other Lossy Image Baseline (Real Data)

Dataset sampled from: `data/ImageData/archive/data`

| Codec | Quality | Samples | Mean PSNR | Mean SSIM | Mean Size Ratio vs Raw RGB |
|---|---:|---:|---:|---:|---:|
| JPEG | 95 | 24 | 47.122 | 0.9838 | 0.0932 |
| JPEG | 85 | 24 | 42.656 | 0.9707 | 0.0641 |
| JPEG | 75 | 24 | 42.023 | 0.9580 | 0.0507 |
| JPEG | 50 | 24 | 41.574 | 0.9324 | 0.0362 |
| WEBP | 90 | 24 | 40.538 | 0.9747 | 0.0600 |
| WEBP | 75 | 24 | 36.134 | 0.9476 | 0.0335 |
| WEBP | 50 | 24 | 34.411 | 0.9296 | 0.0256 |

## Notes

- JPEG/WebP baselines are classic lossy codecs on full-resolution images.
- Model PSNR/SSIM values above come from run reports and may use resized model input dimensions.
- Audio MP3 benchmark in `audio_real_run_v2` is marked skipped due to ffmpeg unavailability at that run time.