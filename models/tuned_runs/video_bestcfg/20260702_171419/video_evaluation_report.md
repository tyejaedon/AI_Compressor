# Video Evaluation Report

- Source: `video_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T14:16:41.228183Z",
  "config": {
    "output_root": "models/tuned_runs/video_bestcfg",
    "preset": "custom",
    "frames": 6,
    "height": 64,
    "width": 64,
    "fps": 10,
    "latent_dim": 160,
    "batch_size": 2,
    "epochs": 2,
    "lr": 9.34e-05,
    "seed": 42,
    "data_dir": "data/VIDEO DATA",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_max_videos": 0,
    "real_max_clips": 1800,
    "real_clip_stride": 4,
    "synthetic_only": true,
    "train_samples": 120,
    "val_samples": 40,
    "test_samples": 40,
    "target_psnr": 22.0,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp4_benchmark": false
  },
  "history": {
    "loss": [
      0.07393841445446014,
      0.05148531123995781
    ],
    "mse": [
      0.03152454271912575,
      0.017547432333230972
    ],
    "psnr_metric": [
      15.525103569030762,
      18.338787078857422
    ],
    "ssim_metric": [
      0.7172447443008423,
      0.7752460241317749
    ],
    "val_loss": [
      0.06499151140451431,
      0.05071123689413071
    ],
    "val_mse": [
      0.0253346748650074,
      0.01669352874159813
    ],
    "val_psnr_metric": [
      16.50589370727539,
      18.729877471923828
    ],
    "val_ssim_metric": [
      0.7360585927963257,
      0.7740952968597412
    ],
    "learning_rate": [
      9.339999814983457e-05,
      9.339999814983457e-05
    ]
  },
  "test_metrics": {
    "loss": 0.04931662231683731,
    "mse": 0.01698591187596321,
    "psnr_metric": 18.67655372619629,
    "ssim_metric": 0.7906102538108826
  },
  "preview": "models/tuned_runs/video_bestcfg/20260702_171419/video_preview_reconstruction.png",
  "benchmark": null
}
```
