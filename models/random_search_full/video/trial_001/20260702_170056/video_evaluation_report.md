# Video Evaluation Report

- Source: `video_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T14:09:14.906221Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_001",
    "preset": "custom",
    "frames": 6,
    "height": 64,
    "width": 64,
    "fps": 10,
    "latent_dim": 160,
    "batch_size": 2,
    "epochs": 8,
    "lr": 9.34e-05,
    "seed": 42,
    "data_dir": "data/VIDEO DATA",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_max_videos": 0,
    "real_max_clips": 1800,
    "real_clip_stride": 4,
    "synthetic_only": true,
    "train_samples": 400,
    "val_samples": 100,
    "test_samples": 100,
    "target_psnr": 22.0,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp4_benchmark": false
  },
  "history": {
    "loss": [
      0.05765943229198456,
      0.03288022056221962
    ],
    "mse": [
      0.021609105169773102,
      0.007924903184175491
    ],
    "psnr_metric": [
      17.871076583862305,
      22.432106018066406
    ],
    "ssim_metric": [
      0.761329710483551,
      0.8300576210021973
    ],
    "val_loss": [
      0.044944677501916885,
      0.024677397683262825
    ],
    "val_mse": [
      0.013714509084820747,
      0.003948165103793144
    ],
    "val_psnr_metric": [
      19.811246871948242,
      24.769977569580078
    ],
    "val_ssim_metric": [
      0.7937556505203247,
      0.8480408191680908
    ],
    "learning_rate": [
      9.339999814983457e-05,
      9.339999814983457e-05
    ]
  },
  "test_metrics": {
    "loss": 0.025666935369372368,
    "mse": 0.004401511047035456,
    "psnr_metric": 24.394493103027344,
    "ssim_metric": 0.8452472686767578
  },
  "preview": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_001/20260702_170056/video_preview_reconstruction.png",
  "benchmark": null
}
```
