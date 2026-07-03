# Video Evaluation Report

- Source: `video_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T19:21:39.050474Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/video/trial_001",
    "preset": "custom",
    "frames": 4,
    "height": 32,
    "width": 32,
    "fps": 10,
    "latent_dim": 192,
    "batch_size": 2,
    "epochs": 16,
    "lr": 5.93e-05,
    "seed": 42,
    "data_dir": "data/VIDEO DATA",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_max_videos": 0,
    "real_max_clips": 1800,
    "real_clip_stride": 4,
    "synthetic_only": true,
    "train_samples": 2400,
    "val_samples": 600,
    "test_samples": 600,
    "target_psnr": 30.0,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp4_benchmark": false
  },
  "history": {
    "loss": [
      0.03764529153704643,
      0.018381105735898018,
      0.01565183699131012,
      0.01416503544896841
    ],
    "mse": [
      0.009822864085435867,
      0.0020278017036616802,
      0.001436030026525259,
      0.0011305235093459487
    ],
    "psnr_metric": [
      22.320505142211914,
      27.544416427612305,
      29.04241943359375,
      30.094511032104492
    ],
    "ssim_metric": [
      0.7970670461654663,
      0.871061384677887,
      0.8869601488113403,
      0.8957766890525818
    ],
    "val_loss": [
      0.020535074174404144,
      0.01691119372844696,
      0.015250041149556637,
      0.013641917146742344
    ],
    "val_mse": [
      0.002512152772396803,
      0.001709025469608605,
      0.0013342053862288594,
      0.0010224004508927464
    ],
    "val_psnr_metric": [
      26.579200744628906,
      28.181272506713867,
      29.29558563232422,
      30.538339614868164
    ],
    "val_ssim_metric": [
      0.8566475510597229,
      0.8807950019836426,
      0.889975368976593,
      0.8985618948936462
    ],
    "learning_rate": [
      5.9300000430084765e-05,
      5.9300000430084765e-05,
      5.9300000430084765e-05,
      5.9300000430084765e-05
    ]
  },
  "test_metrics": {
    "loss": 0.013682223856449127,
    "mse": 0.0010487115941941738,
    "psnr_metric": 30.423437118530273,
    "ssim_metric": 0.898495614528656
  },
  "preview": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/video/trial_001/20260702_183130/video_preview_reconstruction.png",
  "benchmark": null
}
```
