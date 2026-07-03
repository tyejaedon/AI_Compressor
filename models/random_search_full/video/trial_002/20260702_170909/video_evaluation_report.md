# Video Evaluation Report

- Source: `video_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T14:18:11.651294Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_002",
    "preset": "custom",
    "frames": 6,
    "height": 64,
    "width": 64,
    "fps": 10,
    "latent_dim": 160,
    "batch_size": 4,
    "epochs": 6,
    "lr": 6.31e-05,
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
      0.0689653605222702,
      0.04643198475241661,
      0.03465630114078522
    ],
    "mse": [
      0.0284421443939209,
      0.01510793250054121,
      0.008704670704901218
    ],
    "psnr_metric": [
      16.29634666442871,
      19.507604598999023,
      21.871505737304688
    ],
    "ssim_metric": [
      0.72999107837677,
      0.7948705554008484,
      0.8240599036216736
    ],
    "val_loss": [
      0.05415651202201843,
      0.039659421890974045,
      0.030583927407860756
    ],
    "val_mse": [
      0.018923059105873108,
      0.01127687469124794,
      0.006335392128676176
    ],
    "val_psnr_metric": [
      18.4277286529541,
      20.883718490600586,
      23.030691146850586
    ],
    "val_ssim_metric": [
      0.7674807906150818,
      0.8109303116798401,
      0.8298280239105225
    ],
    "learning_rate": [
      6.310000026132911e-05,
      6.310000026132911e-05,
      6.310000026132911e-05
    ]
  },
  "test_metrics": {
    "loss": 0.03080022893846035,
    "mse": 0.006608973257243633,
    "psnr_metric": 22.73647689819336,
    "ssim_metric": 0.8326852917671204
  },
  "preview": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_002/20260702_170909/video_preview_reconstruction.png",
  "benchmark": null
}
```
