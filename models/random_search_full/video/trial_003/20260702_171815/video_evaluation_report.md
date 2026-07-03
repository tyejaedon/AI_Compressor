# Video Evaluation Report

- Source: `video_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T14:24:45.036600Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_003",
    "preset": "custom",
    "frames": 8,
    "height": 64,
    "width": 80,
    "fps": 10,
    "latent_dim": 160,
    "batch_size": 4,
    "epochs": 4,
    "lr": 0.0001899,
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
      0.05508089438080788,
      0.029705967754125595
    ],
    "mse": [
      0.020229782909154892,
      0.006351300980895758
    ],
    "psnr_metric": [
      18.178007125854492,
      23.195486068725586
    ],
    "ssim_metric": [
      0.7696530222892761,
      0.8379508852958679
    ],
    "val_loss": [
      0.03817419707775116,
      0.02096519246697426
    ],
    "val_mse": [
      0.010577495209872723,
      0.0026204055175185204
    ],
    "val_psnr_metric": [
      20.842880249023438,
      26.402875900268555
    ],
    "val_ssim_metric": [
      0.8187354207038879,
      0.8609145283699036
    ],
    "learning_rate": [
      0.00018990000535268337,
      0.00018990000535268337
    ]
  },
  "test_metrics": {
    "loss": 0.02180296927690506,
    "mse": 0.0028792789671570063,
    "psnr_metric": 26.0582275390625,
    "ssim_metric": 0.8567911386489868
  },
  "preview": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/video/trial_003/20260702_171815/video_preview_reconstruction.png",
  "benchmark": null
}
```
