# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T14:00:45.734073Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_004",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "latent_dim": 256,
    "batch_size": 16,
    "epochs": 4,
    "lr": 0.0001481,
    "seed": 42,
    "data_dir": "data/AudioData/ESC-50-master/audio",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_file_limit": 0,
    "synthetic_only": true,
    "train_samples": 600,
    "val_samples": 150,
    "test_samples": 150,
    "target_psnr": 24.0,
    "shuffle_buffer": 1024,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp3_benchmark": false,
    "synthetic_profile": "expansive"
  },
  "history": {
    "loss": [
      0.11023541539907455,
      0.04324430599808693
    ],
    "mse": [
      0.02382967621088028,
      0.004393625073134899
    ],
    "psnr_metric": [
      17.332443237304688,
      23.853227615356445
    ],
    "snr_db_metric": [
      4.0575666427612305,
      10.49506664276123
    ],
    "val_loss": [
      0.05236401408910751,
      0.03290686756372452
    ],
    "val_mse": [
      0.005716869607567787,
      0.0026576693635433912
    ],
    "val_psnr_metric": [
      22.534486770629883,
      25.975276947021484
    ],
    "val_snr_db_metric": [
      9.29586410522461,
      12.645158767700195
    ],
    "learning_rate": [
      0.00014810000720899552,
      0.00014810000720899552
    ]
  },
  "test_metrics": {
    "loss": 0.03430343046784401,
    "mse": 0.0033451742492616177,
    "psnr_metric": 25.007104873657227,
    "snr_db_metric": 11.647649765014648
  },
  "preview_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_004/20260702_165733/audio_preview_waveform.png",
  "preview_wav": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_004/20260702_165733/audio_preview_reconstruction.wav",
  "metrics_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_004/20260702_165733/audio_training_metrics.png",
  "benchmark": null
}
```
