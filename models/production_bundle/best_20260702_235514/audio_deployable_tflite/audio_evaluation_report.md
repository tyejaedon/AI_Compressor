# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-06-29T19:08:42.873944Z",
  "config": {
    "output_root": "models/audio_prod_psnr35",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "latent_dim": 384,
    "batch_size": 8,
    "epochs": 90,
    "lr": 5e-05,
    "seed": 42,
    "data_dir": "data/AudioData/ESC-50-master/audio",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_file_limit": 2000,
    "synthetic_only": false,
    "train_samples": 7000,
    "val_samples": 2000,
    "test_samples": 2000,
    "target_psnr": 35.0,
    "shuffle_buffer": 1024,
    "export_tflite": true,
    "tflite_fp16": true,
    "run_mp3_benchmark": true,
    "synthetic_profile": "expansive"
  },
  "history": {
    "loss": [
      0.048775564879179,
      0.02306978963315487,
      0.018708042800426483,
      0.017088167369365692,
      0.01572561077773571
    ],
    "mse": [
      0.007101740222424269,
      0.0016786548076197505,
      0.001361269038170576,
      0.0011578035773709416,
      0.0009421175927855074
    ],
    "psnr_metric": [
      24.95589256286621,
      30.706398010253906,
      32.007537841796875,
      32.937171936035156,
      33.56520080566406
    ],
    "snr_db_metric": [
      5.802126884460449,
      11.49828052520752,
      12.641432762145996,
      13.58130168914795,
      14.271530151367188
    ],
    "val_loss": [
      0.020797982811927795,
      0.015487374737858772,
      0.014307338744401932,
      0.01326596550643444,
      0.012130621820688248
    ],
    "val_mse": [
      0.0017007520655170083,
      0.0012479157885536551,
      0.001090447069145739,
      0.0009521148167550564,
      0.0008255637367255986
    ],
    "val_psnr_metric": [
      31.153095245361328,
      33.175315856933594,
      33.776363372802734,
      34.49602508544922,
      35.1430778503418
    ],
    "val_snr_db_metric": [
      11.003419876098633,
      13.025569915771484,
      13.626601219177246,
      14.346219062805176,
      14.9932222366333
    ],
    "learning_rate": [
      4.999999873689376e-05,
      4.999999873689376e-05,
      4.999999873689376e-05,
      4.999999873689376e-05,
      4.999999873689376e-05
    ]
  },
  "test_metrics": {
    "loss": 0.021836860105395317,
    "mse": 0.003064024029299617,
    "psnr_metric": 31.4471435546875,
    "snr_db_metric": 12.62895393371582
  },
  "preview_plot": "models/audio_prod_psnr35/20260629_165547/audio_preview_waveform.png",
  "preview_wav": "models/audio_prod_psnr35/20260629_165547/audio_preview_reconstruction.wav",
  "metrics_plot": "models/audio_prod_psnr35/20260629_165547/audio_training_metrics.png",
  "benchmark": "models/audio_prod_psnr35/20260629_165547/audio_mp3_benchmark.json"
}
```
