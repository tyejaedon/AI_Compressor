# Model Params Input

Use `--params-file` to load model/training parameters from JSON instead of passing many CLI flags.

Example file:

- `documentation/model_params_example.json`

The file can be:

1. Sectioned by modality/trainer (`image`, `image_lossy`, `audio`, `video`), or
2. A flat JSON object (for a single trainer).

CLI flags still win over file values if both are provided.

## Examples

```zsh
python train_autoencoder_image_local.py --params-file documentation/model_params_example.json --data-dir data/ImageData/archive --real-only --no-export-tflite
python train_autoencoder_image_lossy_local.py --params-file documentation/model_params_example.json --data-dir data/ImageData/archive --real-only --no-export-tflite
python train_autoencoder_audio_local.py --params-file documentation/model_params_example.json --data-dir data/AudioData/ESC-50-master/audio --no-export-tflite
python train_autoencoder_video_local.py --params-file documentation/model_params_example.json --data-dir "data/VIDEO DATA" --no-export-tflite
```

