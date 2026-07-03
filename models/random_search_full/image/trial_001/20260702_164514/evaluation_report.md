# Evaluation Report

- Source: `evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:46:04.032619Z",
  "config": {
    "preset": "custom",
    "use_cifar10": false,
    "data_dir": "/Users/tyejaedon/PycharmProjects/AI_Compressor/data/_random_search_missing",
    "valid_dir": "data/cifar10_valid",
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_001",
    "block_size": 96,
    "latent_dim": 48,
    "epochs": 6,
    "batch_size": 8,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 0.0001186,
    "allow_synthetic": true,
    "dummy_samples": 180,
    "extensive_dummy_samples": 240,
    "auto_generate_dummy": true,
    "cifar_train_limit": 0,
    "cifar_test_limit": 0,
    "target_psnr": 22.0,
    "export_tflite": false,
    "tflite_fp16": false
  },
  "split_info": {
    "source": "synthetic",
    "train": 126,
    "validation": 27,
    "test": 27
  },
  "history": {
    "loss": [
      0.10044845193624496,
      0.09329557418823242,
      0.07335023581981659,
      0.054782330989837646,
      0.044465843588113785,
      0.03649434447288513
    ],
    "mse": [
      0.04243546351790428,
      0.03977685794234276,
      0.033046405762434006,
      0.02528000809252262,
      0.020223991945385933,
      0.016239885240793228
    ],
    "psnr_metric": [
      13.725005149841309,
      14.01161003112793,
      14.828689575195312,
      15.993639945983887,
      16.959150314331055,
      17.91250228881836
    ],
    "ssim_metric": [
      0.050971321761608124,
      0.13808748126029968,
      0.38925445079803467,
      0.5844698548316956,
      0.6756864786148071,
      0.7434136867523193
    ],
    "val_loss": [
      0.09847237169742584,
      0.08323892951011658,
      0.06151752546429634,
      0.048403069376945496,
      0.040262557566165924,
      0.03223828971385956
    ],
    "val_mse": [
      0.041508764028549194,
      0.036330536007881165,
      0.02811657451093197,
      0.022073544561862946,
      0.01812099665403366,
      0.01416533999145031
    ],
    "val_psnr_metric": [
      13.819923400878906,
      14.396951675415039,
      15.512789726257324,
      16.561906814575195,
      17.416582107543945,
      18.479881286621094
    ],
    "val_ssim_metric": [
      0.07033219933509827,
      0.26157745718955994,
      0.5130560398101807,
      0.6380826830863953,
      0.710534393787384,
      0.7787089943885803
    ],
    "learning_rate": [
      0.00011860000086016953,
      0.00011860000086016953,
      0.00011860000086016953,
      0.00011860000086016953,
      0.00011860000086016953,
      0.00011860000086016953
    ]
  },
  "test_metrics": {
    "loss": 0.032210275530815125,
    "mse": 0.01419022772461176,
    "psnr_metric": 18.467327117919922,
    "ssim_metric": 0.7799257040023804
  },
  "plot_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_001/20260702_164514/training_metrics.png"
}
```
