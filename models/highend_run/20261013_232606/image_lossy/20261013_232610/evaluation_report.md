# Image Lossy Compression Evaluation Report

- Source: `evaluation_report.md`

```json
{
  "timestamp": "2026-10-13T21:04:00.494186Z",
  "config": {
    "preset": "custom",
    "use_cifar10": true,
    "data_dir": "data",
    "train_dir": "",
    "val_dir": "",
    "test_dir": "",
    "split_file_limit": 0,
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261013_232606/image_lossy",
    "params_file": "",
    "block_size": 32,
    "latent_dim": 48,
    "model_base_filters": 32,
    "model_kernel_size": 3,
    "epochs": 6,
    "batch_size": 32,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 0.00025,
    "real_only": false,
    "real_file_limit": 0,
    "min_real_images": 300,
    "min_split_images": 20,
    "target_psnr": 60.0,
    "rate_lambda": 0.0006,
    "jpeg_quality": 85,
    "sample_count": 32,
    "run_baseline_benchmark": true,
    "target_compression_ratio": null,
    "export_tflite": false,
    "tflite_fp16": false,
    "rate_loss_mode": "entropy",
    "enable_entropy_coding": true,
    "entropy_coding_samples": 16,
    "latent_bit_depth": 8,
    "quant_noise_anneal": true
  },
  "split_info": {
    "source": "cifar10",
    "train": 35000,
    "validation": 7500,
    "test": 10000
  },
  "history": {
    "loss": [
      0.05058647692203522,
      0.02592715434730053,
      0.020014053210616112,
      0.017290927469730377,
      0.015549653209745884,
      0.014272725209593773
    ],
    "mse": [
      0.017168870195746422,
      0.0076599521562457085,
      0.00480739027261734,
      0.004057621583342552,
      0.003622613614425063,
      0.00333027308806777
    ],
    "psnr_metric": [
      19.20272445678711,
      22.024484634399414,
      23.84454917907715,
      24.600353240966797,
      25.128536224365234,
      25.525501251220703
    ],
    "ssim_metric": [
      0.6735424399375916,
      0.8324597477912903,
      0.8649306297302246,
      0.8844227194786072,
      0.897235631942749,
      0.9068864583969116
    ],
    "val_loss": [
      0.029589755460619926,
      0.021675890311598778,
      0.018161095678806305,
      0.016176501289010048,
      0.014885536395013332,
      0.013596387580037117
    ],
    "val_mse": [
      0.00928247906267643,
      0.0053225066512823105,
      0.004186554811894894,
      0.0037668708246201277,
      0.003539040684700012,
      0.003200036706402898
    ],
    "val_psnr_metric": [
      21.191186904907227,
      23.418466567993164,
      24.502378463745117,
      24.967281341552734,
      25.196563720703125,
      25.701387405395508
    ],
    "val_ssim_metric": [
      0.8109251856803894,
      0.853844165802002,
      0.8772277235984802,
      0.8925837278366089,
      0.9028803110122681,
      0.9122859239578247
    ],
    "learning_rate": [
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628
    ]
  },
  "test_metrics": {
    "loss": 0.01360605750232935,
    "mse": 0.003210002789273858,
    "psnr_metric": 25.67603302001953,
    "ssim_metric": 0.9122816324234009
  },
  "lossy_size_stats": {
    "sample_dir": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261013_232606/image_lossy/20261013_232610/lossy_samples",
    "sample_count": 32,
    "jpeg_quality": 85,
    "avg_psnr": 25.22449493408203,
    "mean_original_jpeg_bytes": 714.90625,
    "mean_reconstructed_jpeg_bytes": 648.3125,
    "mean_reconstructed_png_bytes": 2241.15625,
    "recon_jpeg_vs_orig_jpeg_ratio": 0.9068496743454124,
    "recon_png_vs_recon_jpeg_ratio": 3.4569073556348213
  },
  "matched_psnr_baseline_benchmark": {
    "available": true,
    "sample_count": 32,
    "model_target_psnr": 25.22449493408203,
    "model_reconstructed_jpeg_bytes": 648.3125,
    "model_export_jpeg_quality": 85,
    "jpeg_baseline": {
      "codec": "jpeg",
      "available": true,
      "target_psnr": 25.22449493408203,
      "matched": {
        "quality": 20,
        "mean_psnr": 25.248273730278015,
        "mean_bytes": 417.4375
      },
      "candidates": [
        {
          "quality": 5,
          "mean_psnr": 20.59721529483795,
          "mean_bytes": 330.3125
        },
        {
          "quality": 10,
          "mean_psnr": 22.952749133110046,
          "mean_bytes": 366.53125
        },
        {
          "quality": 15,
          "mean_psnr": 24.32060742378235,
          "mean_bytes": 395.40625
        },
        {
          "quality": 20,
          "mean_psnr": 25.248273730278015,
          "mean_bytes": 417.4375
        },
        {
          "quality": 25,
          "mean_psnr": 25.903515100479126,
          "mean_bytes": 438.53125
        },
        {
          "quality": 30,
          "mean_psnr": 26.411951422691345,
          "mean_bytes": 456.0625
        },
        {
          "quality": 35,
          "mean_psnr": 26.855553030967712,
          "mean_bytes": 472.65625
        },
        {
          "quality": 40,
          "mean_psnr": 27.231820464134216,
          "mean_bytes": 486.6875
        },
        {
          "quality": 45,
          "mean_psnr": 27.559637904167175,
          "mean_bytes": 502.0
        },
        {
          "quality": 50,
          "mean_psnr": 27.878845989704132,
          "mean_bytes": 515.875
        },
        {
          "quality": 55,
          "mean_psnr": 28.19457447528839,
          "mean_bytes": 530.28125
        },
        {
          "quality": 60,
          "mean_psnr": 28.530547380447388,
          "mean_bytes": 546.90625
        },
        {
          "quality": 65,
          "mean_psnr": 28.975324869155884,
          "mean_bytes": 566.1875
        },
        {
          "quality": 70,
          "mean_psnr": 29.48953253030777,
          "mean_bytes": 590.875
        },
        {
          "quality": 75,
          "mean_psnr": 30.040117263793945,
          "mean_bytes": 618.0625
        },
        {
          "quality": 80,
          "mean_psnr": 30.813414573669434,
          "mean_bytes": 659.6875
        },
        {
          "quality": 85,
          "mean_psnr": 31.715133011341095,
          "mean_bytes": 714.90625
        },
        {
          "quality": 90,
          "mean_psnr": 33.081994116306305,
          "mean_bytes": 804.875
        },
        {
          "quality": 95,
          "mean_psnr": 34.26908963918686,
          "mean_bytes": 985.40625
        }
      ]
    },
    "webp_baseline": {
      "codec": "webp",
      "available": false,
      "reason": "TensorFlow build does not expose tf.io.encode_webp/decode_webp"
    },
    "model_vs_jpeg_matched_size_ratio": 1.553076807905375
  },
  "latent_rate_stats": {
    "latent_entropy_bits_per_symbol": 6.157212576359268,
    "latent_symbols_per_image": 3072,
    "latent_bits_per_pixel": 18.471637729077806,
    "estimated_bitstream_bytes_per_image": 2364.369629321959
  },
  "entropy_coding": {
    "available": true,
    "sample_count": 16,
    "symbols_per_image": 3072,
    "total_symbols_encoded": 49152,
    "actual_compressed_bytes": 37949,
    "actual_compressed_bytes_per_image": 2371.8125,
    "entropy_estimate_bytes_per_image": 2369.282663059857,
    "note": "Static per-sample frequency table is not itself transmitted/counted; a deployed coder would need an agreed-upon or separately-coded entropy model."
  },
  "plot_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261013_232606/image_lossy/20261013_232610/training_metrics.png"
}
```
