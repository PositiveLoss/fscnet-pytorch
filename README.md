# Full-Spectrum Context Network (FSC-Net) PyTorch implementation

[![Python checks](https://github.com/PositiveLoss/fscnet-pytorch/actions/workflows/python-checks.yml/badge.svg)](https://github.com/PositiveLoss/fscnet-pytorch/actions/workflows/python-checks.yml)

This is a runnable implementation of the paper [**“FSC-Net: Integrating Fast Fourier Convolutions and Progressive Learning for Speech Bandwidth Extension.”**](https://arxiv.org/abs/2606.06962)

The paper says the official source code will be released after acceptance, so this repository is a faithful implementation from the article and demo, not the authors' original code.

## What is implemented

- Complex STFT spectral mapping.
- 48 kHz target sample rate with 32 ms STFT window and 16 ms hop by default.
- Channel-wise subband split/merge with 3 subbands by default.
- TF-FFC blocks with:
  - local/global Fast Fourier Convolution branches,
  - intra-frequency BLSTM,
  - time self-attention.
- Frequency-progressive targets using default windows: `257,65,17,5,1`.
- Multi-resolution STFT loss, LSD loss, complex L1 loss.
- Optional per-stage conditional multi-scale LSGAN discriminator and feature matching.
- Training and inference scripts.

## Manifest formats

Plain text:

```txt
/path/to/clean_48k_001.wav
/path/to/clean_48k_002.wav
```

JSONL:

```jsonl
{"hr_path":"/path/to/clean_48k_001.wav"}
{"hr_path":"/path/to/clean_48k_002.wav", "lr_path":"/path/to/precomputed_4k_bandlimited_48k.wav"}
```

If `lr_path` is absent, the dataset creates narrowband input on the fly:

```text
clean target at 48 kHz -> downsample to --input_sr -> resample back to 48 kHz
```

To precompute paired inputs with `fast-audio-resampler`, generate a JSONL
manifest with clean HR files and simulated LR files:

```bash
uv run python -m tools.generate_resampled_manifest \
  --input_dir /path/to/clean_audio \
  --out_dir data/fscnet_4k48 \
  --input_sr 4000 \
  --target_sr 48000 \
  --quality balanced \
  --backend auto \
  --min_duration_seconds 0.1 \
  --max_duration_seconds 30 \
  --workers 0
```

The script writes `data/fscnet_4k48/manifest.jsonl` with `hr_path` and
`lr_path` entries. Files under `lr_4000` are stored at `--target_sr`; the
`4000` means they were downsampled through a 4 kHz bottleneck and resampled
back to the target rate for model input. `--workers 0` uses all available CPU
cores; set `--workers 1` for sequential processing. Files shorter than 0.1s or
longer than 30s are skipped by default.

Split the generated manifest into train and validation files:

```bash
uv run python -m tools.split_manifest \
  --manifest data/fscnet_4k48/manifest.jsonl \
  --valid_ratio 0.1 \
  --seed 1234
```

This writes `data/fscnet_4k48/train.jsonl` and `data/fscnet_4k48/valid.jsonl`.
Use those manifests for training:

```bash
uv run python train.py \
  --train_manifest data/fscnet_4k48/train.jsonl \
  --valid_manifest data/fscnet_4k48/valid.jsonl \
  --out_dir runs/fscnet_4k48 \
  --input_sr 4000 \
  --target_sr 48000
```

## Train

List built-in model size presets:

```bash
uv run python train.py --list_model_sizes
```

| preset | params | blocks | channels | hidden | attention | suggested batch |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `tiny` | 0.144 M | 3 | 24 | 32 | v1 | 8 |
| `small` | 0.363 M | 4 | 32 | 48 | v1 | 6 |
| `compact` | 0.903 M | 5 | 48 | 64 | v1 | 4 |
| `medium` | 2.097 M | 6 | 64 | 96 | v2 bare SDPA | 2 |
| `large` | 4.238 M | 6 | 96 | 128 | v2 bare SDPA | 1 |

4 kHz to 48 kHz:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_4k48k \
  --model_size compact \
  --input_sr 4000 \
  --target_sr 48000 \
  --epochs 100 \
  --precision fp16
```

Use `--precision bf16` for bf16 CUDA autocast, or `--precision fp16` for
explicit fp16 autocast.
Optional pyptx kernels support fp32, fp16, and bf16 training, and are disabled
by default. Set `FSCNET_ENABLE_PYPTX=1` to enable the progressive target kernel
on CUDA. With that global switch enabled, set `FSCNET_ENABLE_PYPTX_NORM=1` to
enable the global layer norm kernel, and set `FSCNET_ENABLE_PYPTX_ROPE_QK=1` to
enable the fused RoPE/QK-normalization kernel for v2 attention models.
Checkpoints are written as `*.safetensors` plus a matching `*.json` sidecar
for config, optimizer, scheduler, and resume metadata.
Training auto-resumes from `OUT_DIR/last.safetensors` when it exists; pass
`--no-auto-resume` to force a fresh run or `--resume PATH` to choose a specific
checkpoint.

Validation runs after each epoch by default when `--valid_manifest` is set.
Use `--eval-steps N` to validate every N optimizer steps instead of waiting
for epoch-end validation.
It reports reconstruction loss and Log-Spectral Distance (LSD), one of the
FSC-Net paper's objective evaluation metrics. PESQ is attempted when the
optional `pesq` package is available. Missing optional metric dependencies are
reported once and do not stop training. Use `--no-eval-metrics` to keep
validation to loss-only reporting.

16 kHz to 48 kHz:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_16k48k \
  --model_size compact \
  --input_sr 16000 \
  --target_sr 48000 \
  --epochs 100 \
  --precision fp16
```

Train different model sizes:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_tiny \
  --model_size tiny \
  --epochs 50 \
  --precision fp16

uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_medium \
  --model_size medium \
  --epochs 100 \
  --precision fp16

uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_large \
  --model_size large \
  --epochs 150 \
  --segment_seconds 1.5 \
  --precision fp16
```

Architecture flags override the preset, so this is valid:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_custom \
  --model_size small \
  --channels 40 \
  --num_blocks 5 \
  --rnn_hidden 64 \
  --progressive_windows 257,65,17,5,1 \
  --precision fp16
```

Enable adversarial training after the reconstruction loss starts converging:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_gan \
  --adv_weight 0.05 \
  --adv_start_step 10000 \
  --fm_weight 10
```

Try the SDPA-based time attention variant:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_v2attn \
  --time_attention v2
```

Compare the time-attention blocks directly:

```bash
uv run python -m tools.compare_time_attention --device cuda
uv run python -m tools.compare_time_attention --device cuda --v2_no_qk_norm --v2_no_rope
```

Track training with Trackio:

```bash
uv run python train.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_tracked \
  --model_size compact \
  --trackio \
  --trackio_project fscnet \
  --trackio_name compact_4k48 \
  --precision fp16
```

Trackio logs locally by default:

```bash
uv run trackio show --project fscnet
```

To send metrics to a hosted Hugging Face Space or self-hosted Trackio server,
add `--trackio_space_id username/space-name` or
`--trackio_server_url http://host:port`.

## Inference

For a real narrowband file:

```bash
uv run python inference.py \
  --checkpoint runs/fscnet_4k48k/last.safetensors \
  --input input_4k.wav \
  --output enhanced_48k.wav \
  --precision bf16 \
  --normalize_input
```

To simulate a 4 kHz input from a full-band file before enhancement:

```bash
uv run python inference.py \
  --checkpoint runs/fscnet_4k48k/last.safetensors \
  --input clean_48k.wav \
  --output enhanced_from_simulated_4k.wav \
  --simulate_input_sr 4000
```

For long files, use chunking:

```bash
uv run python inference.py \
  --checkpoint runs/fscnet_4k48k/last.safetensors \
  --input input_4k.wav \
  --output enhanced_48k.wav \
  --chunk_seconds 4 \
  --overlap_seconds 0.5
```

## ONNX export

Export a trained checkpoint for a fixed input length:

```bash
uv run python -m tools.export_to_onnx \
  --checkpoint runs/fscnet_4k48k/last.safetensors \
  --output runs/fscnet_4k48k/fscnet_1s.onnx \
  --sample_length 48000 \
  --verify
```

Add `--precision bf16` to export a bf16 ONNX graph. ONNX Runtime CPU
verification does not support every bf16 op used by this model.

The default exporter writes the enhanced waveform output. Use
`--output_kind spectrogram` to export the final complex spectrum as
`[batch, 2, freq, frames]` instead.

By default the script uses opset 25, the newest opset verified here with
ONNX Runtime for this model. ONNX 1.21 reports opset 26 as latest, but the
PyTorch 2.12 exporter currently leaves the fused attention op at an invalid
opset-18 form when asked to convert this graph to opset 26.

## Notes

- The default `--num_blocks 5` matches the five progressive windows shown in the demo: `257,65,17,5,1`.
- The default generator is intentionally compact. Increase `--channels` and `--rnn_hidden` if you have GPU memory and want a larger model.
- The optional discriminator is not required for a first run. Start with `--adv_weight 0` and add GAN training later.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
