# FSC-Net PyTorch implementation for arXiv:2606.06962v1

This is a runnable implementation of the paper **“FSC-Net: Integrating Fast Fourier Convolutions and Progressive Learning for Speech Bandwidth Extension.”**

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
{"hr_path":"/path/to/clean_48k_002.wav", "lr_path":"/path/to/precomputed_4k_or_16k.wav"}
```

If `lr_path` is absent, the dataset creates narrowband input on the fly:

```text
clean target at 48 kHz -> downsample to --input_sr -> resample back to 48 kHz
```

## Train

List built-in model size presets:

```bash
python train_fscnet.py --list_model_sizes
```

| preset | params | blocks | channels | hidden | attention | suggested batch |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `tiny` | 0.121 M | 3 | 24 | 32 | v1 | 8 |
| `small` | 0.299 M | 4 | 32 | 48 | v1 | 6 |
| `compact` | 0.754 M | 5 | 48 | 64 | v1 | 4 |
| `medium` | 1.718 M | 6 | 64 | 96 | v2 bare SDPA | 2 |
| `large` | 3.525 M | 6 | 96 | 128 | v2 bare SDPA | 1 |

4 kHz to 48 kHz:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_4k48k \
  --model_size compact \
  --input_sr 4000 \
  --target_sr 48000 \
  --epochs 100 \
  --amp
```

16 kHz to 48 kHz:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_16k48k \
  --model_size compact \
  --input_sr 16000 \
  --target_sr 48000 \
  --epochs 100 \
  --amp
```

Train different model sizes:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_tiny \
  --model_size tiny \
  --epochs 50 \
  --amp

python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_medium \
  --model_size medium \
  --epochs 100 \
  --amp

python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_large \
  --model_size large \
  --epochs 150 \
  --segment_seconds 1.5 \
  --amp
```

Architecture flags override the preset, so this is valid:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_custom \
  --model_size small \
  --channels 40 \
  --num_blocks 5 \
  --rnn_hidden 64 \
  --progressive_windows 257,65,17,5,1 \
  --amp
```

Enable adversarial training after the reconstruction loss starts converging:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_gan \
  --adv_weight 0.05 \
  --adv_start_step 10000 \
  --fm_weight 10
```

Try the SDPA-based time attention variant:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_v2attn \
  --time_attention v2
```

Compare the time-attention blocks directly:

```bash
python compare_time_attention.py --device cuda
python compare_time_attention.py --device cuda --v2_no_qk_norm --v2_no_rope
```

Track training with Trackio:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_tracked \
  --model_size compact \
  --trackio \
  --trackio_project fscnet \
  --trackio_name compact_4k48 \
  --amp
```

Trackio logs locally by default:

```bash
trackio show --project fscnet
```

To send metrics to a hosted Hugging Face Space or self-hosted Trackio server,
add `--trackio_space_id username/space-name` or
`--trackio_server_url http://host:port`.

## Inference

For a real narrowband file:

```bash
python infer_fscnet.py \
  --checkpoint runs/fscnet_4k48k/last.pt \
  --input input_4k.wav \
  --output enhanced_48k.wav \
  --normalize_input
```

To simulate a 4 kHz input from a full-band file before enhancement:

```bash
python infer_fscnet.py \
  --checkpoint runs/fscnet_4k48k/last.pt \
  --input clean_48k.wav \
  --output enhanced_from_simulated_4k.wav \
  --simulate_input_sr 4000
```

For long files, use chunking:

```bash
python infer_fscnet.py \
  --checkpoint runs/fscnet_4k48k/last.pt \
  --input input_4k.wav \
  --output enhanced_48k.wav \
  --chunk_seconds 4 \
  --overlap_seconds 0.5
```

## ONNX export

Export a trained checkpoint for a fixed input length:

```bash
python export_fscnet_onnx.py \
  --checkpoint runs/fscnet_4k48k/last.pt \
  --output runs/fscnet_4k48k/fscnet_1s.onnx \
  --sample_length 48000 \
  --verify
```

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
