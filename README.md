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

4 kHz to 48 kHz:

```bash
pip install -r requirements.txt
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_4k48k \
  --input_sr 4000 \
  --target_sr 48000 \
  --epochs 100 \
  --batch_size 4 \
  --amp
```

16 kHz to 48 kHz:

```bash
python train_fscnet.py \
  --train_manifest train.txt \
  --valid_manifest valid.txt \
  --out_dir runs/fscnet_16k48k \
  --input_sr 16000 \
  --target_sr 48000 \
  --epochs 100 \
  --batch_size 4 \
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

## Notes

- The default `--num_blocks 5` matches the five progressive windows shown in the demo: `257,65,17,5,1`.
- The default generator is intentionally compact. Increase `--channels` and `--rnn_hidden` if you have GPU memory and want a larger model.
- The optional discriminator is not required for a first run. Start with `--adv_weight 0` and add GAN training later.
