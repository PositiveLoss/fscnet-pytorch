set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

# Install / sync the locked environment.
sync:
    uv sync --all-groups

# Run lint and type checks.
check: lint typecheck

# Run Ruff.
lint:
    uv run ruff check .

# Run Pyrefly at error level.
typecheck:
    uv run pyrefly check

# Run Pyrefly including warning-level diagnostics.
typecheck-warn:
    uv run pyrefly check --min-severity warn

# Show configured Pyrefly files and import paths.
typecheck-config:
    uv run pyrefly dump-config

# List model size presets.
sizes:
    uv run python train_fscnet.py --list_model_sizes

# Compare time attention blocks. Override with: just compare-attn cpu 1 32 64 64
compare-attn device="cuda" batch="2" channels="48" freq_groups="257" frames="126":
    uv run python compare_time_attention.py \
      --device {{device}} \
      --batch_size {{batch}} \
      --channels {{channels}} \
      --freq_groups {{freq_groups}} \
      --frames {{frames}}

# Compare bare SDPA V2 against V1.
compare-attn-bare device="cuda" batch="2" channels="48" freq_groups="257" frames="126":
    uv run python compare_time_attention.py \
      --device {{device}} \
      --batch_size {{batch}} \
      --channels {{channels}} \
      --freq_groups {{freq_groups}} \
      --frames {{frames}} \
      --v2_no_qk_norm \
      --v2_no_rope

# Tiny end-to-end training smoke test using synthetic audio.
smoke-train:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=/tmp/fscnet_just_smoke
    rm -rf "$tmp"
    mkdir -p "$tmp"
    uv run python - <<'PY'
    from pathlib import Path
    import numpy as np
    import soundfile as sf

    base = Path("/tmp/fscnet_just_smoke")
    sr = 8000
    for i, freq in enumerate((220, 330)):
        t = np.arange(sr // 4, dtype=np.float32) / sr
        wav = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(base / f"{i}.wav", wav, sr)
    (base / "train.txt").write_text(
        "\n".join(str((base / f"{i}.wav").resolve()) for i in range(2)) + "\n",
        encoding="utf-8",
    )
    PY
    uv run python train_fscnet.py \
      --train_manifest "$tmp/train.txt" \
      --out_dir "$tmp/run" \
      --model_size tiny \
      --target_sr 8000 \
      --input_sr 2000 \
      --segment_seconds 0.125 \
      --n_fft 64 \
      --win_length 64 \
      --hop_length 32 \
      --progressive_windows 17,5,1 \
      --mrstft_fft_sizes 32,64 \
      --epochs 1 \
      --batch_size 1 \
      --num_workers 0 \
      --torch_num_threads 1

# Train with a preset. Example: just train train.txt valid.txt runs/fscnet_medium medium
train train_manifest valid_manifest="" out_dir="runs/fscnet" model_size="compact":
    valid_arg="{{valid_manifest}}"; \
    if [[ -n "$valid_arg" ]]; then valid_arg="--valid_manifest $valid_arg"; fi; \
    uv run python train_fscnet.py \
      --train_manifest {{train_manifest}} \
      $valid_arg \
      --out_dir {{out_dir}} \
      --model_size {{model_size}} \
      --amp

# Train 4 kHz to 48 kHz.
train-4k48 train_manifest valid_manifest="" out_dir="runs/fscnet_4k48k" model_size="compact":
    valid_arg="{{valid_manifest}}"; \
    if [[ -n "$valid_arg" ]]; then valid_arg="--valid_manifest $valid_arg"; fi; \
    uv run python train_fscnet.py \
      --train_manifest {{train_manifest}} \
      $valid_arg \
      --out_dir {{out_dir}} \
      --model_size {{model_size}} \
      --input_sr 4000 \
      --target_sr 48000 \
      --amp

# Train 16 kHz to 48 kHz.
train-16k48 train_manifest valid_manifest="" out_dir="runs/fscnet_16k48k" model_size="compact":
    valid_arg="{{valid_manifest}}"; \
    if [[ -n "$valid_arg" ]]; then valid_arg="--valid_manifest $valid_arg"; fi; \
    uv run python train_fscnet.py \
      --train_manifest {{train_manifest}} \
      $valid_arg \
      --out_dir {{out_dir}} \
      --model_size {{model_size}} \
      --input_sr 16000 \
      --target_sr 48000 \
      --amp

# Run inference. Example: just infer runs/fscnet/last.pt input.wav output.wav
infer checkpoint input output:
    uv run python infer_fscnet.py \
      --checkpoint {{checkpoint}} \
      --input {{input}} \
      --output {{output}} \
      --normalize_input

# Export ONNX for a fixed sample length.
onnx checkpoint output sample_length="48000":
    uv run python export_fscnet_onnx.py \
      --checkpoint {{checkpoint}} \
      --output {{output}} \
      --sample_length {{sample_length}} \
      --verify
