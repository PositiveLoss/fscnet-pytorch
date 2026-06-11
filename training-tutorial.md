# Training FSC-Net on Google Colab with a Hugging Face Dataset

This guide shows how to train your own FSC-Net bandwidth-extension model on
Google Colab using an NVIDIA L4 GPU and a 48 kHz dataset hosted on Hugging Face
Datasets.

The examples below assume you want to train a model that reconstructs 48 kHz
audio from a simulated narrowband input, for example `4 kHz -> 48 kHz` or
`16 kHz -> 48 kHz`.

## 1. Start a Colab GPU runtime

In Colab:

1. Open `Runtime -> Change runtime type`.
2. Select `GPU`.
3. Use the L4 runtime.
4. Optional but recommended: mount Google Drive so checkpoints survive runtime
   resets.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set paths and training choices in one place:

```bash
export REPO_URL="https://github.com/YOUR_USER_OR_ORG/fscnet-pytorch.git"
export WORKDIR="/content/fscnet-pytorch"
export DATA_ROOT="/content/fscnet_data"
export RUN_ROOT="/content/drive/MyDrive/fscnet_runs"

# Replace this with your Hugging Face dataset id.
export HF_DATASET_ID="your-org/your-48khz-dataset"

# Pick the narrowband bottleneck used to create training input.
# Common choices:
#   4000  = strong bandwidth-extension task
#   16000 = easier wideband-to-fullband task
export INPUT_SR="4000"
export TARGET_SR="48000"
```

## 2. Install uv and clone the repo

This project is managed with `uv` and declares its Python dependencies in
`pyproject.toml`.

```bash
pip install -U uv
git clone "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"

# Let uv install the Python version requested by the project if Colab's system
# Python is older than the project's requirement.
uv sync

# Extra tools used only by this tutorial to download/export HF datasets.
uv pip install datasets soundfile huggingface_hub
```

Check that CUDA is visible from the project environment:

```bash
cd "$WORKDIR"
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16:", torch.cuda.is_bf16_supported())
PY
```

List the built-in model presets:

```bash
cd "$WORKDIR"
uv run python train.py --list_model_sizes
```

For an L4, start with `compact`. If you hit CUDA out-of-memory errors, use
`small` or reduce `--batch_size` and `--segment_seconds`.

## 3. Download or export your Hugging Face dataset

The trainer expects local audio paths in a manifest. A plain manifest can be one
audio file path per line, and JSONL can contain `hr_path` plus optional `lr_path`.

Use one of the following dataset preparation methods.

### Option A: Dataset contains audio files directly

Use this if your HF dataset repository contains `.wav`, `.flac`, `.ogg`,
`.aiff`, or similar audio files as regular files.

```bash
cd "$WORKDIR"
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
import os

dataset_id = os.environ["HF_DATASET_ID"]
out_dir = Path(os.environ["DATA_ROOT"]) / "hf_snapshot"
out_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    local_dir=out_dir,
)
print(out_dir)
PY
```

If the dataset is private or gated, log in first:

```bash
uv run huggingface-cli login
```

### Option B: Dataset uses an Audio column

Use this if the dataset is stored as Parquet/Arrow and exposes an Audio feature.
Replace `AUDIO_COLUMN` if your dataset uses a different column name.

```bash
cd "$WORKDIR"
uv run python - <<'PY'
from pathlib import Path
import os

import soundfile as sf
from datasets import Audio, load_dataset

dataset_id = os.environ["HF_DATASET_ID"]
audio_column = os.environ.get("AUDIO_COLUMN", "audio")
split = os.environ.get("HF_SPLIT", "train")
target_sr = int(os.environ.get("TARGET_SR", "48000"))

out_dir = Path(os.environ["DATA_ROOT"]) / "hr_48k"
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(dataset_id, split=split)
ds = ds.cast_column(audio_column, Audio(sampling_rate=target_sr))

manifest = out_dir / "hr_manifest.txt"
with manifest.open("w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        audio = row[audio_column]
        path = out_dir / f"{i:08d}.wav"
        sf.write(path, audio["array"], audio["sampling_rate"], subtype="PCM_16")
        f.write(str(path.resolve()) + "\n")
        if (i + 1) % 1000 == 0:
            print("exported", i + 1)

print("manifest:", manifest)
PY
```

## 4. Build FSC-Net manifests

The recommended path is to precompute paired files once. The script writes:

- `hr/*.wav`: clean 48 kHz target audio
- `lr_${INPUT_SR}/*.wav`: audio downsampled through `INPUT_SR`, then resampled
  back to 48 kHz for model input
- `manifest.jsonl`: rows with `hr_path` and `lr_path`

For Option A, point `--input_dir` at the downloaded snapshot directory. For
Option B, point it at `DATA_ROOT/hr_48k`.

```bash
cd "$WORKDIR"

# Option A input:
export CLEAN_AUDIO_DIR="$DATA_ROOT/hf_snapshot"

# Option B input:
# export CLEAN_AUDIO_DIR="$DATA_ROOT/hr_48k"

uv run python -m tools.generate_resampled_manifest \
  --input_dir "$CLEAN_AUDIO_DIR" \
  --out_dir "$DATA_ROOT/fscnet_${INPUT_SR}_48k" \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --quality balanced \
  --backend auto \
  --min_duration_seconds 0.1 \
  --max_duration_seconds 30 \
  --workers 0
```

Split the generated manifest:

```bash
cd "$WORKDIR"
uv run python -m tools.split_manifest \
  --manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/manifest.jsonl" \
  --valid_ratio 0.1 \
  --seed 1234
```

This creates:

```text
$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl
$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl
```

## 5. Run a tiny smoke test

Before launching a long run, verify that the manifests and GPU path work:

```bash
cd "$WORKDIR"
uv run python train.py \
  --train_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl" \
  --valid_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl" \
  --out_dir "$RUN_ROOT/smoke_${INPUT_SR}_48k" \
  --model_size tiny \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --epochs 1 \
  --batch_size 1 \
  --segment_seconds 1.0 \
  --num_workers 2 \
  --precision bf16 \
  --no-eval-metrics
```

If bf16 is not supported for your runtime, use `--precision fp16` or `--amp`
instead.

## 6. Train on an L4

Start with the `compact` preset. It is the default architecture and is a good
first full run on an L4.

```bash
cd "$WORKDIR"
uv run python train.py \
  --train_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl" \
  --valid_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl" \
  --out_dir "$RUN_ROOT/compact_${INPUT_SR}_48k" \
  --model_size compact \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --epochs 100 \
  --batch_size 4 \
  --segment_seconds 2.0 \
  --num_workers 4 \
  --precision bf16 \
  --save_every 1 \
  --valid_every 1
```

For lower memory:

```bash
cd "$WORKDIR"
uv run python train.py \
  --train_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl" \
  --valid_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl" \
  --out_dir "$RUN_ROOT/small_${INPUT_SR}_48k" \
  --model_size small \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --epochs 100 \
  --batch_size 2 \
  --segment_seconds 1.5 \
  --num_workers 4 \
  --precision bf16
```

For more capacity after you have a stable compact run:

```bash
cd "$WORKDIR"
uv run python train.py \
  --train_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl" \
  --valid_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl" \
  --out_dir "$RUN_ROOT/medium_${INPUT_SR}_48k" \
  --model_size medium \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --epochs 120 \
  --batch_size 2 \
  --segment_seconds 1.5 \
  --num_workers 4 \
  --precision bf16
```

Checkpoints are written to the output directory as:

```text
last.safetensors
checkpoint_epoch_0001.safetensors
checkpoint_epoch_0002.safetensors
...
config.json
```

Training automatically resumes from `OUT_DIR/last.safetensors` when it exists.
To force a fresh run in the same directory, add `--no-auto-resume`.

## 7. Optional: train without precomputed LR files

You can train from clean 48 kHz audio paths only. The dataset loader will create
the narrowband input on the fly:

```text
clean 48 kHz target -> downsample to INPUT_SR -> resample back to 48 kHz
```

Create and split a plain-text manifest yourself:

```bash
find "$CLEAN_AUDIO_DIR" -type f \( -name '*.wav' -o -name '*.flac' -o -name '*.ogg' \) \
  > "$DATA_ROOT/hr_paths.txt"
```

Then split it:

```bash
cd "$WORKDIR"
uv run python -m tools.split_manifest \
  --manifest "$DATA_ROOT/hr_paths.txt" \
  --valid_ratio 0.1 \
  --seed 1234
```

This is simpler, but precomputing LR files is usually faster during training.

## 8. Optional: Trackio logging

Track training locally with Trackio:

```bash
cd "$WORKDIR"
uv run python train.py \
  --train_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/train.jsonl" \
  --valid_manifest "$DATA_ROOT/fscnet_${INPUT_SR}_48k/valid.jsonl" \
  --out_dir "$RUN_ROOT/tracked_compact_${INPUT_SR}_48k" \
  --model_size compact \
  --input_sr "$INPUT_SR" \
  --target_sr "$TARGET_SR" \
  --epochs 100 \
  --batch_size 4 \
  --precision bf16 \
  --trackio \
  --trackio_project fscnet \
  --trackio_name "compact_${INPUT_SR}_48k_l4"
```

Open the local Trackio UI:

```bash
cd "$WORKDIR"
uv run trackio show --project fscnet
```

## 9. Run inference with a trained checkpoint

Use `last.safetensors` or a numbered epoch checkpoint:

```bash
cd "$WORKDIR"
uv run python inference.py \
  --checkpoint "$RUN_ROOT/compact_${INPUT_SR}_48k/last.safetensors" \
  --input "/content/example_input.wav" \
  --output "/content/enhanced_48k.wav" \
  --normalize_input \
  --chunk_seconds 8 \
  --overlap_seconds 0.25
```

If you want to test the model from a clean 48 kHz file by simulating the
narrowband input first:

```bash
cd "$WORKDIR"
uv run python inference.py \
  --checkpoint "$RUN_ROOT/compact_${INPUT_SR}_48k/last.safetensors" \
  --input "/content/clean_48k_example.wav" \
  --output "/content/enhanced_from_simulated_input.wav" \
  --simulate_input_sr "$INPUT_SR" \
  --normalize_input \
  --chunk_seconds 8
```

## 10. Practical L4 settings

Use these as starting points:

| Preset | Batch size | Segment seconds | Precision | Notes |
| --- | ---: | ---: | --- | --- |
| `tiny` | 1-8 | 1.0-2.0 | `bf16` | Smoke tests and quick checks |
| `small` | 2-6 | 1.5-2.0 | `bf16` | Safer if memory is tight |
| `compact` | 2-4 | 2.0 | `bf16` | Recommended first real run |
| `medium` | 1-2 | 1.0-1.5 | `bf16` | Try after compact is stable |
| `large` | 1 | 1.0 | `bf16` | May be slow or memory-limited on L4 |

If you get CUDA out-of-memory errors:

1. Lower `--batch_size`.
2. Lower `--segment_seconds`.
3. Switch from `compact` to `small`.
4. Restart the runtime to clear fragmented GPU memory.
5. Keep `--precision bf16` or use `--precision fp16`.

## 11. Common problems

### `--train_manifest is required`

Run `tools.generate_resampled_manifest` and `tools.split_manifest` first, then
pass the generated `train.jsonl`.

### Dataset audio column has a different name

Set `AUDIO_COLUMN` before running the export cell:

```bash
export AUDIO_COLUMN="your_audio_column"
```

### Private Hugging Face dataset fails to download

Log in inside the same runtime:

```bash
cd "$WORKDIR"
uv run huggingface-cli login
```

### PESQ validation is slow or unavailable

Disable optional evaluation metrics:

```bash
--no-eval-metrics
```

Validation loss still runs when `--valid_manifest` is provided.

### Colab runtime resets

Keep `RUN_ROOT` on Google Drive. Re-run setup, dataset preparation if needed,
then launch the same training command. The trainer resumes automatically from
`last.safetensors` in the output directory.
