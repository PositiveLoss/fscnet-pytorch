"""Run FSC-Net bandwidth extension inference.

Example:
  python inference.py --checkpoint runs/fscnet/last.pt --input noisy_4k.wav --output enhanced_48k.wav
"""

from __future__ import annotations

from pathlib import Path

import torch

from fscnet_pytorch.audio import load_audio, match_length, resample_audio, save_audio
from fscnet_pytorch.cli import option, run
from fscnet_pytorch.model import FSCNet, FSCNetConfig


def enhance_chunked(
    model: FSCNet, wav: torch.Tensor, chunk_seconds: float, overlap_seconds: float
) -> torch.Tensor:
    cfg = model.cfg
    if chunk_seconds <= 0:
        return model.enhance(wav)

    chunk = round(chunk_seconds * cfg.target_sr)
    overlap = round(overlap_seconds * cfg.target_sr)
    if chunk <= 0:
        raise ValueError("chunk_seconds produced zero samples")
    if overlap < 0 or overlap >= chunk:
        raise ValueError("overlap_seconds must be >=0 and smaller than chunk_seconds")
    hop = chunk - overlap
    length = wav.shape[-1]
    if length <= chunk:
        return model.enhance(match_length(wav, chunk))[..., :length]

    out = wav.new_zeros(length + overlap)
    weight = wav.new_zeros(length + overlap)
    window = torch.hann_window(chunk, device=wav.device, dtype=wav.dtype)
    # Avoid zeros at endpoints causing division issues.
    window = window.clamp_min(1.0e-4)

    pos = 0
    while pos < length:
        piece = wav[pos : pos + chunk]
        valid = piece.shape[-1]
        if valid < chunk:
            piece = match_length(piece, chunk)
        pred = model.enhance(piece)[..., :chunk]
        out[pos : pos + chunk] += pred * window
        weight[pos : pos + chunk] += window
        pos += hop

    return (out[:length] / weight[:length].clamp_min(1.0e-6)).contiguous()


def main(
    checkpoint: Path = option(
        ...,
        "--checkpoint",
        help="checkpoint produced by train.py",
    ),
    input: Path = option(..., "--input", help="input narrowband audio"),
    output: Path = option(..., "--output", help="output enhanced audio path"),
    device: str = option(
        "cuda" if torch.cuda.is_available() else "cpu", "--device", help="torch device"
    ),
    simulate_input_sr: int = option(
        0,
        "--simulate-input-sr",
        "--simulate_input_sr",
        help="optional: downsample input to this SR before enhancement",
        min=0,
    ),
    chunk_seconds: float = option(
        0.0,
        "--chunk-seconds",
        "--chunk_seconds",
        help="0 = process whole file",
        min=0.0,
    ),
    overlap_seconds: float = option(
        0.25,
        "--overlap-seconds",
        "--overlap_seconds",
        help="chunk overlap for overlap-add",
        min=0.0,
    ),
    normalize_input: bool = option(
        False,
        "--normalize-input",
        "--normalize_input",
        help="peak normalize before inference and undo scale after",
    ),
    torch_num_threads: int = option(
        1,
        "--torch-num-threads",
        "--torch_num_threads",
        help="CPU intra-op threads",
        min=0,
    ),
) -> None:
    """Run FSC-Net bandwidth extension inference."""
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    torch_device = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = FSCNetConfig.from_dict(ckpt["config"])
    model = FSCNet(cfg).to(torch_device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    wav, sr = load_audio(input, target_sr=None, mono=True)
    if simulate_input_sr > 0:
        wav = resample_audio(wav, sr, cfg.target_sr)
        wav = resample_audio(
            resample_audio(wav, cfg.target_sr, simulate_input_sr),
            simulate_input_sr,
            cfg.target_sr,
        )
    else:
        wav = resample_audio(wav, sr, cfg.target_sr)

    scale = torch.tensor(1.0)
    if normalize_input:
        scale = wav.abs().max().clamp_min(1.0e-8)
        wav = wav / scale

    with torch.no_grad():
        wav_device = wav.to(torch_device)
        enhanced = enhance_chunked(
            model, wav_device, chunk_seconds, overlap_seconds
        ).cpu()

    if normalize_input:
        enhanced = enhanced * scale
    save_audio(output, enhanced, cfg.target_sr)
    print(f"wrote {output} at {cfg.target_sr} Hz")


if __name__ == "__main__":
    run(main)
