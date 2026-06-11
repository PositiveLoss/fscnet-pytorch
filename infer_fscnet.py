#!/usr/bin/env python3
"""Run FSC-Net bandwidth extension inference.

Example:
  python infer_fscnet.py --checkpoint runs/fscnet/last.pt --input noisy_4k.wav --output enhanced_48k.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from fscnet_pytorch.audio import load_audio, match_length, resample_audio, save_audio
from fscnet_pytorch.model import FSCNet, FSCNetConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FSC-Net inference")
    p.add_argument(
        "--checkpoint", required=True, help="checkpoint produced by train_fscnet.py"
    )
    p.add_argument("--input", required=True, help="input narrowband audio")
    p.add_argument("--output", required=True, help="output enhanced audio path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--simulate_input_sr",
        type=int,
        default=0,
        help="optional: downsample input to this SR before enhancement",
    )
    p.add_argument(
        "--chunk_seconds", type=float, default=0.0, help="0 = process whole file"
    )
    p.add_argument(
        "--overlap_seconds",
        type=float,
        default=0.25,
        help="chunk overlap for overlap-add",
    )
    p.add_argument(
        "--normalize_input",
        action="store_true",
        help="peak normalize before inference and undo scale after",
    )
    p.add_argument(
        "--torch_num_threads", type=int, default=1, help="CPU intra-op threads"
    )
    return p


def enhance_chunked(
    model: FSCNet, wav: torch.Tensor, chunk_seconds: float, overlap_seconds: float
) -> torch.Tensor:
    cfg = model.cfg
    if chunk_seconds <= 0:
        return model.enhance(wav)

    chunk = int(round(chunk_seconds * cfg.target_sr))
    overlap = int(round(overlap_seconds * cfg.target_sr))
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


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = FSCNetConfig.from_dict(ckpt["config"])
    model = FSCNet(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    wav, sr = load_audio(args.input, target_sr=None, mono=True)
    if args.simulate_input_sr > 0:
        wav = resample_audio(wav, sr, cfg.target_sr)
        wav = resample_audio(
            resample_audio(wav, cfg.target_sr, args.simulate_input_sr),
            args.simulate_input_sr,
            cfg.target_sr,
        )
    else:
        wav = resample_audio(wav, sr, cfg.target_sr)

    scale = torch.tensor(1.0)
    if args.normalize_input:
        scale = wav.abs().max().clamp_min(1.0e-8)
        wav = wav / scale

    with torch.no_grad():
        wav_device = wav.to(device)
        enhanced = enhance_chunked(
            model, wav_device, args.chunk_seconds, args.overlap_seconds
        ).cpu()

    if args.normalize_input:
        enhanced = enhanced * scale
    save_audio(args.output, enhanced, cfg.target_sr)
    print(f"wrote {args.output} at {cfg.target_sr} Hz")


if __name__ == "__main__":
    main()
