#!/usr/bin/env python3
"""Export an FSC-Net checkpoint to ONNX.

Example:
  python export_fscnet_onnx.py \
    --checkpoint runs/fscnet_4k48k/last.pt \
    --output runs/fscnet_4k48k/fscnet.onnx \
    --sample_length 48000 \
    --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from fscnet_pytorch.model import FSCNet, FSCNetConfig, SpectralTransform

MAX_VERIFIED_OPSET = 25


class FSCNetONNXWrapper(torch.nn.Module):
    def __init__(self, model: FSCNet, sample_length: int, output: str) -> None:
        super().__init__()
        if output not in {"wav", "spectrogram"}:
            raise ValueError(f"Unsupported output kind: {output}")
        self.model = model
        self.sample_length = sample_length
        self.output = output

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        pred_ri = self.model(wav, return_all=False)
        if self.output == "spectrogram":
            return pred_ri
        return self._istft(pred_ri)

    def _istft(self, ri: torch.Tensor) -> torch.Tensor:
        cfg = self.model.cfg
        if not cfg.center:
            raise ValueError("ONNX waveform export currently requires center=True")
        if cfg.win_length != cfg.n_fft:
            raise ValueError(
                "ONNX waveform export currently requires win_length == n_fft"
            )

        real = ri[:, 0]
        imag = ri[:, 1]
        _bsz, freqs, frames = real.shape
        dtype = real.dtype
        device = real.device

        n = torch.arange(cfg.n_fft, device=device, dtype=dtype)
        k = torch.arange(freqs, device=device, dtype=dtype)
        angle = 2.0 * torch.pi * n[:, None] * k[None, :] / float(cfg.n_fft)
        cos = torch.cos(angle) / float(cfg.n_fft)
        sin = torch.sin(angle) / float(cfg.n_fft)

        weights = torch.ones(freqs, device=device, dtype=dtype)
        if freqs > 1:
            end = freqs - 1 if cfg.n_fft % 2 == 0 else freqs
            if end > 1:
                weights[1:end] = 2.0
        real = real * weights.view(1, -1, 1)
        imag = imag * weights.view(1, -1, 1)

        frames_time = torch.einsum("bft,nf->bnt", real, cos) - torch.einsum(
            "bft,nf->bnt", imag, sin
        )
        window = torch.hann_window(cfg.win_length, device=device, dtype=dtype)
        frames_time = frames_time * window.view(1, -1, 1)

        fold_weight = torch.eye(cfg.n_fft, device=device, dtype=dtype).view(
            cfg.n_fft, 1, cfg.n_fft
        )
        wav = F.conv_transpose1d(frames_time, fold_weight, stride=cfg.hop_length)

        envelope_in = window.square().view(1, cfg.n_fft, 1).expand(1, cfg.n_fft, frames)
        envelope = F.conv_transpose1d(
            envelope_in, fold_weight, stride=cfg.hop_length
        ).clamp_min(1.0e-11)
        wav = wav / envelope

        start = cfg.n_fft // 2
        return wav[:, 0, start : start + self.sample_length]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export FSC-Net checkpoint to ONNX")
    p.add_argument(
        "--checkpoint", required=True, help="checkpoint from train_fscnet.py"
    )
    p.add_argument("--output", required=True, help="output .onnx path")
    p.add_argument(
        "--sample_length",
        type=int,
        required=True,
        help="fixed input/output sample length, e.g. 48000 for one second at 48 kHz",
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--output_kind",
        choices=("wav", "spectrogram"),
        default="wav",
        help="wav exports enhanced audio; spectrogram exports final [B,2,F,frames]",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=0,
        help=(
            "ONNX opset. 0 selects the newest opset verified with ONNX Runtime "
            f"for this model ({MAX_VERIFIED_OPSET})."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify", action="store_true", help="run ONNX checker and ORT")
    p.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime provider used by --verify",
    )
    p.add_argument("--atol", type=float, default=1.0e-3)
    p.add_argument("--rtol", type=float, default=1.0e-3)
    return p


def enable_export_fft(model: FSCNet) -> None:
    for module in model.modules():
        if isinstance(module, SpectralTransform):
            module.export_manual_fft = True


def load_model(checkpoint: str, device: torch.device) -> FSCNet:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = FSCNetConfig.from_dict(ckpt["config"])
    model = FSCNet(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    enable_export_fft(model)
    return model


def choose_opset(requested: int) -> int:
    latest = onnx.defs.onnx_opset_version()
    if requested > 0:
        return requested
    return min(latest, MAX_VERIFIED_OPSET)


def verify_onnx(
    onnx_path: Path,
    wrapper: FSCNetONNXWrapper,
    sample: torch.Tensor,
    provider: str,
    atol: float,
    rtol: float,
) -> None:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    available = ort.get_available_providers()
    providers = [provider] if provider in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_providers = session.get_providers()

    with torch.no_grad():
        expected = wrapper(sample).detach().cpu().numpy()
    actual = cast(
        np.ndarray, session.run(None, {"wav": sample.detach().cpu().numpy()})[0]
    )
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"ONNX Runtime output mismatch: max_abs={max_abs:.6g}, "
            f"rtol={rtol}, atol={atol}"
        )
    print(
        f"verified with {active_providers}: shape={actual.shape}, max_abs={max_abs:.6g}"
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.sample_length <= 0:
        raise ValueError("--sample_length must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    opset = choose_opset(args.opset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    wrapper = FSCNetONNXWrapper(model, args.sample_length, args.output_kind).to(device)
    wrapper.eval()
    sample = torch.randn(args.batch_size, args.sample_length, device=device)
    output_name = (
        "enhanced_wav" if args.output_kind == "wav" else "enhanced_spectrogram"
    )

    torch.onnx.export(
        wrapper,
        (sample,),
        output,
        input_names=["wav"],
        output_names=[output_name],
        opset_version=opset,
        dynamo=True,
        optimize=True,
        verify=False,
        external_data=True,
    )

    actual_opsets = [
        imp.version for imp in onnx.load(output).opset_import if imp.domain == ""
    ]
    actual_opset = actual_opsets[0] if actual_opsets else opset
    print(f"wrote {output} with ONNX opset {actual_opset}")
    if actual_opset != opset:
        raise RuntimeError(
            f"Requested opset {opset}, but exporter wrote {actual_opset}"
        )
    if args.verify:
        verify_onnx(output, wrapper, sample, args.provider, args.atol, args.rtol)


if __name__ == "__main__":
    main()
