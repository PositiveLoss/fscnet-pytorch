"""Estimate FSC-Net generator GMACs.

Example:
  uv run python -m tools.measure_gmacs --model_size compact --seconds 2
  uv run python -m tools.measure_gmacs --channels 60 --rnn_hidden 80
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import nn

from fscnet_pytorch.cli import option, run
from fscnet_pytorch.config import resolve_model_config
from fscnet_pytorch.model import (
    FSCNet,
    FSCNetConfig,
    SpectralTransform,
    TimeSelfAttention,
    TimeSelfAttentionV2,
    count_parameters,
)

TimeAttention = Literal["v1", "v2"]


@dataclass
class MacCounter:
    by_kind: dict[str, int]

    @classmethod
    def create(cls) -> "MacCounter":
        return cls(defaultdict(int))

    def add(self, kind: str, macs: int | float) -> None:
        self.by_kind[kind] += int(macs)

    @property
    def total(self) -> int:
        return sum(self.by_kind.values())


def conv_output_macs(module: nn.Conv1d | nn.Conv2d, output: torch.Tensor) -> int:
    batch = output.shape[0]
    out_channels = output.shape[1]
    out_positions = math.prod(output.shape[2:])
    kernel_ops = math.prod(module.kernel_size) * (module.in_channels // module.groups)
    return batch * out_channels * out_positions * kernel_ops


def linear_output_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return output.numel() * module.in_features


def lstm_output_macs(module: nn.LSTM, inputs: tuple[Any, ...]) -> int:
    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        return 0
    if module.batch_first:
        batch, seq_len = x.shape[0], x.shape[1]
    else:
        seq_len, batch = x.shape[0], x.shape[1]
    directions = 2 if module.bidirectional else 1
    input_size = module.input_size
    hidden = module.hidden_size
    total = 0
    for _layer in range(module.num_layers):
        per_direction = 4 * hidden * (input_size + hidden)
        total += batch * seq_len * directions * per_direction
        input_size = hidden * directions
    return total


def attention_macs(module: nn.Module, inputs: tuple[Any, ...]) -> int:
    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        return 0
    batch, channels, freq, frames = x.shape
    if isinstance(module, TimeSelfAttention):
        heads = module.attn.num_heads
    elif isinstance(module, TimeSelfAttentionV2):
        heads = module.heads
    else:
        heads = 1
    dim_head = channels // heads
    qk = batch * freq * heads * frames * frames * dim_head
    av = batch * freq * heads * frames * frames * dim_head
    return qk + av


def spectral_fft_macs(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> int:
    batch, channels, freq, frames = input_shape
    fft_size = max(1, freq * frames)
    # Rough real FFT accounting: O(N log2 N) per channel for rFFT and iRFFT.
    return int(2 * batch * channels * fft_size * math.log2(fft_size))


def add_hooks(model: nn.Module, counter: MacCounter) -> list[Any]:
    handles: list[Any] = []

    def conv_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and isinstance(
            module, nn.Conv1d | nn.Conv2d
        ):
            counter.add(type(module).__name__, conv_output_macs(module, output))

    def linear_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, torch.Tensor) and isinstance(module, nn.Linear):
            counter.add("Linear", linear_output_macs(module, output))

    def lstm_hook(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if isinstance(module, nn.LSTM):
            counter.add("LSTM", lstm_output_macs(module, inputs))

    def attn_hook(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        counter.add(type(module).__name__, attention_macs(module, inputs))

    def spectral_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        x = inputs[0]
        if isinstance(module, SpectralTransform) and isinstance(x, torch.Tensor):
            if isinstance(output, torch.Tensor):
                counter.add(
                    "FFT/IFFT estimate",
                    spectral_fft_macs(tuple(x.shape), tuple(output.shape)),
                )

    for module in model.modules():
        if isinstance(module, nn.Conv1d | nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.LSTM):
            handles.append(module.register_forward_hook(lstm_hook))
        elif isinstance(module, TimeSelfAttention | TimeSelfAttentionV2):
            handles.append(module.register_forward_hook(attn_hook))
        elif isinstance(module, SpectralTransform):
            handles.append(module.register_forward_hook(spectral_hook))
    return handles


def resolve_sample_length(
    cfg: FSCNetConfig, seconds: float, sample_length: int | None
) -> int:
    if sample_length is not None:
        return sample_length
    return max(1, round(seconds * cfg.target_sr))


def main(
    model_size: str = option(
        "compact",
        "--model-size",
        "--model_size",
        help="model size preset; explicit architecture flags override it",
    ),
    seconds: float = option(
        2.0, "--seconds", help="input duration at target SR", min=0.0
    ),
    sample_length: int | None = option(
        None, "--sample-length", "--sample_length", help="override input samples"
    ),
    target_sr: int | None = option(
        None, "--target-sr", "--target_sr", help="target SR"
    ),
    input_sr: int | None = option(None, "--input-sr", "--input_sr", help="input SR"),
    n_fft: int | None = option(None, "--n-fft", "--n_fft", help="STFT FFT size"),
    win_length: int | None = option(
        None, "--win-length", "--win_length", help="STFT window length"
    ),
    hop_length: int | None = option(
        None, "--hop-length", "--hop_length", help="STFT hop length"
    ),
    subbands: int | None = option(None, "--subbands", help="channel-wise subbands"),
    channels: int | None = option(None, "--channels", help="model channels"),
    num_blocks: int | None = option(
        None, "--num-blocks", "--num_blocks", help="number of TF-FFC blocks"
    ),
    rnn_hidden: int | None = option(
        None, "--rnn-hidden", "--rnn_hidden", help="BLSTM hidden size"
    ),
    attention_heads: int | None = option(
        None, "--attention-heads", "--attention_heads", help="time attention heads"
    ),
    time_attention: TimeAttention | None = option(
        None, "--time-attention", "--time_attention", help="time attention variant"
    ),
    ffc_ratio: float | None = option(
        None, "--ffc-ratio", "--ffc_ratio", help="retained for config compatibility"
    ),
    dropout: float | None = option(None, "--dropout", help="dropout"),
    progressive_windows: str | None = option(
        None,
        "--progressive-windows",
        "--progressive_windows",
        help="comma-separated windows; only used to validate block count",
    ),
) -> None:
    """Measure approximate forward MACs for the FSC-Net generator."""
    overrides = {
        "target_sr": target_sr,
        "input_sr": input_sr,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "subbands": subbands,
        "channels": channels,
        "num_blocks": num_blocks,
        "rnn_hidden": rnn_hidden,
        "attention_heads": attention_heads,
        "time_attention": time_attention,
        "ffc_ratio": ffc_ratio,
        "dropout": dropout,
    }
    cfg, windows = resolve_model_config(
        model_size, overrides=overrides, progressive_windows=progressive_windows
    )
    length = resolve_sample_length(cfg, seconds, sample_length)
    model = FSCNet(cfg).eval()
    counter = MacCounter.create()
    handles = add_hooks(model, counter)
    sample = torch.zeros(1, length)
    try:
        with torch.no_grad():
            output = model(sample, return_all=False)
    finally:
        for handle in handles:
            handle.remove()

    print(f"model_size={model_size} windows={windows}")
    print(f"config={cfg.to_dict()}")
    print(f"input_shape={tuple(sample.shape)} output_shape={tuple(output.shape)}")
    print(f"parameters={count_parameters(model):,}")
    print(f"total={counter.total / 1e9:.3f} GMACs")
    for kind, macs in sorted(counter.by_kind.items()):
        print(f"{kind}: {macs / 1e9:.3f} GMACs")


if __name__ == "__main__":
    run(main)
