"""Compare FSC-Net time attention implementations.

Example:
  python compare_time_attention.py --device cuda --batch_size 2 --frames 126
"""

from __future__ import annotations

import time

import torch

from fscnet_pytorch.cli import option, run
from fscnet_pytorch.model import TimeSelfAttention, TimeSelfAttentionV2


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    module.eval()
    y = module(x)
    for _ in range(warmup):
        y = module(x)
    sync(x.device)
    start = time.perf_counter()
    for _ in range(iters):
        y = module(x)
    sync(x.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    return y, elapsed_ms


def time_backward(
    module: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    module.train()
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        y = module(x)
        y.square().mean().backward()
    sync(x.device)
    start = time.perf_counter()
    for _ in range(iters):
        module.zero_grad(set_to_none=True)
        y = module(x)
        y.square().mean().backward()
    sync(x.device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def main(
    device: str = option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="torch device",
    ),
    batch_size: int = option(
        2, "--batch-size", "--batch_size", help="batch size", min=1
    ),
    channels: int = option(48, "--channels", help="feature channels", min=1),
    freq_groups: int = option(
        257, "--freq-groups", "--freq_groups", help="frequency groups", min=1
    ),
    frames: int = option(126, "--frames", help="time frames", min=1),
    heads: int = option(4, "--heads", help="attention heads", min=1),
    dropout: float = option(
        0.0, "--dropout", help="attention dropout", min=0.0, max=1.0
    ),
    v2_no_qk_norm: bool = option(
        False, "--v2-no-qk-norm", "--v2_no_qk_norm", help="disable V2 QK norm"
    ),
    v2_no_rope: bool = option(
        False, "--v2-no-rope", "--v2_no_rope", help="disable V2 RoPE"
    ),
    warmup: int = option(10, "--warmup", help="warmup iterations", min=0),
    iters: int = option(50, "--iters", help="timed iterations", min=1),
    seed: int = option(1234, "--seed", help="random seed"),
    no_backward: bool = option(
        False,
        "--no-backward",
        "--no_backward",
        help="skip forward+backward timing",
    ),
) -> None:
    """Compare FSC-Net time attention blocks."""
    torch.manual_seed(seed)
    torch_device = torch.device(device)

    x = torch.randn(
        batch_size,
        channels,
        freq_groups,
        frames,
        device=torch_device,
    )
    modules = {
        "v1": TimeSelfAttention(channels, heads, dropout).to(torch_device),
        "v2": TimeSelfAttentionV2(
            channels,
            heads,
            dropout,
            qk_norm=not v2_no_qk_norm,
            rope=not v2_no_rope,
        ).to(torch_device),
    }

    print(
        "input_shape="
        f"{tuple(x.shape)} device={torch_device} warmup={warmup} iters={iters}"
    )
    for name, module in modules.items():
        y, fwd_ms = time_forward(module, x, warmup, iters)
        bwd_ms = None
        if not no_backward:
            x_train = x.detach().clone().requires_grad_(True)
            bwd_ms = time_backward(module, x_train, warmup, iters)
        line = (
            f"{name}: params={count_parameters(module):,} "
            f"output_shape={tuple(y.shape)} forward_ms={fwd_ms:.3f}"
        )
        if bwd_ms is not None:
            line += f" forward_backward_ms={bwd_ms:.3f}"
        print(line)


if __name__ == "__main__":
    run(main)
