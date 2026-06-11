#!/usr/bin/env python3
"""Compare FSC-Net time attention implementations.

Example:
  python compare_time_attention.py --device cuda --batch_size 2 --frames 126
"""

from __future__ import annotations

import argparse
import time

import torch

from fscnet_pytorch.model import TimeSelfAttention, TimeSelfAttentionV2


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare FSC-Net time attention blocks")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--freq_groups", type=int, default=257)
    p.add_argument("--frames", type=int, default=126)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--v2_no_qk_norm", action="store_true")
    p.add_argument("--v2_no_rope", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--no_backward",
        action="store_true",
        help="skip forward+backward timing",
    )
    return p


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


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    x = torch.randn(
        args.batch_size,
        args.channels,
        args.freq_groups,
        args.frames,
        device=device,
    )
    modules = {
        "v1": TimeSelfAttention(args.channels, args.heads, args.dropout).to(device),
        "v2": TimeSelfAttentionV2(
            args.channels,
            args.heads,
            args.dropout,
            qk_norm=not args.v2_no_qk_norm,
            rope=not args.v2_no_rope,
        ).to(device),
    }

    print(
        "input_shape="
        f"{tuple(x.shape)} device={device} warmup={args.warmup} iters={args.iters}"
    )
    for name, module in modules.items():
        y, fwd_ms = time_forward(module, x, args.warmup, args.iters)
        bwd_ms = None
        if not args.no_backward:
            x_train = x.detach().clone().requires_grad_(True)
            bwd_ms = time_backward(module, x_train, args.warmup, args.iters)
        line = (
            f"{name}: params={count_parameters(module):,} "
            f"output_shape={tuple(y.shape)} forward_ms={fwd_ms:.3f}"
        )
        if bwd_ms is not None:
            line += f" forward_backward_ms={bwd_ms:.3f}"
        print(line)


if __name__ == "__main__":
    main()
