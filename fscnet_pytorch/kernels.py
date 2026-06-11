"""Optional pyptx kernels for training hot paths."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Sequence

import torch


def _arch() -> str:
    if not torch.cuda.is_available():
        return "sm_80"
    major, minor = torch.cuda.get_device_capability()
    if major >= 10:
        return "sm_100a" if major == 10 else f"sm_{major}{minor}"
    if major >= 9:
        return "sm_90a"
    if major >= 8:
        return "sm_80"
    return f"sm_{major}{minor}"


def _can_use_progressive_target_kernel(
    input_ri: torch.Tensor, target_ri: torch.Tensor
) -> bool:
    return (
        input_ri.is_cuda
        and target_ri.is_cuda
        and input_ri.dtype == torch.float32
        and target_ri.dtype == torch.float32
        and input_ri.ndim == 4
        and target_ri.ndim == 4
        and input_ri.shape == target_ri.shape
        and input_ri.shape[1] == 2
        and not input_ri.requires_grad
        and not target_ri.requires_grad
    )


def make_progressive_targets_pyptx(
    input_ri: torch.Tensor,
    target_ri: torch.Tensor,
    windows: Sequence[int],
    eps: float = 1.0e-8,
) -> list[torch.Tensor] | None:
    """Build progressive targets with pyptx when the safe fast path applies.

    Returns ``None`` when the caller should use the regular PyTorch path.
    """

    if not _can_use_progressive_target_kernel(input_ri, target_ri):
        return None
    if any(window <= 0 or window % 2 == 0 for window in windows):
        return None

    input_c = input_ri.contiguous()
    target_c = target_ri.contiguous()
    bsz, _, freq, frames = input_c.shape
    try:
        return [
            _progressive_target_kernel(bsz, freq, frames, int(window), float(eps))(
                input_c, target_c
            )
            for window in windows
        ]
    except Exception:
        return None


@lru_cache(maxsize=64)
def _progressive_target_kernel(
    bsz: int, freq: int, frames: int, window: int, eps: float
):
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import f32, u32

    block = 128
    grid = ((frames + block - 1) // block, freq, bsz)
    radius = window // 2
    arch = _arch()
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(
            Tile(bsz, 2, freq, frames, f32),
            Tile(bsz, 2, freq, frames, f32),
        ),
        out_specs=(Tile(bsz, 2, freq, frames, f32),),
        grid=grid,
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def progressive_target(input_ri, target_ri, out_ri):
        pin, ptgt, pout = ptx.global_ptrs(input_ri, target_ri, out_ri)

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        block_x = reg.scalar(u32)
        ptx.inst.mov.u32(block_x, ptx.special.ctaid.x())
        frame = block_x * block + tid

        with ptx.if_(frame < frames):
            fidx = reg.scalar(u32)
            ptx.inst.mov.u32(fidx, ptx.special.ctaid.y())
            bidx = reg.scalar(u32)
            ptx.inst.mov.u32(bidx, ptx.special.ctaid.z())

            bt_base = bidx * (2 * freq * frames)
            real_base = bt_base + fidx * frames + frame
            imag_base = bt_base + (freq * frames) + fidx * frames + frame

            in_r = reg.scalar(f32)
            in_i = reg.scalar(f32)
            tgt_r = reg.scalar(f32)
            tgt_i = reg.scalar(f32)
            ptx.inst.ld.global_.f32(in_r, ptx.addr(pin + real_base * 4))
            ptx.inst.ld.global_.f32(in_i, ptx.addr(pin + imag_base * 4))
            ptx.inst.ld.global_.f32(tgt_r, ptx.addr(ptgt + real_base * 4))
            ptx.inst.ld.global_.f32(tgt_i, ptx.addr(ptgt + imag_base * 4))

            in_mag2 = reg.scalar(f32)
            ptx.inst.mul.f32(in_mag2, in_r, in_r)
            ptx.inst.fma.rn.f32(in_mag2, in_i, in_i, in_mag2)
            mag_x = reg.scalar(f32)
            ptx.inst.sqrt.rn.f32(mag_x, in_mag2)

            tgt_mag2 = reg.scalar(f32)
            ptx.inst.mul.f32(tgt_mag2, tgt_r, tgt_r)
            ptx.inst.fma.rn.f32(tgt_mag2, tgt_i, tgt_i, tgt_mag2)
            mag_y = reg.scalar(f32)
            ptx.inst.sqrt.rn.f32(mag_y, tgt_mag2)

            mag = reg.scalar(f32)
            if window <= 1:
                ptx.inst.mov.f32(mag, mag_y)
            else:
                residual_sum = reg.scalar(f32, init=0.0)
                for offset in range(-radius, radius + 1):
                    ff = fidx
                    if offset < 0:
                        valid = fidx >= -offset
                        ff = fidx - (-offset)
                    elif offset > 0:
                        valid = fidx < (freq - offset)
                        ff = fidx + offset
                    else:
                        valid = fidx < freq

                    with ptx.if_(valid):
                        r_base = bt_base + ff * frames + frame
                        i_base = bt_base + (freq * frames) + ff * frames + frame

                        xr = reg.scalar(f32)
                        xi = reg.scalar(f32)
                        yr = reg.scalar(f32)
                        yi = reg.scalar(f32)
                        ptx.inst.ld.global_.f32(xr, ptx.addr(pin + r_base * 4))
                        ptx.inst.ld.global_.f32(xi, ptx.addr(pin + i_base * 4))
                        ptx.inst.ld.global_.f32(yr, ptx.addr(ptgt + r_base * 4))
                        ptx.inst.ld.global_.f32(yi, ptx.addr(ptgt + i_base * 4))

                        xm2 = reg.scalar(f32)
                        ptx.inst.mul.f32(xm2, xr, xr)
                        ptx.inst.fma.rn.f32(xm2, xi, xi, xm2)
                        xm = reg.scalar(f32)
                        ptx.inst.sqrt.rn.f32(xm, xm2)

                        ym2 = reg.scalar(f32)
                        ptx.inst.mul.f32(ym2, yr, yr)
                        ptx.inst.fma.rn.f32(ym2, yi, yi, ym2)
                        ym = reg.scalar(f32)
                        ptx.inst.sqrt.rn.f32(ym, ym2)

                        residual = reg.scalar(f32)
                        ptx.inst.sub.f32(residual, ym, xm)
                        ptx.inst.add.f32(residual_sum, residual_sum, residual)

                inv_window = reg.scalar(f32, init=1.0 / float(window))
                avg = reg.scalar(f32)
                ptx.inst.mul.f32(avg, residual_sum, inv_window)
                ptx.inst.add.f32(mag, mag_x, avg)
                zero = reg.scalar(f32, init=0.0)
                ptx.inst.max.f32(mag, mag, zero)

            eps_reg = reg.scalar(f32, init=eps)
            denom = reg.scalar(f32)
            ptx.inst.max.f32(denom, mag_y, eps_reg)

            phase_r = reg.scalar(f32)
            phase_i = reg.scalar(f32)
            ptx.inst.div.rn.f32(phase_r, tgt_r, denom)
            ptx.inst.div.rn.f32(phase_i, tgt_i, denom)

            out_r = reg.scalar(f32)
            out_i = reg.scalar(f32)
            ptx.inst.mul.f32(out_r, mag, phase_r)
            ptx.inst.mul.f32(out_i, mag, phase_i)
            ptx.inst.st.global_.f32(ptx.addr(pout + real_base * 4), out_r)
            ptx.inst.st.global_.f32(ptx.addr(pout + imag_base * 4), out_i)

        ptx.ret()

    return progressive_target


def fused_global_layer_norm_pyptx(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Global layer norm over [C,F,T] for each batch item.

    Uses pyptx for the forward pass and a PyTorch backward formula.
    """

    if os.environ.get("FSCNET_ENABLE_PYPTX_NORM") != "1":
        return None
    if not _can_use_global_layer_norm_kernel(x, weight, bias):
        return None
    try:
        return _GlobalLayerNormFn.apply(x.contiguous(), weight.reshape(-1), bias.reshape(-1), eps)
    except Exception:
        return None


def _can_use_global_layer_norm_kernel(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype == torch.float32
        and weight.dtype == torch.float32
        and bias.dtype == torch.float32
        and x.ndim == 4
        and weight.numel() == x.shape[1]
        and bias.numel() == x.shape[1]
    )


class _GlobalLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:
        bsz, channels, freq, frames = x.shape
        kernel = _global_layer_norm_kernel(bsz, channels, freq, frames, float(eps))
        out, mean, rstd = kernel(x, weight.contiguous(), bias.contiguous())
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_out = grad_outputs[0]
        x, weight, mean, rstd = ctx.saved_tensors
        bsz, channels, freq, frames = x.shape
        n = float(channels * freq * frames)

        mean_v = mean.view(bsz, 1, 1, 1)
        rstd_v = rstd.view(bsz, 1, 1, 1)
        weight_v = weight.view(1, channels, 1, 1)
        x_hat = (x - mean_v) * rstd_v

        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_out * x_hat).sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum(dim=(0, 2, 3))

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_normed = grad_out * weight_v
            sum_grad = grad_normed.sum(dim=(1, 2, 3), keepdim=True)
            sum_grad_xhat = (grad_normed * x_hat).sum(
                dim=(1, 2, 3), keepdim=True
            )
            grad_x = (grad_normed - sum_grad / n - x_hat * sum_grad_xhat / n) * rstd_v

        return grad_x, grad_weight, grad_bias, None


@lru_cache(maxsize=32)
def _global_layer_norm_kernel(
    bsz: int, channels: int, freq: int, frames: int, eps: float
):
    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.types import f32, u32

    block = 256
    num_warps = block // 32
    row_size = channels * freq * frames
    channel_stride = freq * frames
    arch = _arch()
    version = (8, 7) if arch.startswith("sm_100") else None

    @kernel(
        in_specs=(
            Tile(bsz, channels, freq, frames, f32),
            Tile(channels, f32),
            Tile(channels, f32),
        ),
        out_specs=(
            Tile(bsz, channels, freq, frames, f32),
            Tile(bsz, f32),
            Tile(bsz, f32),
        ),
        grid=(bsz, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def global_layer_norm(x, weight, bias, out, mean_out, rstd_out):
        partials = smem.alloc(f32, (num_warps, 2))
        stats = smem.alloc(f32, (2, 1))
        px, pw, pb, po, pm, pr = ptx.global_ptrs(
            x, weight, bias, out, mean_out, rstd_out
        )

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        lane = tid & 31
        warp_id = tid >> 5
        row = reg.scalar(u32)
        ptx.inst.mov.u32(row, ptx.special.ctaid.x())
        row_base = row * row_size

        sum_x = reg.scalar(f32, init=0.0)
        sum_x2 = reg.scalar(f32, init=0.0)
        for idx_s in ptx.range_(tid, row_size, block):
            idx = reg.scalar(u32)
            ptx.inst.cvt.u32.s32(idx, idx_s)
            val = reg.scalar(f32)
            ptx.inst.ld.global_.f32(val, ptx.addr(px + (row_base + idx) * 4))
            ptx.inst.add.f32(sum_x, sum_x, val)
            ptx.inst.fma.rn.f32(sum_x2, val, val, sum_x2)

        ptx.warp.reduce_sum(sum_x)
        ptx.warp.reduce_sum(sum_x2)

        with ptx.if_(lane == 0):
            partials[warp_id, 0] = sum_x
            partials[warp_id, 1] = sum_x2
        ptx.bar.sync(0)

        with ptx.if_(tid == 0):
            block_sum = reg.scalar(f32, init=0.0)
            block_sum_sq = reg.scalar(f32, init=0.0)
            for i in range(num_warps):
                ptx.inst.add.f32(block_sum, block_sum, partials[i, 0])
                ptx.inst.add.f32(block_sum_sq, block_sum_sq, partials[i, 1])
            stats[0, 0] = block_sum
            stats[1, 0] = block_sum_sq
        ptx.bar.sync(0)

        ptx.inst.mov.f32(sum_x, stats[0, 0])
        ptx.inst.mov.f32(sum_x2, stats[1, 0])

        inv_n = reg.scalar(f32, init=1.0 / float(row_size))
        eps_reg = reg.scalar(f32, init=eps)
        mean = reg.scalar(f32)
        ptx.inst.mul.f32(mean, sum_x, inv_n)
        ex2 = reg.scalar(f32)
        ptx.inst.mul.f32(ex2, sum_x2, inv_n)
        mean_sq = reg.scalar(f32)
        ptx.inst.mul.f32(mean_sq, mean, mean)
        var = reg.scalar(f32)
        ptx.inst.sub.f32(var, ex2, mean_sq)
        ptx.inst.add.f32(var, var, eps_reg)
        rstd = reg.scalar(f32)
        ptx.inst.rsqrt.approx.f32(rstd, var)

        with ptx.if_(tid == 0):
            ptx.inst.st.global_.f32(ptx.addr(pm + row * 4), mean)
            ptx.inst.st.global_.f32(ptx.addr(pr + row * 4), rstd)

        for idx_s in ptx.range_(tid, row_size, block):
            idx = reg.scalar(u32)
            ptx.inst.cvt.u32.s32(idx, idx_s)
            val = reg.scalar(f32)
            ptx.inst.ld.global_.f32(val, ptx.addr(px + (row_base + idx) * 4))

            chan = reg.scalar(u32)
            ptx.inst.div.u32(chan, idx, channel_stride)
            w = reg.scalar(f32)
            b = reg.scalar(f32)
            ptx.inst.ld.global_.f32(w, ptx.addr(pw + chan * 4))
            ptx.inst.ld.global_.f32(b, ptx.addr(pb + chan * 4))

            centered = reg.scalar(f32)
            ptx.inst.sub.f32(centered, val, mean)
            normed = reg.scalar(f32)
            ptx.inst.mul.f32(normed, centered, rstd)
            out_val = reg.scalar(f32)
            ptx.inst.fma.rn.f32(out_val, normed, w, b)
            ptx.inst.st.global_.f32(ptx.addr(po + (row_base + idx) * 4), out_val)

        ptx.ret()

    return global_layer_norm
