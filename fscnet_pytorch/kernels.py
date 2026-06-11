"""Optional pyptx kernels for training hot paths."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Sequence

import torch


_ACTIVATED_KERNELS: set[str] = set()
_KERNEL_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _mark_kernel_active(name: str) -> None:
    _ACTIVATED_KERNELS.add(name)


def activated_kernel_names() -> tuple[str, ...]:
    """Return optional pyptx kernels that have executed in this process."""

    return tuple(sorted(_ACTIVATED_KERNELS))


def reset_activated_kernel_names() -> None:
    """Clear runtime kernel activation state."""

    _ACTIVATED_KERNELS.clear()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _is_half_dtype_name(dtype_name: str) -> bool:
    return dtype_name in ("torch.float16", "torch.bfloat16")


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
        and input_ri.dtype in _KERNEL_DTYPES
        and target_ri.dtype == input_ri.dtype
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
        targets = [
            _progressive_target_kernel(
                bsz,
                freq,
                frames,
                int(window),
                float(eps),
                _dtype_name(input_c.dtype),
            )(
                input_c, target_c
            )
            for window in windows
        ]
        _mark_kernel_active("pyptx_progressive_targets")
        return targets
    except Exception:
        return None


@lru_cache(maxsize=64)
def _progressive_target_kernel(
    bsz: int, freq: int, frames: int, window: int, eps: float, dtype_name: str
):
    from pyptx import Tile, kernel, ptx, reg
    from pyptx.types import b16, bf16, f16, f32, u32

    block = 128
    grid = ((frames + block - 1) // block, freq, bsz)
    radius = window // 2
    arch = _arch()
    version = (8, 7) if arch.startswith("sm_100") else None
    data_t = (
        bf16
        if dtype_name == "torch.bfloat16"
        else f16
        if dtype_name == "torch.float16"
        else f32
    )
    elem_size = 2 if _is_half_dtype_name(dtype_name) else 4

    def load_data(dst, ptr, elem_idx):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.bf16(dst, tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.f16(dst, tmp)
        else:
            ptx.inst.ld.global_.f32(dst, ptx.addr(ptr + elem_idx * elem_size))

    def store_data(ptr, elem_idx, value):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.bf16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.f16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr + elem_idx * elem_size), value)

    @kernel(
        in_specs=(
            Tile(bsz, 2, freq, frames, data_t),
            Tile(bsz, 2, freq, frames, data_t),
        ),
        out_specs=(Tile(bsz, 2, freq, frames, data_t),),
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
            load_data(in_r, pin, real_base)
            load_data(in_i, pin, imag_base)
            load_data(tgt_r, ptgt, real_base)
            load_data(tgt_i, ptgt, imag_base)

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
                        load_data(xr, pin, r_base)
                        load_data(xi, pin, i_base)
                        load_data(yr, ptgt, r_base)
                        load_data(yi, ptgt, i_base)

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
            store_data(pout, real_base, out_r)
            store_data(pout, imag_base, out_i)

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
        out = _GlobalLayerNormFn.apply(
            x.contiguous(), weight.reshape(-1), bias.reshape(-1), eps
        )
        _mark_kernel_active("pyptx_global_layer_norm")
        return out
    except Exception:
        return None


def _can_use_global_layer_norm_kernel(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and bias.is_cuda
        and x.dtype in _KERNEL_DTYPES
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
        kernel = _global_layer_norm_kernel(
            bsz, channels, freq, frames, float(eps), _dtype_name(x.dtype)
        )
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
        compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

        mean_v = mean.view(bsz, 1, 1, 1)
        rstd_v = rstd.view(bsz, 1, 1, 1)
        weight_v = weight.view(1, channels, 1, 1)
        x_compute = x.to(compute_dtype)
        grad_out_compute = grad_out.to(compute_dtype)
        x_hat = (x_compute - mean_v) * rstd_v

        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_out_compute * x_hat).sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out_compute.sum(dim=(0, 2, 3))

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_normed = grad_out_compute * weight_v
            sum_grad = grad_normed.sum(dim=(1, 2, 3), keepdim=True)
            sum_grad_xhat = (grad_normed * x_hat).sum(
                dim=(1, 2, 3), keepdim=True
            )
            grad_x = (grad_normed - sum_grad / n - x_hat * sum_grad_xhat / n) * rstd_v
            grad_x = grad_x.to(x.dtype)

        return grad_x, grad_weight, grad_bias, None


@lru_cache(maxsize=32)
def _global_layer_norm_kernel(
    bsz: int, channels: int, freq: int, frames: int, eps: float, dtype_name: str
):
    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.types import b16, bf16, f16, f32, u32

    block = 256
    num_warps = block // 32
    row_size = channels * freq * frames
    channel_stride = freq * frames
    arch = _arch()
    version = (8, 7) if arch.startswith("sm_100") else None
    data_t = (
        bf16
        if dtype_name == "torch.bfloat16"
        else f16
        if dtype_name == "torch.float16"
        else f32
    )
    elem_size = 2 if _is_half_dtype_name(dtype_name) else 4

    def load_x(dst, ptr, elem_idx):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.bf16(dst, tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.f16(dst, tmp)
        else:
            ptx.inst.ld.global_.f32(dst, ptx.addr(ptr + elem_idx * elem_size))

    def store_out(ptr, elem_idx, value):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.bf16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.f16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr + elem_idx * elem_size), value)

    @kernel(
        in_specs=(
            Tile(bsz, channels, freq, frames, data_t),
            Tile(channels, f32),
            Tile(channels, f32),
        ),
        out_specs=(
            Tile(bsz, channels, freq, frames, data_t),
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
            load_x(val, px, row_base + idx)
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
            load_x(val, px, row_base + idx)

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
            store_out(po, row_base + idx, out_val)

        ptx.ret()

    return global_layer_norm


def fused_rope_qk_norm_pyptx(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply RoPE and QK normalization to [Bf,H,T,D] q/k tensors."""

    if os.environ.get("FSCNET_ENABLE_PYPTX_ROPE_QK") != "1":
        return None
    if not _can_use_rope_qk_norm_kernel(q, k):
        return None
    try:
        frames = q.shape[-2]
        half = q.shape[-1] // 2
        pos = torch.arange(frames, device=q.device, dtype=torch.float32)
        freq = torch.arange(half, device=q.device, dtype=torch.float32)
        inv_freq = 1.0 / (10_000 ** (freq / float(half)))
        angles = pos[:, None] * inv_freq[None, :]
        sin = angles.sin().to(dtype=q.dtype).contiguous()
        cos = angles.cos().to(dtype=q.dtype).contiguous()
        out = _RoPEQKNormFn.apply(
            q.contiguous(), k.contiguous(), sin, cos, float(eps)
        )
        _mark_kernel_active("pyptx_rope_qk_norm")
        return out
    except Exception:
        return None


def _can_use_rope_qk_norm_kernel(q: torch.Tensor, k: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and k.is_cuda
        and q.dtype in _KERNEL_DTYPES
        and k.dtype == q.dtype
        and q.ndim == 4
        and k.shape == q.shape
        and q.shape[-1] % 2 == 0
        and q.shape[-1] <= 32
    )


class _RoPEQKNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        eps: float,
    ) -> Any:
        rows, heads, frames, dim_head = q.shape
        kernel = _rope_qk_norm_kernel(
            rows, heads, frames, dim_head, float(eps), _dtype_name(q.dtype)
        )
        q_out, k_out = kernel(q, k, sin, cos)
        ctx.save_for_backward(q, k)
        ctx.eps = eps
        return q_out, k_out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_q_out, grad_k_out = grad_outputs
        q, k = ctx.saved_tensors
        eps = float(ctx.eps)
        scale = q.shape[-1] ** 0.5
        grad_q = _rope_qk_norm_backward(q, grad_q_out, scale, eps)
        grad_k = _rope_qk_norm_backward(k, grad_k_out, scale, eps)
        return grad_q, grad_k, None, None, None


def _rope_qk_norm_backward(
    x: torch.Tensor, grad_out: torch.Tensor, scale: float, eps: float
) -> torch.Tensor:
    out_dtype = x.dtype
    x_compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    grad_compute = (
        grad_out.float()
        if grad_out.dtype in (torch.float16, torch.bfloat16)
        else grad_out
    )
    z = _apply_rope_torch(x_compute)
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    grad_z = scale * (
        grad_compute / norm
        - z * (grad_compute * z).sum(dim=-1, keepdim=True) / norm.pow(3)
    )
    return _apply_inverse_rope_torch(grad_z).to(out_dtype)


def _apply_rope_torch(x: torch.Tensor) -> torch.Tensor:
    frames = x.shape[-2]
    dim_head = x.shape[-1]
    half = dim_head // 2
    pos = torch.arange(frames, device=x.device, dtype=torch.float32)
    freq = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10_000 ** (freq / float(half)))
    angles = pos[:, None] * inv_freq[None, :]
    sin = angles.sin().to(dtype=x.dtype).view(1, 1, frames, half)
    cos = angles.cos().to(dtype=x.dtype).view(1, 1, frames, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _apply_inverse_rope_torch(x: torch.Tensor) -> torch.Tensor:
    frames = x.shape[-2]
    dim_head = x.shape[-1]
    half = dim_head // 2
    pos = torch.arange(frames, device=x.device, dtype=torch.float32)
    freq = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10_000 ** (freq / float(half)))
    angles = pos[:, None] * inv_freq[None, :]
    sin = angles.sin().to(dtype=x.dtype).view(1, 1, frames, half)
    cos = angles.cos().to(dtype=x.dtype).view(1, 1, frames, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)


@lru_cache(maxsize=64)
def _rope_qk_norm_kernel(
    rows: int, heads: int, frames: int, dim_head: int, eps: float, dtype_name: str
):
    from pyptx import Tile, kernel, ptx, reg, smem
    from pyptx.types import b16, bf16, f16, f32, u32

    block = 32
    half = dim_head // 2
    vectors = rows * heads * frames
    scale = dim_head**0.5
    arch = _arch()
    version = (8, 7) if arch.startswith("sm_100") else None
    data_t = (
        bf16
        if dtype_name == "torch.bfloat16"
        else f16
        if dtype_name == "torch.float16"
        else f32
    )
    elem_size = 2 if _is_half_dtype_name(dtype_name) else 4

    def load_data(dst, ptr, elem_idx):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.bf16(dst, tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.ld.global_.b16(tmp, ptx.addr(ptr + elem_idx * elem_size))
            ptx.inst.cvt.f32.f16(dst, tmp)
        else:
            ptx.inst.ld.global_.f32(dst, ptx.addr(ptr + elem_idx * elem_size))

    def store_data(ptr, elem_idx, value):
        if dtype_name == "torch.bfloat16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.bf16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        elif dtype_name == "torch.float16":
            tmp = reg.scalar(b16)
            ptx.inst.cvt.rn.f16.f32(tmp, value)
            ptx.inst.st.global_.b16(ptx.addr(ptr + elem_idx * elem_size), tmp)
        else:
            ptx.inst.st.global_.f32(ptx.addr(ptr + elem_idx * elem_size), value)

    @kernel(
        in_specs=(
            Tile(rows, heads, frames, dim_head, data_t),
            Tile(rows, heads, frames, dim_head, data_t),
            Tile(frames, half, data_t),
            Tile(frames, half, data_t),
        ),
        out_specs=(
            Tile(rows, heads, frames, dim_head, data_t),
            Tile(rows, heads, frames, dim_head, data_t),
        ),
        grid=(vectors, 1, 1),
        block=(block, 1, 1),
        arch=arch,
        version=version,
    )
    def rope_qk_norm(q, k, sin_table, cos_table, q_out, k_out):
        stats = smem.alloc(f32, (2, 1))
        pq, pk, psin, pcos, pqo, pko = ptx.global_ptrs(
            q, k, sin_table, cos_table, q_out, k_out
        )

        tid = reg.scalar(u32)
        ptx.inst.mov.u32(tid, ptx.special.tid.x())
        vector = reg.scalar(u32)
        ptx.inst.mov.u32(vector, ptx.special.ctaid.x())

        tmp = reg.scalar(u32)
        ptx.inst.div.u32(tmp, vector, frames)
        frame_mul = reg.scalar(u32)
        ptx.inst.mul.lo.u32(frame_mul, tmp, frames)
        frame = reg.scalar(u32)
        ptx.inst.sub.u32(frame, vector, frame_mul)

        head_base = reg.scalar(u32)
        ptx.inst.div.u32(head_base, tmp, heads)
        head_mul = reg.scalar(u32)
        ptx.inst.mul.lo.u32(head_mul, head_base, heads)
        head = reg.scalar(u32)
        ptx.inst.sub.u32(head, tmp, head_mul)
        row = head_base

        vec_base = (((row * heads + head) * frames + frame) * dim_head)
        pair_idx = tid

        sin_val = reg.scalar(f32)
        cos_val = reg.scalar(f32)

        q_a = reg.scalar(f32, init=0.0)
        q_b = reg.scalar(f32, init=0.0)
        k_a = reg.scalar(f32, init=0.0)
        k_b = reg.scalar(f32, init=0.0)
        q_norm_sum = reg.scalar(f32, init=0.0)
        k_norm_sum = reg.scalar(f32, init=0.0)

        with ptx.if_(tid < half):
            trig_idx = frame * half + pair_idx
            load_data(sin_val, psin, trig_idx)
            load_data(cos_val, pcos, trig_idx)
            low_idx = pair_idx
            high_idx = pair_idx + half
            q_low = reg.scalar(f32)
            q_high = reg.scalar(f32)
            k_low = reg.scalar(f32)
            k_high = reg.scalar(f32)
            load_data(q_low, pq, vec_base + low_idx)
            load_data(q_high, pq, vec_base + high_idx)
            load_data(k_low, pk, vec_base + low_idx)
            load_data(k_high, pk, vec_base + high_idx)

            neg_sin = reg.scalar(f32, init=0.0)
            ptx.inst.sub.f32(neg_sin, neg_sin, sin_val)
            ptx.inst.mul.f32(q_a, q_low, cos_val)
            ptx.inst.fma.rn.f32(q_a, q_high, neg_sin, q_a)
            ptx.inst.mul.f32(q_b, q_low, sin_val)
            ptx.inst.fma.rn.f32(q_b, q_high, cos_val, q_b)
            ptx.inst.mul.f32(k_a, k_low, cos_val)
            ptx.inst.fma.rn.f32(k_a, k_high, neg_sin, k_a)
            ptx.inst.mul.f32(k_b, k_low, sin_val)
            ptx.inst.fma.rn.f32(k_b, k_high, cos_val, k_b)

            ptx.inst.fma.rn.f32(q_norm_sum, q_a, q_a, q_norm_sum)
            ptx.inst.fma.rn.f32(q_norm_sum, q_b, q_b, q_norm_sum)
            ptx.inst.fma.rn.f32(k_norm_sum, k_a, k_a, k_norm_sum)
            ptx.inst.fma.rn.f32(k_norm_sum, k_b, k_b, k_norm_sum)

        ptx.warp.reduce_sum(q_norm_sum)
        ptx.warp.reduce_sum(k_norm_sum)

        with ptx.if_(tid == 0):
            eps_sq_reg = reg.scalar(f32, init=eps * eps)
            ptx.inst.max.f32(q_norm_sum, q_norm_sum, eps_sq_reg)
            ptx.inst.max.f32(k_norm_sum, k_norm_sum, eps_sq_reg)
            q_rnorm = reg.scalar(f32)
            k_rnorm = reg.scalar(f32)
            ptx.inst.rsqrt.approx.f32(q_rnorm, q_norm_sum)
            ptx.inst.rsqrt.approx.f32(k_rnorm, k_norm_sum)
            stats[0, 0] = q_rnorm
            stats[1, 0] = k_rnorm
        ptx.bar.sync(0)

        q_rnorm = reg.scalar(f32)
        k_rnorm = reg.scalar(f32)
        ptx.inst.mov.f32(q_rnorm, stats[0, 0])
        ptx.inst.mov.f32(k_rnorm, stats[1, 0])
        scale_reg = reg.scalar(f32, init=scale)

        with ptx.if_(tid < half):
            low_idx = pair_idx
            high_idx = pair_idx + half
            out_q_low = reg.scalar(f32)
            out_q_high = reg.scalar(f32)
            out_k_low = reg.scalar(f32)
            out_k_high = reg.scalar(f32)
            ptx.inst.mul.f32(out_q_low, q_a, q_rnorm)
            ptx.inst.mul.f32(out_q_low, out_q_low, scale_reg)
            ptx.inst.mul.f32(out_q_high, q_b, q_rnorm)
            ptx.inst.mul.f32(out_q_high, out_q_high, scale_reg)
            ptx.inst.mul.f32(out_k_low, k_a, k_rnorm)
            ptx.inst.mul.f32(out_k_low, out_k_low, scale_reg)
            ptx.inst.mul.f32(out_k_high, k_b, k_rnorm)
            ptx.inst.mul.f32(out_k_high, out_k_high, scale_reg)
            store_data(pqo, vec_base + low_idx, out_q_low)
            store_data(pqo, vec_base + high_idx, out_q_high)
            store_data(pko, vec_base + low_idx, out_k_low)
            store_data(pko, vec_base + high_idx, out_k_high)

        ptx.ret()

    return rope_qk_norm
