"""Optional pyptx kernels for training hot paths."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

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
