"""Audio and STFT helpers for the FSC-Net PyTorch implementation.

The scripts prefer torchaudio when it is installed and ABI-compatible, but fall
back to soundfile + scipy for I/O and resampling. Model internals use only
PyTorch STFT/ISTFT.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # torchaudio is convenient but optional.
    import torchaudio  # type: ignore

    _TORCHAUDIO_OK = True
except Exception:  # pragma: no cover - depends on local environment
    torchaudio = None  # type: ignore
    _TORCHAUDIO_OK = False

try:
    import soundfile as sf  # type: ignore
except Exception as exc:  # pragma: no cover
    sf = None  # type: ignore
    _SF_IMPORT_ERROR = exc
else:
    _SF_IMPORT_ERROR = None

try:
    from scipy.signal import resample_poly  # type: ignore
except Exception as exc:  # pragma: no cover
    resample_poly = None  # type: ignore
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def ensure_2d_audio(wav: torch.Tensor) -> torch.Tensor:
    """Return audio shaped [channels, samples]."""
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        return wav
    raise ValueError(f"Expected [T] or [C,T] audio, got {tuple(wav.shape)}")


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    return ensure_2d_audio(wav).mean(dim=0)


def resample_audio(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Resample a [T] or [C,T] tensor for data preparation / inference."""
    if orig_sr == new_sr:
        return wav
    if orig_sr <= 0 or new_sr <= 0:
        raise ValueError(f"Invalid sample rates: {orig_sr} -> {new_sr}")

    was_1d = wav.ndim == 1
    wav_2d = ensure_2d_audio(wav)

    if _TORCHAUDIO_OK:
        out = torchaudio.functional.resample(wav_2d, orig_sr, new_sr)
        return out.squeeze(0) if was_1d else out

    if resample_poly is None:  # pragma: no cover
        raise RuntimeError(
            "Need either torchaudio or scipy for resampling. "
            f"scipy import error: {_SCIPY_IMPORT_ERROR!r}"
        )
    x = wav_2d.detach().cpu().numpy()
    gcd = math.gcd(orig_sr, new_sr)
    y = resample_poly(x, new_sr // gcd, orig_sr // gcd, axis=-1).astype(
        np.float32, copy=False
    )
    out = torch.from_numpy(y).to(dtype=wav.dtype)
    return out.squeeze(0) if was_1d else out


def load_audio(
    path: str | Path, target_sr: Optional[int] = None, mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load audio as float32. Returns [T] if mono else [C,T], and sample rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if _TORCHAUDIO_OK:
        try:
            wav, sr = torchaudio.load(str(path))
            wav = wav.to(torch.float32)
        except Exception:
            wav = None
    else:
        wav = None

    if wav is None:
        if sf is None:  # pragma: no cover
            raise RuntimeError(
                f"soundfile is required for audio I/O: {_SF_IMPORT_ERROR!r}"
            )
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = torch.from_numpy(data.T.copy())

    if mono:
        wav = to_mono(wav)
    if target_sr is not None and sr != target_sr:
        wav = resample_audio(wav, sr, target_sr)
        sr = target_sr
    return wav.contiguous(), sr


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    """Save audio with clipping to [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().to(torch.float32)
    wav = torch.nan_to_num(wav).clamp(-1.0, 1.0)
    wav_2d = ensure_2d_audio(wav)

    if _TORCHAUDIO_OK:
        try:
            torchaudio.save(str(path), wav_2d, sample_rate)
            return
        except Exception:
            pass
    if sf is None:  # pragma: no cover
        raise RuntimeError(f"soundfile is required for audio I/O: {_SF_IMPORT_ERROR!r}")
    sf.write(str(path), wav_2d.T.numpy(), sample_rate)


def match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    """Pad or crop the last dimension to a target length."""
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]))


def stft_complex(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    center: bool = True,
) -> torch.Tensor:
    """STFT for [B,T] or [T] waveforms. Returns complex [B,F,frames]."""
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"Expected [B,T] or [T], got {tuple(wav.shape)}")
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )


def istft_complex(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    length: Optional[int] = None,
    center: bool = True,
) -> torch.Tensor:
    """ISTFT for complex [B,F,frames] spectrograms. Returns [B,T]."""
    if spec.ndim != 3 or not torch.is_complex(spec):
        raise ValueError(
            f"Expected complex [B,F,T], got shape={tuple(spec.shape)}, dtype={spec.dtype}"
        )
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )


def complex_to_ri(spec: torch.Tensor) -> torch.Tensor:
    """Complex [B,F,T] -> real/imag [B,2,F,T]."""
    if not torch.is_complex(spec):
        raise ValueError("complex_to_ri expects a complex tensor")
    return torch.stack((spec.real, spec.imag), dim=1)


def ri_to_complex(ri: torch.Tensor) -> torch.Tensor:
    """Real/imag [B,2,F,T] -> complex [B,F,T]."""
    if ri.ndim != 4 or ri.shape[1] != 2:
        raise ValueError(f"Expected [B,2,F,T], got {tuple(ri.shape)}")
    return torch.complex(ri[:, 0], ri[:, 1])


def peak_normalize_pair(
    lr: torch.Tensor, hr: torch.Tensor, eps: float = 1.0e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    peak = torch.maximum(lr.abs().amax(), hr.abs().amax()).clamp_min(eps)
    return lr / peak, hr / peak
