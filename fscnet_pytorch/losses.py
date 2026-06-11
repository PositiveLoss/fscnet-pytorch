"""Losses and progressive targets for FSC-Net training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .audio import istft_complex, ri_to_complex, stft_complex
from .model import FSCNetConfig


def sliding_average_frequency(x: torch.Tensor, window: int) -> torch.Tensor:
    """Average a [B,F,T] tensor along frequency with zero padding."""
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError("Progressive smoothing windows should be odd")
    bsz, freq, frames = x.shape
    y = x.permute(0, 2, 1).reshape(bsz * frames, 1, freq)
    y = F.avg_pool1d(
        y, kernel_size=window, stride=1, padding=window // 2, count_include_pad=True
    )
    return y.reshape(bsz, frames, freq).permute(0, 2, 1)


def make_progressive_targets(
    input_ri: torch.Tensor,
    target_ri: torch.Tensor,
    windows: Sequence[int],
    eps: float = 1.0e-8,
) -> List[torch.Tensor]:
    """Build the stage targets described by the article.

    The target magnitude at a stage is:
        |X_lr| + avg_freq_window(|Y_hr| - |X_lr|)
    and the target phase is the clean HR phase. W=1 recovers the exact HR
    complex spectrogram.
    """
    x = ri_to_complex(input_ri)
    y = ri_to_complex(target_ri)
    mag_x = x.abs()
    mag_y = y.abs()
    phase_y = y / mag_y.clamp_min(eps)
    residual = mag_y - mag_x

    targets: List[torch.Tensor] = []
    for window in windows:
        if window <= 1:
            mag = mag_y
        else:
            mag = (mag_x + sliding_average_frequency(residual, int(window))).clamp_min(
                0.0
            )
        spec = mag * phase_y
        targets.append(torch.stack((spec.real, spec.imag), dim=1))
    return targets


class MultiResolutionSTFTLoss(nn.Module):
    """Average spectral-convergence + log-magnitude loss over FFT sizes."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_ratio: float = 0.25,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        self.fft_sizes = tuple(int(v) for v in fft_sizes)
        self.hop_ratio = hop_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        total = pred.new_tensor(0.0)
        for n_fft in self.fft_sizes:
            hop = max(1, int(round(n_fft * self.hop_ratio)))
            win = n_fft
            pred_spec = stft_complex(pred, n_fft=n_fft, hop_length=hop, win_length=win)
            target_spec = stft_complex(
                target, n_fft=n_fft, hop_length=hop, win_length=win
            )
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()
            sc = torch.linalg.vector_norm(
                pred_mag - target_mag
            ) / torch.linalg.vector_norm(target_mag).clamp_min(self.eps)
            log_mag = F.l1_loss(
                torch.log(pred_mag + self.eps), torch.log(target_mag + self.eps)
            )
            total = total + sc + log_mag
        return total / len(self.fft_sizes)


class LogSpectralDistance(nn.Module):
    """Log-spectral distance in dB."""

    def __init__(
        self,
        n_fft: int = 1536,
        hop_length: int = 768,
        win_length: int = 1536,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_spec = stft_complex(pred, self.n_fft, self.hop_length, self.win_length)
        target_spec = stft_complex(target, self.n_fft, self.hop_length, self.win_length)
        pred_db = 20.0 * torch.log10(pred_spec.abs().clamp_min(self.eps))
        target_db = 20.0 * torch.log10(target_spec.abs().clamp_min(self.eps))
        # Mean over time, sqrt over frequency, then batch mean.
        return torch.sqrt(
            torch.mean((pred_db - target_db) ** 2, dim=1).clamp_min(self.eps)
        ).mean()


def complex_l1(pred_ri: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_ri, target_ri)


@dataclass
class StageLossWeights:
    mrstft: float = 1.0
    lsd: float = 0.1
    complex_l1: float = 1.0


class StageReconstructionLoss(nn.Module):
    """Reconstruction objective aggregated over progressive stage outputs."""

    def __init__(
        self,
        cfg: FSCNetConfig,
        windows: Sequence[int],
        weights: StageLossWeights = StageLossWeights(),
        mrstft_fft_sizes: Sequence[int] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.windows = tuple(int(v) for v in windows)
        self.weights = weights
        self.mrstft = MultiResolutionSTFTLoss(mrstft_fft_sizes)
        self.lsd = LogSpectralDistance(cfg.n_fft, cfg.hop_length, cfg.win_length)

    def _wav_from_ri(self, ri: torch.Tensor, length: int) -> torch.Tensor:
        return istft_complex(
            ri_to_complex(ri),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            length=length,
            center=self.cfg.center,
        )

    def forward(
        self,
        pred_stages_ri: Sequence[torch.Tensor],
        input_ri: torch.Tensor,
        target_ri: torch.Tensor,
        waveform_length: int,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
        if len(pred_stages_ri) != len(self.windows):
            raise ValueError(
                f"Got {len(pred_stages_ri)} predictions but {len(self.windows)} windows"
            )
        target_stages_ri = make_progressive_targets(input_ri, target_ri, self.windows)
        total = target_ri.new_tensor(0.0)
        logs: Dict[str, float] = {}
        pred_wavs: List[torch.Tensor] = []
        target_wavs: List[torch.Tensor] = []

        for idx, (pred_ri, target_stage_ri) in enumerate(
            zip(pred_stages_ri, target_stages_ri)
        ):
            pred_wav = self._wav_from_ri(pred_ri, waveform_length)
            target_wav = self._wav_from_ri(target_stage_ri, waveform_length)
            pred_wavs.append(pred_wav)
            target_wavs.append(target_wav)

            loss = target_ri.new_tensor(0.0)
            if self.weights.mrstft:
                loss = loss + self.weights.mrstft * self.mrstft(pred_wav, target_wav)
            if self.weights.lsd:
                loss = loss + self.weights.lsd * self.lsd(pred_wav, target_wav)
            if self.weights.complex_l1:
                loss = loss + self.weights.complex_l1 * complex_l1(
                    pred_ri, target_stage_ri
                )
            total = total + loss
            logs[f"stage_{idx + 1}_loss"] = float(loss.detach().cpu())

        total = total / len(pred_stages_ri)
        logs["recon_loss"] = float(total.detach().cpu())
        return total, logs, pred_wavs, target_wavs
