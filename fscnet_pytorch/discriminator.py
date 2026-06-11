"""Least-squares GAN discriminators for FSC-Net stage outputs.

The paper uses a per-stage multi-scale discriminator over pairs
``Z = (waveform, complex spectrogram)``. This module follows that interface:
each scale receives a candidate waveform and its corresponding RI spectrogram.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
        waveform_channels: int = 1,
        spec_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        wave_channels = [
            base_channels,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 16,
        ]
        strides = [1, 4, 4, 4, 1]
        kernels = [15, 41, 41, 41, 5]
        wave_layers: list[nn.Module] = []
        prev = waveform_channels
        for ch, stride, kernel in zip(wave_channels, strides, kernels):
            wave_layers.append(
                nn.Conv1d(
                    prev, ch, kernel_size=kernel, stride=stride, padding=kernel // 2
                )
            )
            wave_layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = ch
        self.wave_layers = nn.ModuleList(wave_layers)
        self.spec_layers = nn.ModuleList(
            [
                nn.Conv2d(spec_channels, base_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    base_channels,
                    base_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    base_channels * 4,
                    base_channels * 8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.spec_to_wave = nn.Conv1d(base_channels * 8, prev, kernel_size=1)
        self.score = nn.Conv1d(prev, 1, kernel_size=3, padding=1)

    def forward(
        self, waveform: torch.Tensor, spec_ri: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        h_wave = _as_waveform_channels(waveform)
        for layer in self.wave_layers:
            h_wave = layer(h_wave)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h_wave)

        h_spec = spec_ri
        for layer in self.spec_layers:
            h_spec = layer(h_spec)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h_spec)

        spec_context = F.adaptive_avg_pool2d(h_spec, output_size=(1, 1)).squeeze(-1)
        spec_context = self.spec_to_wave(spec_context)
        h_wave = h_wave + spec_context.expand(-1, -1, h_wave.shape[-1])
        score = self.score(h_wave)
        return score, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self,
        waveform_channels: int = 1,
        spec_channels: int = 2,
        num_scales: int = 3,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(
                    waveform_channels=waveform_channels,
                    spec_channels=spec_channels,
                    base_channels=base_channels,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool1d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False
        )

    def forward(
        self, waveform: torch.Tensor, spec_ri: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outs: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        h_wave = _as_waveform_channels(waveform)
        h_spec = spec_ri
        for disc in self.discriminators:
            outs.append(disc(h_wave, h_spec))
            h_wave = self.downsample(h_wave)
            h_spec = _downsample_spec(h_spec)
        return outs


def _as_waveform_channels(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 2:
        return waveform.unsqueeze(1)
    if waveform.ndim != 3:
        raise ValueError(
            f"Expected waveform [B,T] or [B,C,T], got {tuple(waveform.shape)}"
        )
    return waveform


def _downsample_spec(spec_ri: torch.Tensor) -> torch.Tensor:
    if spec_ri.shape[-1] <= 1 or spec_ri.shape[-2] <= 1:
        return spec_ri
    return F.avg_pool2d(spec_ri, kernel_size=2, stride=2, count_include_pad=False)


def discriminator_lsgan_loss(
    disc: MultiScaleDiscriminator,
    real_waveform: torch.Tensor,
    real_spec_ri: torch.Tensor,
    fake_waveform: torch.Tensor,
    fake_spec_ri: torch.Tensor,
) -> torch.Tensor:
    real_out = disc(real_waveform, real_spec_ri)
    fake_out = disc(fake_waveform.detach(), fake_spec_ri.detach())
    loss = real_waveform.new_tensor(0.0)
    for (real_score, _), (fake_score, _) in zip(real_out, fake_out):
        loss = loss + torch.mean((real_score - 1.0) ** 2) + torch.mean(fake_score**2)
    return loss / len(real_out)


def generator_lsgan_fm_loss(
    disc: MultiScaleDiscriminator,
    real_waveform: torch.Tensor,
    real_spec_ri: torch.Tensor,
    fake_waveform: torch.Tensor,
    fake_spec_ri: torch.Tensor,
    fm_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_out = disc(fake_waveform, fake_spec_ri)
    with torch.no_grad():
        real_out = disc(real_waveform, real_spec_ri)

    adv = fake_waveform.new_tensor(0.0)
    fm = fake_waveform.new_tensor(0.0)
    n_feats = 0
    for (fake_score, fake_feats), (real_score, real_feats) in zip(fake_out, real_out):
        adv = adv + torch.mean((fake_score - 1.0) ** 2)
        for ff, rf in zip(fake_feats, real_feats):
            fm = fm + F.l1_loss(ff, rf.detach())
            n_feats += 1
    adv = adv / len(fake_out)
    if n_feats > 0:
        fm = fm / n_feats * fm_weight
    return adv, fm


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)
