"""Least-squares GAN discriminators for FSC-Net stage outputs.

The article defines per-stage discriminator inputs as
``Z = (waveform, complex spectrogram)`` and cites MelGAN-style multi-scale
discriminators. This module keeps that contract explicit: every scale returns a
waveform branch score and a spectrogram branch score, with feature matching
computed over both branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


FeatureList = List[torch.Tensor]


@dataclass
class DiscriminatorScaleOutput:
    wave_score: torch.Tensor
    spec_score: torch.Tensor
    wave_features: FeatureList
    spec_features: FeatureList


class MelGANWaveDiscriminator(nn.Module):
    """MelGAN-style waveform discriminator for one scale."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 16,
        ]
        strides = [1, 4, 4, 4, 1]
        kernels = [15, 41, 41, 41, 5]
        groups = [1, 4, 8, 16, 16]
        layers: list[nn.Module] = []
        prev = in_channels
        for ch, stride, kernel, group in zip(channels, strides, kernels, groups):
            layers.append(
                nn.Conv1d(
                    prev,
                    ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    groups=min(group, prev),
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = ch
        layers.append(nn.Conv1d(prev, 1, kernel_size=3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, FeatureList]:
        h = _as_waveform_channels(waveform)
        features: FeatureList = []
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                features.append(h)
        return h, features


class PatchSpectrogramDiscriminator(nn.Module):
    """PatchGAN-style RI spectrogram discriminator for one scale."""

    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        strides = [(1, 1), (2, 2), (2, 2), (2, 1)]
        layers: list[nn.Module] = []
        prev = in_channels
        for ch, stride in zip(channels, strides):
            layers.append(
                nn.Conv2d(
                    prev,
                    ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = ch
        layers.append(nn.Conv2d(prev, 1, kernel_size=3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, spec_ri: torch.Tensor) -> Tuple[torch.Tensor, FeatureList]:
        if spec_ri.ndim != 4:
            raise ValueError(
                f"Expected spectrogram [B,2,F,T], got {tuple(spec_ri.shape)}"
            )
        h = spec_ri
        features: FeatureList = []
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                features.append(h)
        return h, features


class ScaleDiscriminator(nn.Module):
    """One paired waveform/spectrogram discriminator scale."""

    def __init__(
        self,
        waveform_channels: int = 1,
        spec_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.wave = MelGANWaveDiscriminator(waveform_channels, base_channels)
        self.spec = PatchSpectrogramDiscriminator(spec_channels, base_channels)

    def forward(
        self, waveform: torch.Tensor, spec_ri: torch.Tensor
    ) -> DiscriminatorScaleOutput:
        wave_score, wave_features = self.wave(waveform)
        spec_score, spec_features = self.spec(spec_ri)
        return DiscriminatorScaleOutput(
            wave_score=wave_score,
            spec_score=spec_score,
            wave_features=wave_features,
            spec_features=spec_features,
        )


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
        self.wave_downsample = nn.AvgPool1d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False
        )

    def forward(
        self, waveform: torch.Tensor, spec_ri: torch.Tensor
    ) -> List[DiscriminatorScaleOutput]:
        outputs: List[DiscriminatorScaleOutput] = []
        h_wave = _as_waveform_channels(waveform)
        h_spec = spec_ri
        for disc in self.discriminators:
            outputs.append(disc(h_wave, h_spec))
            h_wave = self.wave_downsample(h_wave)
            h_spec = _downsample_spec(h_spec)
        return outputs


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


def _score_lsgan_loss(
    real_score: torch.Tensor, fake_score: torch.Tensor
) -> torch.Tensor:
    return torch.mean((real_score - 1.0) ** 2) + torch.mean(fake_score**2)


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
    count = 0
    for real_scale, fake_scale in zip(real_out, fake_out):
        loss = loss + _score_lsgan_loss(real_scale.wave_score, fake_scale.wave_score)
        loss = loss + _score_lsgan_loss(real_scale.spec_score, fake_scale.spec_score)
        count += 2
    return loss / max(1, count)


def _generator_score_loss(score: torch.Tensor) -> torch.Tensor:
    return torch.mean((score - 1.0) ** 2)


def _feature_matching_loss(
    fake_features: FeatureList, real_features: FeatureList
) -> tuple[torch.Tensor, int]:
    if not fake_features:
        raise ValueError("Feature matching requires at least one feature tensor")
    total = fake_features[0].new_tensor(0.0)
    count = 0
    for fake, real in zip(fake_features, real_features):
        total = total + F.l1_loss(fake, real.detach())
        count += 1
    return total, count


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
    score_count = 0
    feature_count = 0
    for fake_scale, real_scale in zip(fake_out, real_out):
        adv = adv + _generator_score_loss(fake_scale.wave_score)
        adv = adv + _generator_score_loss(fake_scale.spec_score)
        score_count += 2

        wave_fm, wave_count = _feature_matching_loss(
            fake_scale.wave_features, real_scale.wave_features
        )
        spec_fm, spec_count = _feature_matching_loss(
            fake_scale.spec_features, real_scale.spec_features
        )
        fm = fm + wave_fm + spec_fm
        feature_count += wave_count + spec_count

    adv = adv / max(1, score_count)
    if feature_count > 0:
        fm = fm / feature_count * fm_weight
    return adv, fm


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)
