"""Least-squares GAN discriminators for FSC-Net stage outputs.

The paper uses a per-stage multi-scale discriminator. This module implements a
conditional waveform multi-scale discriminator: each discriminator receives the
upsampled narrow-band waveform concatenated with either a generated or target
stage waveform.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ScaleDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
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
        layers: list[nn.Module] = []
        prev = in_channels
        for ch, stride, kernel in zip(channels, strides, kernels):
            layers.append(
                nn.Conv1d(
                    prev, ch, kernel_size=kernel, stride=stride, padding=kernel // 2
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = ch
        layers.append(nn.Conv1d(prev, 1, kernel_size=3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(h)
        score = h
        return score, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self, in_channels: int = 2, num_scales: int = 3, base_channels: int = 16
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(in_channels=in_channels, base_channels=base_channels)
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool1d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outs: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        h = x
        for disc in self.discriminators:
            outs.append(disc(h))
            h = self.downsample(h)
        return outs


def _conditional_pair(condition: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if condition.ndim == 2:
        condition = condition.unsqueeze(1)
    if candidate.ndim == 2:
        candidate = candidate.unsqueeze(1)
    if condition.shape[-1] != candidate.shape[-1]:
        length = min(condition.shape[-1], candidate.shape[-1])
        condition = condition[..., :length]
        candidate = candidate[..., :length]
    return torch.cat((condition, candidate), dim=1)


def discriminator_lsgan_loss(
    disc: MultiScaleDiscriminator,
    condition: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    real_out = disc(_conditional_pair(condition, real))
    fake_out = disc(_conditional_pair(condition, fake.detach()))
    loss = real.new_tensor(0.0)
    for (real_score, _), (fake_score, _) in zip(real_out, fake_out):
        loss = loss + torch.mean((real_score - 1.0) ** 2) + torch.mean(fake_score**2)
    return loss / len(real_out)


def generator_lsgan_fm_loss(
    disc: MultiScaleDiscriminator,
    condition: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    fm_weight: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_out = disc(_conditional_pair(condition, fake))
    with torch.no_grad():
        real_out = disc(_conditional_pair(condition, real))

    adv = fake.new_tensor(0.0)
    fm = fake.new_tensor(0.0)
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
