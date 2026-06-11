"""FSC-Net style generator.

This is a runnable implementation of the architecture described in arXiv:2606.06962v1:
complex spectral mapping, channel-wise subband processing, TF-FFC blocks,
Fast Fourier Convolution branches, intra-frequency BLSTM, and stage outputs
for frequency-progressive learning.

The paper does not release exact source code or every hyperparameter, so the
module exposes the architectural choices as config fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Tuple, overload

import torch
from torch import nn
import torch.nn.functional as F

from .audio import complex_to_ri, istft_complex, ri_to_complex, stft_complex
from .kernels import fused_global_layer_norm_pyptx
from .validation import validate_fscnet_config_data


@dataclass
class FSCNetConfig:
    target_sr: int = 48_000
    input_sr: int = 4_000
    n_fft: int = 1536  # 32 ms at 48 kHz
    win_length: int = 1536
    hop_length: int = 768  # 16 ms at 48 kHz
    subbands: int = 3
    channels: int = 48
    num_blocks: int = 5
    ffc_ratio: float = 0.5
    attention_heads: int = 4
    time_attention: str = "v1"
    time_attention_qk_norm: bool = True
    time_attention_rope: bool = True
    rnn_hidden: int = 64
    dropout: float = 0.0
    center: bool = True

    def to_dict(self) -> Dict[str, int | float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "FSCNetConfig":
        valid = validate_fscnet_config_data(data, cls.__dataclass_fields__)
        return cls(**valid)


class GlobalLayerNorm(nn.Module):
    """Global layer normalization for [B,C,F,T] feature maps."""

    def __init__(self, channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_global_layer_norm_pyptx(x, self.weight, self.bias, self.eps)
        if out is not None:
            return out
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SingleGroupNorm(nn.Module):
    """GroupNorm(num_groups=1) expressed with primitive tensor ops."""

    def __init__(self, channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_global_layer_norm_pyptx(x, self.weight, self.bias, self.eps)
        if out is not None:
            return out
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return (x - mean) * torch.rsqrt(var + self.eps) * weight + bias


class SpectralTransform(nn.Module):
    """Global branch used inside Fast Fourier Convolution.

    It performs rFFT2 over the time-frequency feature map, applies lightweight
    convolution to concatenated real/imaginary Fourier coefficients, then uses
    inverse rFFT2 to return to the feature domain.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("SpectralTransform requires channels > 0")
        self.export_manual_fft = False
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
            SingleGroupNorm(channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1),
        )

    @staticmethod
    def _dft_basis(
        size: int, freqs: int | None, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = size if freqs is None else freqs
        n = torch.arange(size, device=device, dtype=dtype)
        k = torch.arange(freqs, device=device, dtype=dtype)
        angle = 2.0 * torch.pi * n[:, None] * k[None, :] / float(size)
        scale = float(size) ** -0.5
        return torch.cos(angle) * scale, torch.sin(angle) * scale

    @classmethod
    def _manual_rfft2(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2], x.shape[-1]
        dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        x = x.to(dtype)
        cos_w, sin_w = cls._dft_basis(width, width // 2 + 1, x.device, dtype)
        real = torch.einsum("bchw,wk->bchk", x, cos_w)
        imag = -torch.einsum("bchw,wk->bchk", x, sin_w)

        cos_h, sin_h = cls._dft_basis(height, None, x.device, dtype)
        out_real = torch.einsum("bchk,hf->bcfk", real, cos_h) + torch.einsum(
            "bchk,hf->bcfk", imag, sin_h
        )
        out_imag = torch.einsum("bchk,hf->bcfk", imag, cos_h) - torch.einsum(
            "bchk,hf->bcfk", real, sin_h
        )
        return out_real, out_imag

    @classmethod
    def _manual_irfft2(
        cls, real: torch.Tensor, imag: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        dtype = real.dtype
        cos_h, sin_h = cls._dft_basis(height, None, real.device, dtype)
        time_real = torch.einsum("bcfk,hf->bchk", real, cos_h) - torch.einsum(
            "bcfk,hf->bchk", imag, sin_h
        )
        time_imag = torch.einsum("bcfk,hf->bchk", real, sin_h) + torch.einsum(
            "bcfk,hf->bchk", imag, cos_h
        )

        freqs = width // 2 + 1
        cos_w, sin_w = cls._dft_basis(width, freqs, real.device, dtype)
        weights = torch.ones(freqs, device=real.device, dtype=dtype)
        if freqs > 1:
            end = freqs - 1 if width % 2 == 0 else freqs
            if end > 1:
                weights[1:end] = 2.0
        time_real = time_real * weights.view(1, 1, 1, -1)
        time_imag = time_imag * weights.view(1, 1, 1, -1)
        return torch.einsum("bchk,wk->bchw", time_real, cos_w) - torch.einsum(
            "bchk,wk->bchw", time_imag, sin_w
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        if self.export_manual_fft:
            real, imag = self._manual_rfft2(x)
            z_ri = torch.cat((real, imag), dim=1).to(dtype=x.dtype)
        else:
            z = torch.fft.rfft2(x.float(), norm="ortho")
            z_ri = torch.cat((z.real, z.imag), dim=1).to(dtype=x.dtype)
        z_ri = self.net(z_ri)
        real, imag = z_ri.float().chunk(2, dim=1)
        if self.export_manual_fft:
            out = self._manual_irfft2(real, imag, height, width)
        else:
            z = torch.complex(real, imag)
            out = torch.fft.irfft2(z, s=(height, width), norm="ortho")
        return out.to(dtype=x.dtype)


class FastFourierConv(nn.Module):
    """Fast Fourier Convolution block with local and global branches."""

    def __init__(
        self, channels: int, ratio_g: float = 0.5, kernel_size: int = 3
    ) -> None:
        super().__init__()
        if not (0.0 < ratio_g < 1.0):
            raise ValueError("ratio_g must be between 0 and 1")
        self.channels = channels
        self.c_g = max(1, round(channels * ratio_g))
        self.c_l = channels - self.c_g
        if self.c_l <= 0:
            self.c_l = 1
            self.c_g = channels - 1
        pad = kernel_size // 2

        self.l2l = nn.Conv2d(self.c_l, self.c_l, kernel_size, padding=pad)
        self.l2g = nn.Conv2d(self.c_l, self.c_g, kernel_size, padding=pad)
        self.g2l = nn.Conv2d(self.c_g, self.c_l, kernel_size, padding=pad)
        self.g2g = SpectralTransform(self.c_g)
        self.norm = GlobalLayerNorm(channels)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = torch.split(x, [self.c_l, self.c_g], dim=1)
        y_l = self.l2l(x_l) + self.g2l(x_g)
        y_g = self.l2g(x_l) + self.g2g(x_g)
        y = torch.cat((y_l, y_g), dim=1)
        return self.fuse(self.act(self.norm(y)))


class ResidualFFC(nn.Module):
    def __init__(self, channels: int, ratio_g: float, dropout: float) -> None:
        super().__init__()
        self.norm = GlobalLayerNorm(channels)
        self.ffc = FastFourierConv(channels, ratio_g=ratio_g)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ffc(self.norm(x)))


class IntraFrequencyRNN(nn.Module):
    """BLSTM over frequency bins for each time frame."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.rnn = nn.LSTM(
            input_size=channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden * 2, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, freq, frames = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(bsz * frames, freq, channels)
        y, _ = self.rnn(self.norm(seq))
        y = self.dropout(self.proj(y))
        y = y.reshape(bsz, frames, freq, channels).permute(0, 3, 2, 1)
        return x + y


class TimeSelfAttention(nn.Module):
    """Self-attention over time frames for each frequency bin."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, freq, frames = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bsz * freq, frames, channels)
        y, _ = self.attn(
            self.norm(seq), self.norm(seq), self.norm(seq), need_weights=False
        )
        y = self.dropout(y)
        y = y.reshape(bsz, freq, frames, channels).permute(0, 3, 1, 2)
        return x + y


class TimeSelfAttentionV2(nn.Module):
    """SDPA time attention with optional RoPE and QK normalization."""

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float,
        qk_norm: bool = True,
        rope: bool = True,
    ) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.dim_head = channels // heads
        self.qk_norm = qk_norm
        self.rope = rope and self.dim_head % 2 == 0
        self.dropout_p = dropout
        self.norm = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[-2]
        half = self.dim_head // 2
        pos = torch.arange(frames, device=x.device, dtype=torch.float32)
        freq = torch.arange(half, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (10_000 ** (freq / float(half)))
        angles = pos[:, None] * inv_freq[None, :]
        sin = angles.sin().to(dtype=x.dtype).view(1, 1, frames, half)
        cos = angles.cos().to(dtype=x.dtype).view(1, 1, frames, half)
        x1, x2 = x[..., :half], x[..., half : half * 2]
        rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        if self.dim_head == half * 2:
            return rotated
        return torch.cat((rotated, x[..., half * 2 :]), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, freq, frames = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bsz * freq, frames, channels)
        qkv = self.to_qkv(self.norm(seq))
        qkv = qkv.view(bsz * freq, frames, 3, self.heads, self.dim_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        if self.qk_norm:
            scale = self.dim_head**0.5
            q = F.normalize(q, dim=-1) * scale
            k = F.normalize(k, dim=-1) * scale

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y = y.permute(0, 2, 1, 3).reshape(bsz * freq, frames, channels)
        y = self.dropout(self.to_out(y))
        y = y.reshape(bsz, freq, frames, channels).permute(0, 3, 1, 2)
        return x + y


class TFFFCBlock(nn.Module):
    """TF-FFC block: FFC stack, retained intra-RNN, and attention."""

    def __init__(self, cfg: FSCNetConfig) -> None:
        super().__init__()
        self.ffc1 = ResidualFFC(cfg.channels, cfg.ffc_ratio, cfg.dropout)
        self.ffc2 = ResidualFFC(cfg.channels, cfg.ffc_ratio, cfg.dropout)
        self.intra_rnn = IntraFrequencyRNN(cfg.channels, cfg.rnn_hidden, cfg.dropout)
        self.attn: nn.Module
        if cfg.time_attention == "v1":
            self.attn = TimeSelfAttention(
                cfg.channels, cfg.attention_heads, cfg.dropout
            )
        elif cfg.time_attention == "v2":
            self.attn = TimeSelfAttentionV2(
                cfg.channels,
                cfg.attention_heads,
                cfg.dropout,
                qk_norm=cfg.time_attention_qk_norm,
                rope=cfg.time_attention_rope,
            )
        else:
            raise ValueError(f"Unknown time_attention={cfg.time_attention!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffc1(x)
        x = self.ffc2(x)
        x = self.intra_rnn(x)
        x = self.attn(x)
        return x


def cws_split(ri: torch.Tensor, subbands: int) -> Tuple[torch.Tensor, int]:
    """Channel-wise subband split: [B,2,F,T] -> [B,2*S,ceil(F/S),T]."""
    if ri.ndim != 4 or ri.shape[1] != 2:
        raise ValueError(f"Expected [B,2,F,T], got {tuple(ri.shape)}")
    bsz, ch, freq, frames = ri.shape
    pad = (-freq) % subbands
    if pad:
        ri = F.pad(ri, (0, 0, 0, pad))
    freq_pad = ri.shape[2]
    y = ri.view(bsz, ch, freq_pad // subbands, subbands, frames)
    y = y.permute(0, 1, 3, 2, 4).reshape(
        bsz, ch * subbands, freq_pad // subbands, frames
    )
    return y, pad


def cws_merge(x: torch.Tensor, subbands: int, original_freq: int) -> torch.Tensor:
    """Inverse of cws_split for tensors with channels=2*S."""
    bsz, channels, freq_groups, frames = x.shape
    if channels % subbands != 0:
        raise ValueError(f"channels={channels} is not divisible by subbands={subbands}")
    ch = channels // subbands
    y = x.view(bsz, ch, subbands, freq_groups, frames)
    y = y.permute(0, 1, 3, 2, 4).reshape(bsz, ch, freq_groups * subbands, frames)
    return y[:, :, :original_freq, :]


class FSCNet(nn.Module):
    """Full-Spectrum Context Network style generator."""

    def __init__(self, cfg: FSCNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = 2 * cfg.subbands
        self.input = nn.Sequential(
            nn.Conv2d(in_ch, cfg.channels, kernel_size=3, padding=1),
            GlobalLayerNorm(cfg.channels),
            nn.SiLU(),
        )
        self.ffc_in = ResidualFFC(cfg.channels, cfg.ffc_ratio, cfg.dropout)
        self.blocks = nn.ModuleList([TFFFCBlock(cfg) for _ in range(cfg.num_blocks)])
        self.stage_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualFFC(cfg.channels, cfg.ffc_ratio, cfg.dropout),
                    nn.Conv2d(cfg.channels, in_ch, kernel_size=3, padding=1),
                )
                for _ in range(cfg.num_blocks)
            ]
        )

    def encode_input(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = stft_complex(
            wav,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            center=self.cfg.center,
        )
        ri = complex_to_ri(spec)
        return spec, ri

    @overload
    def forward(
        self, wav_lr_up: torch.Tensor, return_all: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, wav_lr_up: torch.Tensor, return_all: Literal[True]
    ) -> tuple[list[torch.Tensor], torch.Tensor]: ...

    def forward(
        self, wav_lr_up: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor | tuple[list[torch.Tensor], torch.Tensor]:
        """Map upsampled narrow-band waveform [B,T] to complex spectra.

        Returns:
            if return_all=False: final [B,2,F,T_frames]
            if return_all=True: (list_of_stage_outputs, input_ri)
        """
        if wav_lr_up.ndim == 1:
            wav_lr_up = wav_lr_up.unsqueeze(0)
        _, input_ri = self.encode_input(wav_lr_up)
        original_freq = input_ri.shape[2]
        x, _ = cws_split(input_ri, self.cfg.subbands)
        h = self.ffc_in(self.input(x))

        stage_outputs: List[torch.Tensor] = []
        for block, head in zip(self.blocks, self.stage_heads):
            h = block(h)
            delta = cws_merge(head(h), self.cfg.subbands, original_freq)
            stage_outputs.append(input_ri + delta)

        if return_all:
            return stage_outputs, input_ri
        return stage_outputs[-1]

    def spec_to_wav(self, ri: torch.Tensor, length: int) -> torch.Tensor:
        return istft_complex(
            ri_to_complex(ri),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            length=length,
            center=self.cfg.center,
        )

    @torch.no_grad()
    def enhance(self, wav_lr_up: torch.Tensor) -> torch.Tensor:
        was_1d = wav_lr_up.ndim == 1
        if was_1d:
            wav_lr_up = wav_lr_up.unsqueeze(0)
        pred_ri = self.forward(wav_lr_up, return_all=False)
        wav = self.spec_to_wav(pred_ri, length=wav_lr_up.shape[-1])
        return wav.squeeze(0) if was_1d else wav


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
