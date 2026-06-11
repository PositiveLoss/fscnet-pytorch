"""Dataset utilities for speech bandwidth extension."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .audio import load_audio, match_length, peak_normalize_pair, resample_audio


def _resolve_path(base: Path, value: str) -> str:
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    """Read jsonl/csv/tsv/plain manifests.

    Accepted keys for clean target audio: hr_path, hr, path, wav, audio.
    Accepted keys for optional precomputed low-rate input: lr_path, lr, input.
    Plain-text manifests are treated as one clean target path per line.
    """
    path = Path(path)
    base = path.parent
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    items: List[Dict[str, str]] = []

    def normalize(row: Dict[str, str]) -> Dict[str, str]:
        hr = (
            row.get("hr_path")
            or row.get("hr")
            or row.get("path")
            or row.get("wav")
            or row.get("audio")
        )
        if not hr:
            raise ValueError(f"Manifest row has no HR path field: {row}")
        out = {"hr_path": _resolve_path(base, hr)}
        lr = row.get("lr_path") or row.get("lr") or row.get("input")
        if lr:
            out["lr_path"] = _resolve_path(base, lr)
        return out

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(normalize(json.loads(line)))
    elif suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                items.append(normalize({k: v for k, v in row.items() if v}))
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append({"hr_path": _resolve_path(base, line)})

    if not items:
        raise ValueError(f"No audio items found in {path}")
    return items


class BandwidthExtensionDataset(Dataset):
    """Random-crop paired dataset.

    If lr_path is absent, the low-band input is simulated by downsampling the
    clean target to input_sr and resampling it back to target_sr.
    """

    def __init__(
        self,
        manifest: str | Path,
        target_sr: int = 48_000,
        input_sr: int = 4_000,
        segment_seconds: float = 2.0,
        normalize: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.items = read_manifest(manifest)
        self.target_sr = int(target_sr)
        self.input_sr = int(input_sr)
        self.segment_samples = int(round(segment_seconds * target_sr))
        self.normalize = normalize
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.items)

    def _crop_or_pad(
        self, wav: torch.Tensor, start: Optional[int] = None
    ) -> tuple[torch.Tensor, int]:
        length = wav.shape[-1]
        seg = self.segment_samples
        if length >= seg:
            if start is None:
                start = random.randint(0, length - seg) if self.random_crop else 0
            return wav[start : start + seg], start
        wav = match_length(wav, seg)
        return wav, 0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        hr, _ = load_audio(item["hr_path"], target_sr=self.target_sr, mono=True)
        hr, start = self._crop_or_pad(hr)

        if "lr_path" in item:
            lr, _ = load_audio(item["lr_path"], target_sr=self.target_sr, mono=True)
            lr, _ = self._crop_or_pad(
                lr, start=start if lr.shape[-1] >= self.segment_samples else None
            )
        else:
            # Simulate the BWE input used in the paper: target -> narrowband -> target.
            lr_low = resample_audio(hr, self.target_sr, self.input_sr)
            lr = resample_audio(lr_low, self.input_sr, self.target_sr)
            lr = match_length(lr, self.segment_samples)

        if self.normalize:
            lr, hr = peak_normalize_pair(lr, hr)
        return {"lr": lr.to(torch.float32), "hr": hr.to(torch.float32)}
