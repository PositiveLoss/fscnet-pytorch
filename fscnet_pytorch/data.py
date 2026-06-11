"""Dataset utilities for speech bandwidth extension."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .audio import load_audio, match_length, peak_normalize_pair, resample_audio
from .validation import normalize_manifest_row


def iter_manifest(path: str | Path) -> Iterator[Dict[str, str]]:
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

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must contain a list of rows: {path}")
        for row in data:
            if not isinstance(row, dict):
                raise ValueError(f"JSON manifest rows must be objects: {path}")
            yield normalize_manifest_row(row, base)
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield normalize_manifest_row(json.loads(line), base)
    elif suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                yield normalize_manifest_row({k: v for k, v in row.items() if v}, base)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield normalize_manifest_row({"path": line}, base)


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    items = list(iter_manifest(path))
    if not items:
        raise ValueError(f"No audio items found in {path}")
    return items


def count_manifest_rows(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must contain a list of rows: {path}")
        return len(data)
    if suffix in {".jsonl", ".txt"}:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _row in csv.DictReader(f, dialect=dialect))
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _shuffle_buffered(
    items: Iterator[Dict[str, str]], buffer_size: int
) -> Iterator[Dict[str, str]]:
    if buffer_size <= 1:
        yield from items
        return

    rng = random.Random(torch.initial_seed())
    buffer: list[Dict[str, str]] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item

    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _item_to_sample(
    item: Dict[str, str],
    target_sr: int,
    input_sr: int,
    segment_samples: int,
    normalize: bool,
    random_crop: bool,
) -> Dict[str, torch.Tensor]:
    hr, _ = load_audio(item["hr_path"], target_sr=target_sr, mono=True)
    hr, start = _crop_or_pad(hr, segment_samples, random_crop=random_crop)

    if "lr_path" in item:
        lr, _ = load_audio(item["lr_path"], target_sr=target_sr, mono=True)
        lr, _ = _crop_or_pad(
            lr,
            segment_samples,
            start=start if lr.shape[-1] >= segment_samples else None,
            random_crop=random_crop,
        )
    else:
        # Simulate the BWE input used in the paper: target -> narrowband -> target.
        lr_low = resample_audio(hr, target_sr, input_sr)
        lr = resample_audio(lr_low, input_sr, target_sr)
        lr = match_length(lr, segment_samples)

    if normalize:
        lr, hr = peak_normalize_pair(lr, hr)
    return {"lr": lr.to(torch.float32), "hr": hr.to(torch.float32)}


def _crop_or_pad(
    wav: torch.Tensor,
    segment_samples: int,
    start: Optional[int] = None,
    random_crop: bool = True,
) -> tuple[torch.Tensor, int]:
    length = wav.shape[-1]
    if length >= segment_samples:
        if start is None:
            start = random.randint(0, length - segment_samples) if random_crop else 0
        return wav[start : start + segment_samples], start
    wav = match_length(wav, segment_samples)
    return wav, 0


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
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_samples = round(segment_seconds * target_sr)
        self.normalize = normalize
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.items)

    def _crop_or_pad(
        self, wav: torch.Tensor, start: Optional[int] = None
    ) -> tuple[torch.Tensor, int]:
        return _crop_or_pad(
            wav,
            self.segment_samples,
            start=start,
            random_crop=self.random_crop,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        return _item_to_sample(
            item,
            self.target_sr,
            self.input_sr,
            self.segment_samples,
            self.normalize,
            self.random_crop,
        )


class StreamingBandwidthExtensionDataset(IterableDataset):
    """Streaming manifest dataset with bounded shuffle buffering."""

    def __init__(
        self,
        manifest: str | Path,
        target_sr: int = 48_000,
        input_sr: int = 4_000,
        segment_seconds: float = 2.0,
        normalize: bool = True,
        random_crop: bool = True,
        shuffle_buffer_size: int = 8192,
        row_count: int | None = None,
        shard_rank: int = 0,
        shard_world_size: int = 1,
    ) -> None:
        self.manifest = Path(manifest)
        self.target_sr = target_sr
        self.input_sr = input_sr
        self.segment_samples = round(segment_seconds * target_sr)
        self.normalize = normalize
        self.random_crop = random_crop
        self.shuffle_buffer_size = shuffle_buffer_size
        self.row_count = row_count
        self.shard_rank = shard_rank
        self.shard_world_size = shard_world_size

    def __len__(self) -> int:
        if self.row_count is None:
            raise TypeError("Streaming dataset length is unknown")
        return max(
            0,
            (self.row_count + self.shard_world_size - 1 - self.shard_rank)
            // self.shard_world_size,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        total_shards = self.shard_world_size * num_workers
        shard_id = self.shard_rank * num_workers + worker_id

        def rows() -> Iterator[Dict[str, str]]:
            for index, item in enumerate(iter_manifest(self.manifest)):
                if index % total_shards == shard_id:
                    yield item

        for item in _shuffle_buffered(rows(), self.shuffle_buffer_size):
            yield _item_to_sample(
                item,
                self.target_sr,
                self.input_sr,
                self.segment_samples,
                self.normalize,
                self.random_crop,
            )
