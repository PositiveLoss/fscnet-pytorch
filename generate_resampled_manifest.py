#!/usr/bin/env python3
"""Prepare paired FSC-Net manifests with fast-audio-resampler.

The generated JSONL manifest contains:

  {"hr_path": "...", "lr_path": "..."}

`hr_path` is clean audio at --target_sr. `lr_path` is the same clean audio
resampled down to --input_sr, then back to --target_sr, so the trainer can use
precomputed narrowband inputs without falling back to its built-in resampler.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import numpy as np


DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc")


@dataclass(frozen=True)
class WorkItem:
    index: int
    total: int
    source: Path
    input_dir: Path
    out_dir: Path
    manifest_path: Path
    input_sr: int
    target_sr: int
    channels: int
    quality: str
    backend: str
    chunk_frames: int
    absolute_paths: bool
    overwrite: bool


@dataclass(frozen=True)
class WorkResult:
    index: int
    row: dict[str, str]
    message: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an FSC-Net training manifest with fast-audio-resampler"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="directory containing clean source audio files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="directory for resampled HR/LR files and the manifest",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="output JSONL path; defaults to <out_dir>/manifest.jsonl",
    )
    parser.add_argument("--target_sr", type=int, default=48_000)
    parser.add_argument("--input_sr", type=int, default=4_000)
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=(1, 2),
        help="output channel count; 1 mixes source audio to mono",
    )
    parser.add_argument(
        "--quality",
        default="balanced",
        choices=("fast", "balanced", "best"),
        help="fast-audio-resampler quality preset",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="fast-audio-resampler backend, for example auto or scalar",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=0,
        help="streaming chunk size in frames; 0 processes each file in one call",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="comma-separated input extensions",
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="write absolute paths into the manifest instead of paths relative to it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing generated wav files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel worker processes; 0 uses os.cpu_count(), 1 disables concurrency",
    )
    parser.add_argument("--limit", type=int, default=None, help="process at most N files")
    return parser


def audio_paths(input_dir: Path, extensions: Sequence[str]) -> list[Path]:
    ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ext_set
    )


def load_audio(path: Path, channels: int) -> tuple[np.ndarray, int]:
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if channels == 1:
        data = data.mean(axis=1, keepdims=True)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > channels:
        data = data[:, :channels]
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate)


def resample_audio(
    audio: np.ndarray,
    input_rate: int,
    output_rate: int,
    channels: int,
    quality: str,
    backend: str,
    chunk_frames: int,
) -> np.ndarray:
    import numpy as np

    if input_rate == output_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    try:
        from fast_audio_resampler import F32Resampler
    except ImportError as exc:
        raise RuntimeError(
            "fast-audio-resampler is required. Install the wheel declared in "
            "pyproject.toml before running this script."
        ) from exc

    resampler = F32Resampler(
        int(input_rate),
        int(output_rate),
        int(channels),
        quality=quality,
        backend=backend,
    )
    interleaved = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    output: list[float] = []
    frame_step = max(0, chunk_frames) * channels
    chunks: Iterable[np.ndarray]
    if frame_step > 0:
        chunks = (
            interleaved[start : start + frame_step]
            for start in range(0, interleaved.shape[0], frame_step)
        )
    else:
        chunks = (interleaved,)

    for chunk in chunks:
        resampled, _stats = _process_resampler_chunk(resampler, chunk)
        output.extend(resampled)
    tail, _tail_stats = resampler.finish()
    output.extend(tail)

    out = np.asarray(output, dtype=np.float32)
    frame_count = out.shape[0] // channels
    if frame_count * channels != out.shape[0]:
        raise RuntimeError(
            f"Resampler returned {out.shape[0]} samples for {channels} channels"
        )
    return out.reshape(frame_count, channels)


def _process_resampler_chunk(resampler: object, chunk: np.ndarray) -> tuple[list[float], object]:
    process = getattr(resampler, "process")
    try:
        output, stats = process(chunk)
    except TypeError:
        output, stats = process(chunk.tolist())
    return list(output), stats


def relative_or_absolute(path: Path, manifest_path: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    return os.path.relpath(path.resolve(), manifest_path.parent.resolve())


def output_paths(
    source: Path, input_dir: Path, out_dir: Path, input_sr: int
) -> tuple[Path, Path]:
    rel = source.relative_to(input_dir)
    stem_path = rel.with_suffix("")
    hr_path = out_dir / "hr" / stem_path.with_suffix(".wav")
    lr_path = out_dir / f"lr_{input_sr}" / stem_path.with_suffix(".wav")
    return hr_path, lr_path


def process_file(item: WorkItem) -> WorkResult:
    hr_path, lr_path = output_paths(
        item.source, item.input_dir, item.out_dir, item.input_sr
    )
    row = {
        "hr_path": relative_or_absolute(
            hr_path, item.manifest_path, item.absolute_paths
        ),
        "lr_path": relative_or_absolute(
            lr_path, item.manifest_path, item.absolute_paths
        ),
    }
    if not item.overwrite and hr_path.exists() and lr_path.exists():
        return WorkResult(
            item.index, row, f"[{item.index}/{item.total}] reused {lr_path}"
        )

    audio, source_sr = load_audio(item.source, item.channels)
    hr = resample_audio(
        audio,
        source_sr,
        item.target_sr,
        item.channels,
        item.quality,
        item.backend,
        item.chunk_frames,
    )
    lr_low = resample_audio(
        hr,
        item.target_sr,
        item.input_sr,
        item.channels,
        item.quality,
        item.backend,
        item.chunk_frames,
    )
    lr = resample_audio(
        lr_low,
        item.input_sr,
        item.target_sr,
        item.channels,
        item.quality,
        item.backend,
        item.chunk_frames,
    )

    write_audio(hr_path, hr, item.target_sr)
    write_audio(lr_path, lr, item.target_sr)
    return WorkResult(
        item.index, row, f"[{item.index}/{item.total}] {item.source} -> {lr_path}"
    )


def resolve_workers(workers: int) -> int:
    if workers < 0:
        raise ValueError("--workers must be >= 0")
    if workers == 0:
        return os.cpu_count() or 1
    return workers


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else out_dir / "manifest.jsonl"
    )
    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext)
    sources = audio_paths(input_dir, extensions)
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError(f"No audio files found in {input_dir}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    work_items = [
        WorkItem(
            index=index,
            total=len(sources),
            source=source,
            input_dir=input_dir,
            out_dir=out_dir,
            manifest_path=manifest_path,
            input_sr=args.input_sr,
            target_sr=args.target_sr,
            channels=args.channels,
            quality=args.quality,
            backend=args.backend,
            chunk_frames=args.chunk_frames,
            absolute_paths=args.absolute_paths,
            overwrite=args.overwrite,
        )
        for index, source in enumerate(sources, start=1)
    ]

    workers = resolve_workers(args.workers)
    print(f"Processing {len(work_items)} files with {workers} worker(s)")
    if workers == 1:
        results = [process_file(item) for item in work_items]
        for result in results:
            print(result.message)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_file, item) for item in work_items]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(result.message)

    rows = [result.row for result in sorted(results, key=lambda result: result.index)]

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
