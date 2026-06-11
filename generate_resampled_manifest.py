#!/usr/bin/env python3
"""Prepare paired FSC-Net manifests with fast-audio-resampler.

The generated JSONL manifest contains:

  {"hr_path": "...", "lr_path": "..."}

`hr_path` is clean audio at --target_sr. `lr_path` is the same clean audio
resampled down to --input_sr, then back to --target_sr, so the trainer can use
precomputed narrowband inputs without falling back to its built-in resampler.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np

from fscnet_pytorch.cli import option, run


DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc")
Quality = Literal["fast", "balanced", "best"]


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
        import fast_audio_resampler
    except ImportError as exc:
        raise RuntimeError(
            "fast-audio-resampler is required. Install the wheel declared in "
            "pyproject.toml before running this script."
        ) from exc

    f32_resampler = getattr(fast_audio_resampler, "F32Resampler")
    resampler = f32_resampler(
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


def _process_resampler_chunk(
    resampler: object, chunk: np.ndarray
) -> tuple[list[float], object]:
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


def main(
    input_dir: Path = option(
        ...,
        "--input-dir",
        "--input_dir",
        help="directory containing clean source audio files",
    ),
    out_dir: Path = option(
        ...,
        "--out-dir",
        "--out_dir",
        help="directory for resampled HR/LR files and the manifest",
    ),
    manifest: Path | None = option(
        None,
        "--manifest",
        help="output JSONL path; defaults to <out_dir>/manifest.jsonl",
    ),
    target_sr: int = option(
        48_000, "--target-sr", "--target_sr", help="HR sample rate", min=1
    ),
    input_sr: int = option(
        4_000, "--input-sr", "--input_sr", help="narrowband rate", min=1
    ),
    channels: int = option(
        1,
        "--channels",
        help="output channel count; 1 mixes source audio to mono",
        min=1,
        max=2,
    ),
    quality: Quality = option(
        "balanced",
        "--quality",
        help="fast-audio-resampler quality preset",
    ),
    backend: str = option(
        "auto",
        "--backend",
        help="fast-audio-resampler backend, for example auto or scalar",
    ),
    chunk_frames: int = option(
        0,
        "--chunk-frames",
        "--chunk_frames",
        help="streaming chunk size in frames; 0 processes each file in one call",
        min=0,
    ),
    extensions: str = option(
        ",".join(DEFAULT_EXTENSIONS),
        "--extensions",
        help="comma-separated input extensions",
    ),
    absolute_paths: bool = option(
        False,
        "--absolute-paths",
        "--absolute_paths",
        help="write absolute paths into the manifest instead of paths relative to it",
    ),
    overwrite: bool = option(
        False,
        "--overwrite",
        help="overwrite existing generated wav files",
    ),
    workers: int = option(
        0,
        "--workers",
        help="parallel worker processes; 0 uses os.cpu_count(), 1 disables concurrency",
        min=0,
    ),
    limit: int | None = option(None, "--limit", help="process at most N files", min=1),
) -> None:
    """Generate an FSC-Net training manifest with fast-audio-resampler."""
    if channels not in {1, 2}:
        raise ValueError("--channels must be 1 or 2")

    input_dir = input_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    manifest_path = (
        manifest.expanduser().resolve()
        if manifest is not None
        else out_dir / "manifest.jsonl"
    )
    extension_values = tuple(
        ext.strip().lower() for ext in extensions.split(",") if ext
    )
    sources = audio_paths(input_dir, extension_values)
    if limit is not None:
        sources = sources[:limit]
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
            input_sr=input_sr,
            target_sr=target_sr,
            channels=channels,
            quality=quality,
            backend=backend,
            chunk_frames=chunk_frames,
            absolute_paths=absolute_paths,
            overwrite=overwrite,
        )
        for index, source in enumerate(sources, start=1)
    ]

    worker_count = resolve_workers(workers)
    print(f"Processing {len(work_items)} files with {worker_count} worker(s)")
    if worker_count == 1:
        results = [process_file(item) for item in work_items]
        for result in results:
            print(result.message)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
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
    run(main)
