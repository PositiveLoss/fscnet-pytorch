"""Prepare paired FSC-Net manifests with fast-audio-resampler.

The generated JSONL manifest contains:

  {"hr_path": "...", "lr_path": "..."}

`hr_path` is clean audio at --target_sr. `lr_path` is the same clean audio
resampled down to --input_sr, then back to --target_sr, so the trainer can use
precomputed narrowband inputs without falling back to its built-in resampler.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Sequence, TextIO

if TYPE_CHECKING:
    import numpy as np

from fscnet_pytorch.cli import option, run


DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".opus")
Quality = Literal["fast", "balanced", "best"]
LOGGER = logging.getLogger("generate_resampled_manifest")


@dataclass(frozen=True)
class WorkItem:
    index: int
    total: int | None
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
    log_level: str
    min_duration_seconds: float
    max_duration_seconds: float | None


@dataclass(frozen=True)
class WorkResult:
    index: int
    row: dict[str, str] | None
    message: str
    skipped: bool = False


def audio_paths(input_dir: Path, extensions: Sequence[str]) -> Iterator[Path]:
    ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            yield path


def audio_duration_seconds(path: Path) -> float:
    data, sample_rate = load_audio_with_torchaudio(path)
    return float(data.shape[0]) / float(sample_rate)


def load_audio(path: Path, channels: int) -> tuple[np.ndarray, int]:
    data, sample_rate = load_audio_with_torchaudio(path)
    normalized = normalize_channels(data, channels)
    LOGGER.info(
        "Loaded %s: sr=%d frames=%d channels=%d",
        path,
        sample_rate,
        normalized.shape[0],
        normalized.shape[1],
    )
    return normalized, sample_rate


def load_audio_with_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import numpy as np
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    LOGGER.debug("Decoded %s with torchaudio", path)
    data = waveform.transpose(0, 1).contiguous().numpy()
    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def normalize_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    import numpy as np

    data = np.asarray(audio, dtype=np.float32)
    if channels == 1:
        data = data.mean(axis=1, keepdims=True)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > channels:
        data = data[:, :channels]
    return np.ascontiguousarray(data, dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import numpy as np
    import torch
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    waveform = torch.from_numpy(data.T.copy())
    torchaudio.save(str(path), waveform, sample_rate)


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
        input_rate,
        output_rate,
        channels,
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
    configure_logging(item.log_level)
    started = time.perf_counter()
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
        skip_result = duration_skip_result(item, audio_duration_seconds(item.source))
        if skip_result is not None:
            return skip_result
        LOGGER.info(
            "[%s] Reusing existing outputs for %s",
            progress_label(item.index, item.total),
            item.source,
        )
        LOGGER.debug("Existing HR: %s", hr_path)
        LOGGER.debug("Existing LR: %s", lr_path)
        return WorkResult(
            item.index,
            row,
            f"[{progress_label(item.index, item.total)}] reused {lr_path}",
        )

    LOGGER.info(
        "[%s] Processing %s", progress_label(item.index, item.total), item.source
    )
    audio, source_sr = load_audio(item.source, item.channels)
    source_duration = audio.shape[0] / float(source_sr)
    skip_result = duration_skip_result(item, source_duration)
    if skip_result is not None:
        return skip_result
    LOGGER.info(
        "[%s] Resampling HR %s: %d Hz -> %d Hz",
        progress_label(item.index, item.total),
        item.source,
        source_sr,
        item.target_sr,
    )
    hr = resample_audio(
        audio,
        source_sr,
        item.target_sr,
        item.channels,
        item.quality,
        item.backend,
        item.chunk_frames,
    )
    LOGGER.info(
        "[%s] Creating narrowband input: %d Hz -> %d Hz -> %d Hz",
        progress_label(item.index, item.total),
        item.target_sr,
        item.input_sr,
        item.target_sr,
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

    LOGGER.info("[%s] Writing HR %s", progress_label(item.index, item.total), hr_path)
    write_audio(hr_path, hr, item.target_sr)
    LOGGER.info("[%s] Writing LR %s", progress_label(item.index, item.total), lr_path)
    write_audio(lr_path, lr, item.target_sr)
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "[%s] Finished %s in %.2fs",
        progress_label(item.index, item.total),
        item.source,
        elapsed,
    )
    return WorkResult(
        item.index,
        row,
        f"[{progress_label(item.index, item.total)}] "
        f"{item.source} -> {lr_path} ({elapsed:.2f}s)",
    )


def duration_skip_result(item: WorkItem, duration: float) -> WorkResult | None:
    if duration < item.min_duration_seconds:
        return WorkResult(
            item.index,
            None,
            (
                f"[{progress_label(item.index, item.total)}] skipped {item.source}: "
                f"duration {duration:.3f}s < min "
                f"{item.min_duration_seconds:.3f}s"
            ),
            skipped=True,
        )
    if item.max_duration_seconds is not None and duration > item.max_duration_seconds:
        return WorkResult(
            item.index,
            None,
            (
                f"[{progress_label(item.index, item.total)}] skipped {item.source}: "
                f"duration {duration:.3f}s > max "
                f"{item.max_duration_seconds:.3f}s"
            ),
            skipped=True,
        )
    return None


def progress_label(index: int, total: int | None) -> str:
    if total is None:
        return str(index)
    return f"{index}/{total}"


def make_work_item(
    index: int,
    source: Path,
    total: int | None,
    input_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    input_sr: int,
    target_sr: int,
    channels: int,
    quality: str,
    backend: str,
    chunk_frames: int,
    absolute_paths: bool,
    overwrite: bool,
    log_level: str,
    min_duration_seconds: float,
    max_duration_seconds: float | None,
) -> WorkItem:
    return WorkItem(
        index=index,
        total=total,
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
        log_level=log_level,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )


def resolve_workers(workers: int) -> int:
    if workers < 0:
        raise ValueError("--workers must be >= 0")
    if workers == 0:
        return os.cpu_count() or 1
    return workers


def configure_logging(log_level: str) -> None:
    level_name = log_level.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown --log-level {log_level!r}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def write_manifest_row(manifest_file: TextIO, row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def drain_completed_futures(
    pending: set[Future[WorkResult]],
    completed: dict[int, WorkResult],
) -> None:
    done, pending_items = wait(pending, return_when=FIRST_COMPLETED)
    pending.clear()
    pending.update(pending_items)
    for future in done:
        result = future.result()
        completed[result.index] = result


def flush_ordered_results(
    completed: dict[int, WorkResult],
    next_index: int,
    manifest_file: TextIO,
) -> tuple[int, int, int]:
    written = 0
    skipped = 0
    while next_index in completed:
        result = completed.pop(next_index)
        if result.skipped:
            skipped += 1
        if write_manifest_row(manifest_file, result.row):
            written += 1
        LOGGER.info(result.message)
        next_index += 1
    return next_index, written, skipped


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
    log_level: str = option(
        "INFO",
        "--log-level",
        "--log_level",
        help="logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL",
    ),
    min_duration_seconds: float = option(
        0.1,
        "--min-duration-seconds",
        "--min_duration_seconds",
        help="skip files shorter than this many seconds",
        min=0.0,
    ),
    max_duration_seconds: float | None = option(
        30.0,
        "--max-duration-seconds",
        "--max_duration_seconds",
        help="skip files longer than this many seconds; set 0 for no maximum",
        min=0.0,
    ),
    limit: int | None = option(None, "--limit", help="process at most N files", min=1),
    prefetch: int = option(
        4,
        "--prefetch",
        help="parallel backlog per worker; lower values use less memory",
        min=1,
    ),
    sort_paths: bool = option(
        False,
        "--sort-paths",
        "--sort_paths",
        help="sort discovered paths before processing; requires loading all paths",
    ),
) -> None:
    """Generate an FSC-Net training manifest with fast-audio-resampler."""
    configure_logging(log_level)
    if channels not in {1, 2}:
        raise ValueError("--channels must be 1 or 2")
    if max_duration_seconds == 0.0:
        max_duration_seconds = None
    if max_duration_seconds is not None and max_duration_seconds < min_duration_seconds:
        raise ValueError("--max-duration-seconds must be >= --min-duration-seconds")

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
    if not extension_values:
        raise ValueError("--extensions must include at least one extension")
    LOGGER.info(
        "Scanning %s for extensions: %s", input_dir, ", ".join(extension_values)
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing generated audio under %s", out_dir)
    LOGGER.info("Writing manifest to %s", manifest_path)

    worker_count = resolve_workers(workers)
    LOGGER.info("Processing files with %d worker(s)", worker_count)
    if limit is not None:
        LOGGER.info("Limiting run to first %d discovered file(s)", limit)
    if sort_paths:
        LOGGER.info("Sorting discovered paths before processing")

    discovered = 0
    written = 0
    skipped = 0
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        dir=manifest_path.parent,
        text=True,
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as manifest_file:
            sources = audio_paths(input_dir, extension_values)
            if sort_paths:
                sources = iter(sorted(sources))
            if worker_count == 1:
                for index, source in enumerate(sources, start=1):
                    if limit is not None and index > limit:
                        break
                    discovered = index
                    result = process_file(
                        make_work_item(
                            index,
                            source,
                            None,
                            input_dir,
                            out_dir,
                            manifest_path,
                            input_sr,
                            target_sr,
                            channels,
                            quality,
                            backend,
                            chunk_frames,
                            absolute_paths,
                            overwrite,
                            log_level,
                            min_duration_seconds,
                            max_duration_seconds,
                        )
                    )
                    if result.skipped:
                        skipped += 1
                    if write_manifest_row(manifest_file, result.row):
                        written += 1
                    LOGGER.info(result.message)
            else:
                max_pending = max(1, worker_count * prefetch)
                pending: set[Future[WorkResult]] = set()
                completed: dict[int, WorkResult] = {}
                next_write_index = 1
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    for index, source in enumerate(sources, start=1):
                        if limit is not None and index > limit:
                            break
                        discovered = index
                        pending.add(
                            executor.submit(
                                process_file,
                                make_work_item(
                                    index,
                                    source,
                                    None,
                                    input_dir,
                                    out_dir,
                                    manifest_path,
                                    input_sr,
                                    target_sr,
                                    channels,
                                    quality,
                                    backend,
                                    chunk_frames,
                                    absolute_paths,
                                    overwrite,
                                    log_level,
                                    min_duration_seconds,
                                    max_duration_seconds,
                                ),
                            )
                        )
                        if len(pending) >= max_pending:
                            drain_completed_futures(pending, completed)
                            (
                                next_write_index,
                                rows_written,
                                rows_skipped,
                            ) = flush_ordered_results(
                                completed, next_write_index, manifest_file
                            )
                            written += rows_written
                            skipped += rows_skipped

                    while pending:
                        drain_completed_futures(pending, completed)
                        (
                            next_write_index,
                            rows_written,
                            rows_skipped,
                        ) = flush_ordered_results(
                            completed, next_write_index, manifest_file
                        )
                        written += rows_written
                        skipped += rows_skipped

                    if completed:
                        pending_indexes = ", ".join(str(index) for index in completed)
                        raise RuntimeError(
                            "Internal error: completed results could not be written "
                            f"in order: {pending_indexes}"
                        )

        if discovered == 0:
            raise ValueError(f"No audio files found in {input_dir}")
        if written == 0:
            raise ValueError("No audio files remained after duration filtering")
        temp_path.replace(manifest_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    LOGGER.info(
        "Wrote %d rows to %s (%d discovered, %d skipped)",
        written,
        manifest_path,
        discovered,
        skipped,
    )


if __name__ == "__main__":
    run(main)
