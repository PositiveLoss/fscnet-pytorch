"""Split an FSC-Net manifest into train and validation manifests."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from fscnet_pytorch.cli import option, run


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must contain a list: {path}")
        rows = data
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, dialect=dialect))
    else:
        rows = [
            {"path": line.strip()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not rows:
        raise ValueError(f"No rows found in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Manifest rows must be objects: {path}")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    raise ValueError(f"Output manifest must be .jsonl or .json: {path}")


def split_count(total: int, valid_ratio: float) -> int:
    count = round(total * valid_ratio)
    if total > 1 and valid_ratio > 0.0:
        count = max(1, count)
    if total > 1:
        count = min(total - 1, count)
    return count


def default_output_paths(manifest: Path) -> tuple[Path, Path]:
    return manifest.with_name("train.jsonl"), manifest.with_name("valid.jsonl")


def main(
    manifest: Path = option(
        ...,
        "--manifest",
        help="input manifest to split",
    ),
    train_out: Path | None = option(
        None,
        "--train-out",
        "--train_out",
        help="training output path; defaults to train.jsonl next to input",
    ),
    valid_out: Path | None = option(
        None,
        "--valid-out",
        "--valid_out",
        help="validation output path; defaults to valid.jsonl next to input",
    ),
    valid_ratio: float = option(
        0.1,
        "--valid-ratio",
        "--valid_ratio",
        help="fraction of rows to place in validation",
        min=0.0,
        max=1.0,
    ),
    seed: int = option(1234, "--seed", help="deterministic shuffle seed"),
    no_shuffle: bool = option(
        False,
        "--no-shuffle",
        "--no_shuffle",
        help="preserve input order before splitting",
    ),
) -> None:
    """Split a manifest into train and validation files."""

    manifest = manifest.expanduser().resolve()
    default_train, default_valid = default_output_paths(manifest)
    train_path = (train_out or default_train).expanduser().resolve()
    valid_path = (valid_out or default_valid).expanduser().resolve()
    if train_path == valid_path:
        raise ValueError("--train-out and --valid-out must be different paths")

    rows = read_rows(manifest)
    if not no_shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    n_valid = split_count(len(rows), valid_ratio)
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]
    write_rows(train_path, train_rows)
    write_rows(valid_path, valid_rows)
    print(
        f"Wrote {len(train_rows)} train rows to {train_path} and "
        f"{len(valid_rows)} validation rows to {valid_path}"
    )


if __name__ == "__main__":
    run(main)
