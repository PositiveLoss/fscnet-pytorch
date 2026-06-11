"""Optional Pydantic-backed validation helpers.

The project can run without Pydantic installed. When it is available, these
helpers use it for convenient type coercion and structured errors, then keep a
small set of local checks for invariants that are specific to this model.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
from typing import Any, Iterable, Mapping


CONFIG_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "target_sr": int,
    "input_sr": int,
    "n_fft": int,
    "win_length": int,
    "hop_length": int,
    "subbands": int,
    "channels": int,
    "num_blocks": int,
    "ffc_ratio": float,
    "attention_heads": int,
    "time_attention": str,
    "time_attention_qk_norm": bool,
    "time_attention_rope": bool,
    "rnn_hidden": int,
    "dropout": float,
    "center": bool,
}

POSITIVE_INT_FIELDS = (
    "target_sr",
    "input_sr",
    "n_fft",
    "win_length",
    "hop_length",
    "subbands",
    "channels",
    "num_blocks",
    "attention_heads",
    "rnn_hidden",
)


def _pydantic() -> Any | None:
    try:
        return importlib.import_module("pydantic")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _manifest_row_model() -> Any | None:
    pydantic = _pydantic()
    if pydantic is None:
        return None
    field = getattr(pydantic, "Field")
    create_model = getattr(pydantic, "create_model")
    return create_model(
        "ManifestRowSchema",
        hr_path=(str | None, field(default=None)),
        hr=(str | None, field(default=None)),
        path=(str | None, field(default=None)),
        wav=(str | None, field(default=None)),
        audio=(str | None, field(default=None)),
        lr_path=(str | None, field(default=None)),
        lr=(str | None, field(default=None)),
        input=(str | None, field(default=None)),
        __config__=getattr(pydantic, "ConfigDict")(extra="ignore"),
    )


def validate_fscnet_config_data(
    data: Mapping[str, Any], valid_fields: Iterable[str]
) -> dict[str, Any]:
    valid = set(valid_fields)
    filtered = {key: value for key, value in data.items() if key in valid}
    validated = _validate_with_pydantic(filtered)
    _validate_config_invariants(validated)
    return validated


def _validate_with_pydantic(data: Mapping[str, Any]) -> dict[str, Any]:
    pydantic = _pydantic()
    if pydantic is None:
        return dict(data)

    field = getattr(pydantic, "Field")
    create_model = getattr(pydantic, "create_model")
    fields: dict[str, tuple[Any, Any]] = {}
    for key in data:
        expected = CONFIG_FIELD_TYPES[key]
        if key in POSITIVE_INT_FIELDS:
            fields[key] = (expected, field(gt=0))
        elif key in {"ffc_ratio", "dropout"}:
            fields[key] = (expected, field(ge=0.0, le=1.0))
        else:
            fields[key] = (expected, field())

    if not fields:
        return {}

    model = create_model(
        "FSCNetConfigSchema",
        **fields,
        __config__=getattr(pydantic, "ConfigDict")(extra="forbid"),
    )
    parsed = model.model_validate(data)
    return parsed.model_dump()


def _validate_config_invariants(data: Mapping[str, Any]) -> None:
    for field in POSITIVE_INT_FIELDS:
        if field in data:
            value = data[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")

    for field in ("ffc_ratio", "dropout"):
        if field in data:
            value = data[field]
            if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value!r}")

    if "time_attention" in data and data["time_attention"] not in {"v1", "v2"}:
        raise ValueError("time_attention must be one of: v1, v2")

    if (
        "win_length" in data
        and "n_fft" in data
        and int(data["win_length"]) > int(data["n_fft"])
    ):
        raise ValueError("win_length must be <= n_fft")

    if (
        "hop_length" in data
        and "win_length" in data
        and int(data["hop_length"]) > int(data["win_length"])
    ):
        raise ValueError("hop_length must be <= win_length")


def validate_progressive_windows(windows: Iterable[int]) -> tuple[int, ...]:
    validated = tuple(windows)
    if not validated:
        raise ValueError("At least one progressive window is required")
    for window in validated:
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError(
                f"Progressive windows must be positive integers: {validated}"
            )
    return validated


def normalize_manifest_row(row: Mapping[str, Any], base: Path) -> dict[str, str]:
    parsed = _validate_manifest_row(row)
    hr = (
        parsed.get("hr_path")
        or parsed.get("hr")
        or parsed.get("path")
        or parsed.get("wav")
        or parsed.get("audio")
    )
    if not hr:
        raise ValueError(f"Manifest row has no HR path field: {row}")

    out = {"hr_path": _resolve_path(base, hr)}
    lr = parsed.get("lr_path") or parsed.get("lr") or parsed.get("input")
    if lr:
        out["lr_path"] = _resolve_path(base, lr)
    return out


def _validate_manifest_row(row: Mapping[str, Any]) -> dict[str, str | None]:
    model = _manifest_row_model()
    if model is None:
        return {str(key): str(value) for key, value in row.items() if value}
    parsed = model.model_validate(row)
    return parsed.model_dump(exclude_none=True)


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())
