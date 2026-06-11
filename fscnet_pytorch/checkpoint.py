"""Checkpoint helpers backed by safetensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .model import FSCNet


def checkpoint_sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".safetensors":
        return path.with_suffix(".json")
    return path.with_suffix(path.suffix + ".json")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: FSCNet,
    config: dict[str, Any],
    windows: tuple[int, ...],
    epoch: int,
    step: int,
    args: dict[str, Any],
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: Any,
    discriminators: torch.nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scheduler_d: Any = None,
) -> None:
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    _add_state_tensors(tensors, "model", model.state_dict())
    if discriminators is not None:
        _add_state_tensors(tensors, "discriminators", discriminators.state_dict())

    sidecar = {
        "format": "fscnet-safetensors-v1",
        "config": config,
        "windows": list(windows),
        "epoch": epoch,
        "step": step,
        "args": args,
        "optimizer_g": _extract_tensors(
            optimizer_g.state_dict(), "optimizer_g", tensors
        ),
        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
        "optimizer_d": (
            _extract_tensors(optimizer_d.state_dict(), "optimizer_d", tensors)
            if optimizer_d is not None
            else None
        ),
        "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    checkpoint_sidecar_path(path).write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got {path}")

    tensors = load_file(path, device="cpu")
    sidecar_path = checkpoint_sidecar_path(path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing checkpoint sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("format") != "fscnet-safetensors-v1":
        raise ValueError(f"Unsupported checkpoint format in {sidecar_path}")

    ckpt: dict[str, Any] = {
        "model": _prefixed_state_dict(tensors, "model"),
        "config": sidecar["config"],
        "windows": tuple(sidecar["windows"]),
        "epoch": sidecar["epoch"],
        "step": sidecar["step"],
        "args": sidecar.get("args", {}),
        "optimizer_g": _restore_tensors(sidecar["optimizer_g"], tensors),
        "scheduler_g": sidecar.get("scheduler_g"),
    }
    if sidecar.get("optimizer_d") is not None:
        ckpt["optimizer_d"] = _restore_tensors(sidecar["optimizer_d"], tensors)
    if sidecar.get("scheduler_d") is not None:
        ckpt["scheduler_d"] = sidecar["scheduler_d"]
    disc = _prefixed_state_dict(tensors, "discriminators")
    if disc:
        ckpt["discriminators"] = disc
    return ckpt


def _add_state_tensors(
    tensors: dict[str, torch.Tensor], prefix: str, state: dict[str, torch.Tensor]
) -> None:
    for name, tensor in state.items():
        tensors[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()


def _prefixed_state_dict(
    tensors: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    prefix_dot = f"{prefix}."
    return {
        key[len(prefix_dot) :]: tensor
        for key, tensor in tensors.items()
        if key.startswith(prefix_dot)
    }


def _extract_tensors(obj: Any, prefix: str, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        key = f"{prefix}.{len(tensors)}"
        tensors[key] = obj.detach().cpu().contiguous()
        return {"__tensor__": key}
    if isinstance(obj, dict):
        return {
            "__dict__": [
                [_extract_key(key), _extract_tensors(value, prefix, tensors)]
                for key, value in obj.items()
            ]
        }
    if isinstance(obj, tuple):
        return {
            "__tuple__": [_extract_tensors(value, prefix, tensors) for value in obj]
        }
    if isinstance(obj, list):
        return [_extract_tensors(value, prefix, tensors) for value in obj]
    return obj


def _restore_tensors(obj: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            return tensors[obj["__tensor__"]]
        if "__tuple__" in obj:
            return tuple(_restore_tensors(value, tensors) for value in obj["__tuple__"])
        if "__dict__" in obj:
            return {
                _restore_key(key): _restore_tensors(value, tensors)
                for key, value in obj["__dict__"]
            }
    if isinstance(obj, list):
        return [_restore_tensors(value, tensors) for value in obj]
    return obj


def _extract_key(key: Any) -> dict[str, Any]:
    if isinstance(key, bool):
        return {"type": "bool", "value": key}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    if isinstance(key, int):
        return {"type": "int", "value": key}
    if isinstance(key, float):
        return {"type": "float", "value": key}
    if key is None:
        return {"type": "none", "value": None}
    raise TypeError(f"Unsupported checkpoint metadata key type: {type(key).__name__}")


def _restore_key(payload: dict[str, Any]) -> Any:
    key_type = payload["type"]
    value = payload["value"]
    if key_type == "str":
        return value
    if key_type == "int":
        return int(value)
    if key_type == "float":
        return float(value)
    if key_type == "bool":
        return bool(value)
    if key_type == "none":
        return None
    raise ValueError(f"Unsupported checkpoint metadata key type: {key_type!r}")
