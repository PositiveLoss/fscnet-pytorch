"""Model size presets for FSC-Net training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import FSCNetConfig


@dataclass(frozen=True)
class FSCNetModelPreset:
    name: str
    description: str
    config: FSCNetConfig
    progressive_windows: tuple[int, ...]
    suggested_batch_size: int


MODEL_PRESETS: dict[str, FSCNetModelPreset] = {
    "tiny": FSCNetModelPreset(
        name="tiny",
        description="Fast smoke tests and low-memory experiments.",
        config=FSCNetConfig(
            channels=24,
            num_blocks=3,
            rnn_hidden=32,
            attention_heads=3,
            time_attention="v1",
        ),
        progressive_windows=(65, 17, 1),
        suggested_batch_size=8,
    ),
    "small": FSCNetModelPreset(
        name="small",
        description="Light training runs with lower memory than the default.",
        config=FSCNetConfig(
            channels=32,
            num_blocks=4,
            rnn_hidden=48,
            attention_heads=4,
            time_attention="v1",
        ),
        progressive_windows=(129, 33, 9, 1),
        suggested_batch_size=6,
    ),
    "compact": FSCNetModelPreset(
        name="compact",
        description="Current default architecture.",
        config=FSCNetConfig(),
        progressive_windows=(257, 65, 17, 5, 1),
        suggested_batch_size=4,
    ),
    "medium": FSCNetModelPreset(
        name="medium",
        description="Higher-capacity model for larger datasets and GPUs.",
        config=FSCNetConfig(
            channels=64,
            num_blocks=6,
            rnn_hidden=96,
            attention_heads=4,
            time_attention="v2",
            time_attention_qk_norm=False,
            time_attention_rope=False,
        ),
        progressive_windows=(257, 129, 65, 17, 5, 1),
        suggested_batch_size=2,
    ),
    "large": FSCNetModelPreset(
        name="large",
        description="Large experiment; expect substantially higher memory use.",
        config=FSCNetConfig(
            channels=96,
            num_blocks=6,
            rnn_hidden=128,
            attention_heads=8,
            time_attention="v2",
            time_attention_qk_norm=False,
            time_attention_rope=False,
        ),
        progressive_windows=(257, 129, 65, 17, 5, 1),
        suggested_batch_size=1,
    ),
}


def model_preset_names() -> tuple[str, ...]:
    return tuple(MODEL_PRESETS)


def get_model_preset(name: str) -> FSCNetModelPreset:
    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(model_preset_names())
        raise ValueError(
            f"Unknown model preset {name!r}. Valid presets: {valid}"
        ) from exc


def parse_progressive_windows(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("At least one progressive window is required")
    return vals


def resolve_model_config(
    preset_name: str,
    overrides: dict[str, Any] | None = None,
    progressive_windows: str | tuple[int, ...] | None = None,
) -> tuple[FSCNetConfig, tuple[int, ...]]:
    preset = get_model_preset(preset_name)
    data = preset.config.to_dict()
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    cfg = FSCNetConfig.from_dict(data)

    if progressive_windows is None:
        windows = preset.progressive_windows
    elif isinstance(progressive_windows, str):
        windows = parse_progressive_windows(progressive_windows)
    else:
        windows = tuple(int(w) for w in progressive_windows)

    if len(windows) != cfg.num_blocks:
        raise ValueError(
            f"Need one progressive window per block: got {len(windows)} windows "
            f"for num_blocks={cfg.num_blocks}"
        )
    return cfg, windows
