"""Shared optional Typer helpers for scripts."""

from __future__ import annotations

from typing import Any, Callable

try:
    import typer
except ImportError:  # pragma: no cover - depends on local environment
    typer = None  # type: ignore[assignment]


def option(default: Any, *param_decls: str, help: str, **kwargs: Any) -> Any:
    if typer is None:
        return default
    return typer.Option(default, *param_decls, help=help, **kwargs)


def run(main: Callable[..., None]) -> None:
    if typer is None:
        raise SystemExit(
            "Typer is required to run this CLI. Install typer, then rerun."
        )
    typer.run(main)
