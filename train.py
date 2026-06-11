"""Train an FSC-Net style speech bandwidth extension model.

Example:
  uv run python train.py \
    --train_manifest train.jsonl --valid_manifest valid.jsonl \
    --input_sr 4000 --target_sr 48000 --epochs 100 --batch_size 8
"""

from __future__ import annotations

import atexit
import importlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Iterator, Literal, Sequence, cast

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fscnet_pytorch.audio import complex_to_ri, resample_audio, stft_complex
from fscnet_pytorch.checkpoint import (
    checkpoint_sidecar_path,
    load_training_checkpoint,
    save_training_checkpoint,
)
from fscnet_pytorch.cli import option, run
from fscnet_pytorch.config import (
    get_model_preset,
    model_preset_names,
    resolve_model_config,
)
from fscnet_pytorch.data import (
    BandwidthExtensionDataset,
    StreamingBandwidthExtensionDataset,
    count_manifest_rows,
)
from fscnet_pytorch.discriminator import (
    MultiScaleDiscriminator,
    discriminator_lsgan_loss,
    generator_lsgan_fm_loss,
    set_requires_grad,
)
from fscnet_pytorch.losses import StageLossWeights, StageReconstructionLoss
from fscnet_pytorch.kernels import (
    activated_kernel_names,
    pyptx_disabled,
    reset_activated_kernel_names,
)
from fscnet_pytorch.model import FSCNet, count_parameters


MODEL_OVERRIDE_FIELDS = (
    "target_sr",
    "input_sr",
    "n_fft",
    "win_length",
    "hop_length",
    "subbands",
    "channels",
    "num_blocks",
    "rnn_hidden",
    "attention_heads",
    "time_attention",
    "time_attention_qk_norm",
    "time_attention_rope",
    "ffc_ratio",
    "dropout",
)

TrackMetricValue = int | float | str | bool | None
TimeAttention = Literal["v1", "v2"]
PrecisionMode = Literal["fp32", "fp16", "bf16"]
OptionalMetricState = dict[str, Any]


class TrackioRun:
    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.finished = False
        self.disabled = False

    def log(self, metrics: dict[str, TrackMetricValue]) -> None:
        if self.disabled or self.finished:
            return
        try:
            log_fn = getattr(self.module, "log")
            log_fn(metrics)
        except Exception as exc:
            self.disabled = True
            print(f"Trackio logging disabled after error: {exc}")

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            finish_fn = getattr(self.module, "finish")
            finish_fn()
        except Exception as exc:
            print(f"Trackio finish failed: {exc}")


def print_model_sizes() -> None:
    for name in model_preset_names():
        preset = get_model_preset(name)
        cfg = preset.config
        print(
            f"{name}: {preset.description} "
            f"channels={cfg.channels} blocks={cfg.num_blocks} "
            f"hidden={cfg.rnn_hidden} heads={cfg.attention_heads} "
            f"attention={cfg.time_attention} windows={preset.progressive_windows} "
            f"suggested_batch_size={preset.suggested_batch_size}"
        )


def init_trackio(
    args: SimpleNamespace,
    model_config: dict[str, Any],
    windows: Sequence[int],
    parameter_count: int,
) -> TrackioRun | None:
    if not args.trackio:
        return None
    try:
        trackio = importlib.import_module("trackio")
    except ImportError as exc:
        raise RuntimeError(
            "Trackio logging was requested with --trackio, but the trackio package "
            "is not installed in the current environment."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": args.trackio_project,
        "config": {
            "args": vars(args),
            "model": model_config,
            "windows": tuple(windows),
            "parameters": parameter_count,
        },
    }
    for arg_name, key in (
        ("trackio_name", "name"),
        ("trackio_group", "group"),
        ("trackio_space_id", "space_id"),
        ("trackio_server_url", "server_url"),
    ):
        value = getattr(args, arg_name)
        if value:
            init_kwargs[key] = value

    init_fn = getattr(trackio, "init")
    init_fn(**init_kwargs)
    run = TrackioRun(trackio)
    atexit.register(run.finish)
    return run


def scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def gpu_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=na"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"gpu_mem={allocated:.2f}G/{reserved:.2f}G peak={peak:.2f}G"


def train_stage_loss_summary(logs: dict[str, float]) -> str:
    stage_items = sorted(
        (key, value) for key, value in logs.items() if key.startswith("stage_")
    )
    return " ".join(
        f"{key.removesuffix('_loss')}={value:.4f}" for key, value in stage_items
    )


def gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def cycle_loader(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError(
                "Training loader produced no batches. Check that the training "
                "manifest is non-empty and contains enough samples for batch_size "
                "with drop_last=True."
            )


def print_kernel_configuration(
    args: SimpleNamespace,
    device: torch.device,
    precision_name: str,
) -> None:
    kernel_precision = precision_name in ("fp32", "fp16", "bf16")
    if pyptx_disabled():
        progressive = norm = rope_qk = "disabled"
    else:
        progressive = (
            "enabled" if device.type == "cuda" and kernel_precision else "inactive"
        )
        norm = (
            "enabled"
            if os.environ.get("FSCNET_ENABLE_PYPTX_NORM") == "1"
            and device.type == "cuda"
            and kernel_precision
            else "inactive"
        )
        rope_qk = (
            "enabled"
            if os.environ.get("FSCNET_ENABLE_PYPTX_ROPE_QK") == "1"
            and device.type == "cuda"
            and kernel_precision
            and args.time_attention == "v2"
            and args.time_attention_qk_norm
            and args.time_attention_rope
            else "inactive"
        )
    print(
        "Kernel configuration: "
        f"progressive_targets={progressive}, "
        f"global_layer_norm={norm}, "
        f"rope_qk_norm={rope_qk}"
    )


def print_runtime_kernel_activations(prefix: str = "Activated kernels") -> None:
    names = activated_kernel_names()
    if names:
        print(f"{prefix}: {', '.join(names)}")
    else:
        print(f"{prefix}: none")


def resolve_checkpoint_to_resume(args: SimpleNamespace, out_path: Path) -> Path | None:
    if args.resume:
        return Path(args.resume)
    if not args.auto_resume:
        return None
    candidate = out_path / "last.safetensors"
    if candidate.exists():
        return candidate
    return None


def init_optional_eval_metrics(
    enabled: bool,
    target_sr: int,
) -> OptionalMetricState:
    state: OptionalMetricState = {
        "enabled": enabled,
        "target_sr": target_sr,
        "pesq": None,
        "pesq_note_printed": False,
        "pesq_failed_samples": 0,
    }
    if not enabled:
        return state

    try:
        pesq_module = importlib.import_module("pesq")
        state["pesq"] = getattr(pesq_module, "pesq")
    except Exception as exc:
        state["pesq_error"] = str(exc)

    return state


def maybe_pesq_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    state: OptionalMetricState,
) -> tuple[float, int]:
    pesq_fn = state.get("pesq")
    if pesq_fn is None:
        if not state.get("pesq_note_printed"):
            print(
                f"PESQ evaluation unavailable: {state.get('pesq_error', 'not installed')}"
            )
            state["pesq_note_printed"] = True
        return 0.0, 0

    eval_sr = int(state["target_sr"])
    mode = "wb"
    if eval_sr not in (8000, 16000):
        pred = resample_audio(pred.detach().cpu(), eval_sr, 16000)
        target = resample_audio(target.detach().cpu(), eval_sr, 16000)
        eval_sr = 16000
    else:
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        mode = "nb" if eval_sr == 8000 else "wb"

    total = 0.0
    count = 0
    for pred_item, target_item in zip(pred, target):
        try:
            total += float(
                pesq_fn(
                    eval_sr,
                    target_item.to(torch.float32).numpy(),
                    pred_item.to(torch.float32).numpy(),
                    mode,
                )
            )
            count += 1
        except Exception as exc:
            state["pesq_failed_samples"] = int(state.get("pesq_failed_samples", 0)) + 1
            if not state.get("pesq_note_printed"):
                print(f"PESQ evaluation skipping failed samples after error: {exc}")
                state["pesq_note_printed"] = True
            continue
    return total, count


def cosine_warmup_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return fn


def resolve_precision(
    precision: PrecisionMode, device: torch.device
) -> tuple[bool, torch.dtype | None, bool, str]:
    """Return autocast/scaler settings for the requested training precision."""
    if device.type != "cuda":
        return False, None, False, "fp32"
    if precision == "fp32":
        return False, None, False, "fp32"
    if precision == "fp16":
        return True, torch.float16, True, "fp16"
    if precision == "bf16":
        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "CUDA bf16 training was requested, but this GPU does not support bf16."
            )
        return True, torch.bfloat16, False, "bf16"
    raise ValueError(f"Unknown precision={precision!r}")


def save_checkpoint(
    path: Path,
    model: FSCNet,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g,
    epoch: int,
    step: int,
    args: SimpleNamespace,
    windows: Sequence[int],
    discriminators: nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scheduler_d=None,
) -> None:
    save_training_checkpoint(
        path,
        model=model,
        config=model.cfg.to_dict(),
        windows=tuple(windows),
        epoch=epoch,
        step=step,
        args=vars(args),
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminators=discriminators,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
    )


def delete_checkpoint(path: Path) -> None:
    path.unlink(missing_ok=True)
    checkpoint_sidecar_path(path).unlink(missing_ok=True)


def prune_numbered_checkpoints(out_path: Path, keep_n_checkpoints: int) -> None:
    if keep_n_checkpoints <= 0:
        return
    checkpoints = sorted(out_path.glob("checkpoint_epoch_*.safetensors"))
    for checkpoint in checkpoints[:-keep_n_checkpoints]:
        delete_checkpoint(checkpoint)


@torch.no_grad()
def validate(
    model: FSCNet,
    loss_fn: StageReconstructionLoss,
    loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype | None,
    optional_eval_metrics: OptionalMetricState,
) -> dict[str, float]:
    model.eval()
    total_recon = 0.0
    total_lsd = 0.0
    total_pesq = 0.0
    count = 0
    pesq_count = 0
    eval_metrics_enabled = bool(optional_eval_metrics.get("enabled"))
    for batch in tqdm(loader, desc="valid", leave=True):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred_wav: torch.Tensor | None = None
        lsd: torch.Tensor | None = None
        with torch.autocast(
            device_type="cuda",
            enabled=autocast_enabled and device.type == "cuda",
            dtype=autocast_dtype,
        ):
            pred_stages, input_ri = model(lr, return_all=True)
            target_ri = complex_to_ri(
                stft_complex(
                    hr,
                    model.cfg.n_fft,
                    model.cfg.hop_length,
                    model.cfg.win_length,
                    center=model.cfg.center,
                )
            )
            loss, _, _, _ = loss_fn(pred_stages, input_ri, target_ri, hr.shape[-1])
            if eval_metrics_enabled:
                pred_wav = loss_fn._wav_from_ri(pred_stages[-1], hr.shape[-1])
                lsd = loss_fn.lsd(pred_wav, hr)
        batch_size = lr.shape[0]
        total_recon += float(loss.detach().cpu()) * batch_size
        count += lr.shape[0]
        if eval_metrics_enabled:
            assert pred_wav is not None
            assert lsd is not None
            total_lsd += float(lsd.detach().cpu()) * batch_size
            pesq_total_batch, pesq_count_batch = maybe_pesq_batch(
                pred_wav, hr, optional_eval_metrics
            )
            total_pesq += pesq_total_batch
            pesq_count += pesq_count_batch
    model.train()
    metrics = {"valid_recon_loss": total_recon / max(1, count)}
    if eval_metrics_enabled:
        metrics["valid_lsd"] = total_lsd / max(1, count)
    if pesq_count > 0:
        metrics["valid_pesq"] = total_pesq / pesq_count
    return metrics


def run_validation(
    *,
    model: FSCNet,
    recon_loss_fn: StageReconstructionLoss,
    valid_loader: DataLoader,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype | None,
    optional_eval_metrics: OptionalMetricState,
    tracker: TrackioRun | None,
    epoch: int,
    global_step: int,
    args: SimpleNamespace,
    out_path: Path,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g,
    windows: Sequence[int],
    best_valid_recon_loss: float,
    best_metrics_path: Path,
    discriminators: nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scheduler_d=None,
) -> float:
    started = time.perf_counter()
    print(
        f"validation start step={global_step} epoch={epoch + 1} "
        f"batches={len(valid_loader)}",
        flush=True,
    )
    metrics = validate(
        model,
        recon_loss_fn,
        valid_loader,
        device,
        autocast_enabled,
        autocast_dtype,
        optional_eval_metrics,
    )
    elapsed = time.perf_counter() - started
    print(f"validation done step={global_step} seconds={elapsed:.1f} metrics={metrics}")
    if tracker is not None:
        valid_metrics: dict[str, TrackMetricValue] = {
            "step": global_step,
            "epoch": epoch + 1,
        }
        for key, value in metrics.items():
            valid_metrics[f"valid/{key.removeprefix('valid_')}"] = value
        tracker.log(valid_metrics)
    valid_recon_loss = metrics["valid_recon_loss"]
    if args.save_best_checkpoint and valid_recon_loss < best_valid_recon_loss:
        best_valid_recon_loss = valid_recon_loss
        save_checkpoint(
            out_path / "best.safetensors",
            model,
            optimizer_g,
            scheduler_g,
            epoch,
            global_step,
            args,
            windows,
            discriminators=discriminators,
            optimizer_d=optimizer_d,
            scheduler_d=scheduler_d,
        )
        best_metrics_path.write_text(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "valid_recon_loss": best_valid_recon_loss,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if tracker is not None:
            tracker.log(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "checkpoint/best_epoch": epoch + 1,
                    "checkpoint/best_valid_recon_loss": best_valid_recon_loss,
                }
            )
    return best_valid_recon_loss


def main(
    train_manifest: str | None = option(
        None,
        "--train-manifest",
        "--train_manifest",
        help="jsonl/csv/txt manifest for training audio",
    ),
    valid_manifest: str | None = option(
        None,
        "--valid-manifest",
        "--valid_manifest",
        help="optional validation manifest",
    ),
    out_dir: str = option(
        "runs/fscnet", "--out-dir", "--out_dir", help="checkpoint/log directory"
    ),
    model_size: str = option(
        "compact",
        "--model-size",
        "--model_size",
        help="model size preset; explicit architecture flags override it",
    ),
    list_model_sizes: bool = option(
        False,
        "--list-model-sizes",
        "--list_model_sizes",
        help="print available model size presets and exit",
    ),
    target_sr: int | None = option(
        None, "--target-sr", "--target_sr", help="target SR"
    ),
    input_sr: int | None = option(None, "--input-sr", "--input_sr", help="input SR"),
    segment_seconds: float = option(
        2.0, "--segment-seconds", "--segment_seconds", help="segment seconds", min=0.0
    ),
    n_fft: int | None = option(None, "--n-fft", "--n_fft", help="STFT FFT size"),
    win_length: int | None = option(
        None, "--win-length", "--win_length", help="STFT window length"
    ),
    hop_length: int | None = option(
        None, "--hop-length", "--hop_length", help="STFT hop length"
    ),
    subbands: int | None = option(None, "--subbands", help="channel-wise subbands"),
    channels: int | None = option(None, "--channels", help="model channels"),
    num_blocks: int | None = option(
        None, "--num-blocks", "--num_blocks", help="number of TF-FFC blocks"
    ),
    rnn_hidden: int | None = option(
        None, "--rnn-hidden", "--rnn_hidden", help="BLSTM hidden size"
    ),
    attention_heads: int | None = option(
        None,
        "--attention-heads",
        "--attention_heads",
        help="time attention heads",
    ),
    time_attention: TimeAttention | None = option(
        None, "--time-attention", "--time_attention", help="time attention variant"
    ),
    time_attention_qk_norm: bool | None = option(
        None,
        "--time-attention-qk-norm/--no-time-attention-qk-norm",
        "--time_attention_qk_norm/--no-time_attention_qk_norm",
        help="enable QK normalization for --time_attention v2",
    ),
    time_attention_rope: bool | None = option(
        None,
        "--time-attention-rope/--no-time-attention-rope",
        "--time_attention_rope/--no-time_attention_rope",
        help="enable rotary time positions for --time_attention v2",
    ),
    ffc_ratio: float | None = option(
        None, "--ffc-ratio", "--ffc_ratio", help="FFC ratio"
    ),
    dropout: float | None = option(None, "--dropout", help="dropout"),
    progressive_windows: str | None = option(
        None,
        "--progressive-windows",
        "--progressive_windows",
        help="comma-separated windows; defaults to the selected model size preset",
    ),
    epochs: int = option(100, "--epochs", help="training epochs", min=1),
    batch_size: int | None = option(
        None,
        "--batch-size",
        "--batch_size",
        help="defaults to the selected model size preset recommendation",
    ),
    valid_batch_size: int | None = option(
        None,
        "--valid-batch-size",
        "--valid_batch_size",
        help="validation batch size; defaults to --batch_size",
        min=1,
    ),
    num_workers: int = option(
        4, "--num-workers", "--num_workers", help="DataLoader workers", min=0
    ),
    stream_train_manifest: bool = option(
        False,
        "--stream-train-manifest/--no-stream-train-manifest",
        "--stream_train_manifest/--no_stream_train_manifest",
        help="stream the training manifest instead of loading all rows into memory",
    ),
    train_shuffle_buffer: int = option(
        8192,
        "--train-shuffle-buffer",
        "--train_shuffle_buffer",
        help="bounded shuffle buffer for --stream_train_manifest",
        min=1,
    ),
    steps_per_epoch: int | None = option(
        None,
        "--steps-per-epoch",
        "--steps_per_epoch",
        help="optimizer steps per epoch; useful with streaming train manifests",
        min=1,
    ),
    lr_g: float = option(2e-4, "--lr-g", "--lr_g", help="generator LR", min=0.0),
    lr_d: float = option(1e-4, "--lr-d", "--lr_d", help="discriminator LR", min=0.0),
    warmup_steps: int = option(
        2000, "--warmup-steps", "--warmup_steps", help="LR warmup steps", min=0
    ),
    min_lr_ratio: float = option(
        0.05, "--min-lr-ratio", "--min_lr_ratio", help="minimum LR ratio", min=0.0
    ),
    clip_grad_norm: float = option(
        5.0, "--clip-grad-norm", "--clip_grad_norm", help="gradient clipping", min=0.0
    ),
    precision: PrecisionMode = option(
        "fp32",
        "--precision",
        help="training precision: fp32, fp16, or bf16",
    ),
    trackio: bool = option(False, "--trackio", help="enable Trackio logging"),
    trackio_project: str = option(
        "fscnet", "--trackio-project", "--trackio_project", help="Trackio project name"
    ),
    trackio_name: str | None = option(
        None, "--trackio-name", "--trackio_name", help="optional Trackio run name"
    ),
    trackio_group: str | None = option(
        None, "--trackio-group", "--trackio_group", help="optional Trackio run group"
    ),
    trackio_space_id: str | None = option(
        None,
        "--trackio-space-id",
        "--trackio_space_id",
        help="optional Hugging Face Space id for hosted Trackio logs",
    ),
    trackio_server_url: str | None = option(
        None,
        "--trackio-server-url",
        "--trackio_server_url",
        help="optional self-hosted Trackio server URL",
    ),
    trackio_log_every: int = option(
        1,
        "--trackio-log-every",
        "--trackio_log_every",
        help="log training metrics every N optimizer steps; <=0 disables step logs",
    ),
    train_log_every: int = option(
        1,
        "--train-log-every",
        "--train_log_every",
        help="print training metrics every N optimizer steps; <=0 disables console logs",
    ),
    mrstft_weight: float = option(
        1.0, "--mrstft-weight", "--mrstft_weight", help="MR-STFT loss weight"
    ),
    lsd_weight: float = option(
        5.0, "--lsd-weight", "--lsd_weight", help="LSD loss weight"
    ),
    complex_l1_weight: float = option(
        0.0, "--complex-l1-weight", "--complex_l1_weight", help="complex L1 loss weight"
    ),
    mrstft_fft_sizes: str = option(
        "512,1024,2048",
        "--mrstft-fft-sizes",
        "--mrstft_fft_sizes",
        help="MR-STFT FFT sizes",
    ),
    adv_weight: float = option(
        0.34,
        "--adv-weight",
        "--adv_weight",
        help="set >0 to enable per-stage LSGAN",
        min=0.0,
    ),
    fm_weight: float = option(
        0.1, "--fm-weight", "--fm_weight", help="feature matching weight"
    ),
    adv_start_step: int = option(
        0, "--adv-start-step", "--adv_start_step", help="adversarial start step", min=0
    ),
    disc_scales: int = option(
        3, "--disc-scales", "--disc_scales", help="discriminator scales", min=1
    ),
    disc_channels: int = option(
        16, "--disc-channels", "--disc_channels", help="discriminator channels", min=1
    ),
    save_every: int = option(
        1, "--save-every", "--save_every", help="save every N epochs", min=1
    ),
    keep_n_checkpoints: int = option(
        0,
        "--keep-n-checkpoints",
        "--keep_n_checkpoints",
        help="keep only the newest N numbered epoch checkpoints; 0 keeps all",
        min=0,
    ),
    save_best_checkpoint: bool = option(
        False,
        "--save-best-checkpoint/--no-save-best-checkpoint",
        "--save_best_checkpoint/--no_save_best_checkpoint",
        help="save best.safetensors when validation reconstruction loss improves",
    ),
    valid_every: int = option(
        1, "--valid-every", "--valid_every", help="validate every N epochs", min=1
    ),
    eval_steps: int = option(
        0,
        "--eval-steps",
        "--eval_steps",
        help="validate every N optimizer steps; 0 keeps epoch-based validation",
        min=0,
    ),
    eval_metrics: bool = option(
        True,
        "--eval-metrics/--no-eval-metrics",
        "--eval_metrics/--no_eval_metrics",
        help="compute paper evaluation metrics after validation epochs",
    ),
    seed: int = option(1234, "--seed", help="random seed"),
    torch_num_threads: int = option(
        1,
        "--torch-num-threads",
        "--torch_num_threads",
        help="CPU intra-op threads; 1 avoids oversubscription on many machines",
        min=0,
    ),
    resume: str | None = option(None, "--resume", help="checkpoint to resume"),
    auto_resume: bool = option(
        True,
        "--auto-resume/--no-auto-resume",
        "--auto_resume/--no_auto_resume",
        help="resume from out_dir/last.safetensors when it exists and --resume is unset",
    ),
) -> None:
    """Train FSC-Net for speech bandwidth extension."""
    args = SimpleNamespace(**locals())
    if args.list_model_sizes:
        print_model_sizes()
        return
    if args.train_manifest is None:
        raise ValueError(
            "--train_manifest is required unless --list_model_sizes is used"
        )

    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    overrides = {field: getattr(args, field) for field in MODEL_OVERRIDE_FIELDS}
    cfg, windows = resolve_model_config(
        args.model_size,
        overrides=overrides,
        progressive_windows=args.progressive_windows,
    )
    args.time_attention = cfg.time_attention
    args.time_attention_qk_norm = cfg.time_attention_qk_norm
    args.time_attention_rope = cfg.time_attention_rope
    if args.batch_size is None:
        args.batch_size = get_model_preset(args.model_size).suggested_batch_size
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size
    autocast_enabled, autocast_dtype, scaler_enabled, precision_name = (
        resolve_precision(args.precision, device)
    )
    args.precision = precision_name
    reset_activated_kernel_names()
    print_kernel_configuration(args, device, precision_name)
    parsed_mrstft_fft_sizes = tuple(
        int(x) for x in args.mrstft_fft_sizes.split(",") if x
    )
    model = FSCNet(cfg).to(device)
    parameter_count = count_parameters(model)
    print(f"Model preset: {args.model_size}; windows={windows}; config={cfg.to_dict()}")
    print(f"Generator parameters: {parameter_count / 1e6:.3f} M")

    if args.stream_train_manifest:
        train_row_count = (
            None
            if args.steps_per_epoch is not None
            else count_manifest_rows(args.train_manifest)
        )
        if train_row_count == 0:
            raise ValueError(f"No audio items found in {args.train_manifest}")
        train_ds = StreamingBandwidthExtensionDataset(
            args.train_manifest,
            target_sr=cfg.target_sr,
            input_sr=cfg.input_sr,
            segment_seconds=args.segment_seconds,
            normalize=True,
            random_crop=True,
            shuffle_buffer_size=args.train_shuffle_buffer,
            row_count=train_row_count,
        )
    else:
        train_ds = BandwidthExtensionDataset(
            args.train_manifest,
            target_sr=cfg.target_sr,
            input_sr=cfg.input_sr,
            segment_seconds=args.segment_seconds,
            normalize=True,
            random_crop=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.stream_train_manifest,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = None
    if args.valid_manifest:
        valid_ds = BandwidthExtensionDataset(
            args.valid_manifest,
            target_sr=cfg.target_sr,
            input_sr=cfg.input_sr,
            segment_seconds=args.segment_seconds,
            normalize=True,
            random_crop=False,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    weights = StageLossWeights(
        args.mrstft_weight, args.lsd_weight, args.complex_l1_weight
    )
    recon_loss_fn = StageReconstructionLoss(
        cfg, windows, weights=weights, mrstft_fft_sizes=parsed_mrstft_fft_sizes
    ).to(device)

    discriminators = None
    optimizer_d = None
    scheduler_d = None
    if args.adv_weight > 0:
        discriminators = nn.ModuleList(
            [
                MultiScaleDiscriminator(
                    waveform_channels=1,
                    spec_channels=2,
                    num_scales=args.disc_scales,
                    base_channels=args.disc_channels,
                )
                for _ in windows
            ]
        ).to(device)
        optimizer_d = torch.optim.AdamW(
            discriminators.parameters(),
            lr=args.lr_d,
            betas=(0.8, 0.99),
            weight_decay=1e-4,
        )

    optimizer_g = torch.optim.AdamW(
        model.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=1e-4
    )
    if args.steps_per_epoch is None:
        train_steps_per_epoch = len(train_loader)
    else:
        if not args.stream_train_manifest:
            natural_train_steps = len(train_loader)
            if args.steps_per_epoch > natural_train_steps:
                raise ValueError(
                    f"--steps_per_epoch={args.steps_per_epoch} exceeds the finite "
                    f"training loader length ({natural_train_steps}). Use "
                    "--stream_train_manifest to cycle the manifest, lower "
                    "--steps_per_epoch, or omit it."
                )
        train_steps_per_epoch = args.steps_per_epoch
    cycle_train_loader = (
        args.stream_train_manifest and args.steps_per_epoch is not None
    )
    total_steps = max(1, train_steps_per_epoch * args.epochs)
    print(
        "Training schedule: "
        f"steps_per_epoch={train_steps_per_epoch}, "
        f"total_steps={total_steps}, "
        f"stream_train_manifest={args.stream_train_manifest}, "
        f"cycle_train_loader={cycle_train_loader}",
        flush=True,
    )
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        cosine_warmup_lambda(total_steps, args.warmup_steps, args.min_lr_ratio),
    )
    if optimizer_d is not None:
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optimizer_d,
            cosine_warmup_lambda(total_steps, args.warmup_steps, args.min_lr_ratio),
        )

    start_epoch = 0
    global_step = 0
    resume_checkpoint = resolve_checkpoint_to_resume(args, out_path)
    if resume_checkpoint is not None:
        ckpt = load_training_checkpoint(resume_checkpoint)
        model.load_state_dict(ckpt["model"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if ckpt.get("scheduler_g"):
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if discriminators is not None and ckpt.get("discriminators"):
            discriminators.load_state_dict(ckpt["discriminators"])
        if optimizer_d is not None and ckpt.get("optimizer_d"):
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if scheduler_d is not None and ckpt.get("scheduler_d"):
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        print(
            f"Resumed from {resume_checkpoint} at epoch {start_epoch + 1} "
            f"with global_step={global_step}"
        )

    (out_path / "config.json").write_text(
        json.dumps(
            {"model": cfg.to_dict(), "windows": windows, "args": vars(args)}, indent=2
        ),
        encoding="utf-8",
    )

    scaler = GradScaler("cuda", enabled=scaler_enabled and device.type == "cuda")
    model.train()
    tracker = init_trackio(args, cfg.to_dict(), windows, parameter_count)
    optional_eval_metrics = init_optional_eval_metrics(
        args.eval_metrics and valid_loader is not None,
        cfg.target_sr,
    )
    best_valid_recon_loss = math.inf
    best_metrics_path = out_path / "best_metrics.json"
    if args.save_best_checkpoint and best_metrics_path.exists():
        try:
            best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            best_valid_recon_loss = float(best_metrics["valid_recon_loss"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Ignoring unreadable best checkpoint metrics: {exc}")
    printed_kernel_runtime = False

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        data_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_epoch_loader = (
            cycle_loader(train_loader) if cycle_train_loader else train_loader
        )
        pbar = tqdm(
            train_epoch_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            total=train_steps_per_epoch,
        )
        for step_in_epoch, batch in enumerate(
            islice(pbar, train_steps_per_epoch), start=1
        ):
            data_time = time.perf_counter() - data_started
            step_started = time.perf_counter()
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=autocast_enabled and device.type == "cuda",
                dtype=autocast_dtype,
            ):
                pred_stages, input_ri = model(lr, return_all=True)
                target_ri = complex_to_ri(
                    stft_complex(
                        hr, cfg.n_fft, cfg.hop_length, cfg.win_length, center=cfg.center
                    )
                )
                recon_loss, logs, pred_wavs, _target_wavs = recon_loss_fn(
                    pred_stages, input_ri, target_ri, hr.shape[-1]
                )

            d_loss_val = 0.0
            use_adv = discriminators is not None and global_step >= args.adv_start_step
            if use_adv:
                assert optimizer_d is not None
                discriminators_active = cast(nn.ModuleList, discriminators)
                set_requires_grad(discriminators_active, True)
                optimizer_d.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    enabled=autocast_enabled and device.type == "cuda",
                    dtype=autocast_dtype,
                ):
                    d_loss = lr.new_tensor(0.0)
                    for disc, fake_wav, fake_ri in zip(
                        discriminators_active, pred_wavs, pred_stages
                    ):
                        d_loss = d_loss + discriminator_lsgan_loss(
                            cast(MultiScaleDiscriminator, disc),
                            hr,
                            target_ri,
                            fake_wav,
                            fake_ri,
                        )
                    d_loss = d_loss / len(pred_wavs)
                cast(torch.Tensor, scaler.scale(d_loss)).backward()
                scaler.unscale_(optimizer_d)
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        discriminators_active.parameters(), args.clip_grad_norm
                    )
                d_step_ok = gradients_are_finite(discriminators_active.parameters())
                scaler.step(optimizer_d)
                if scheduler_d is not None and d_step_ok:
                    scheduler_d.step()
                d_loss_val = float(d_loss.detach().cpu())

            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                enabled=autocast_enabled and device.type == "cuda",
                dtype=autocast_dtype,
            ):
                loss_g = recon_loss
                adv_val = lr.new_tensor(0.0)
                fm_val = lr.new_tensor(0.0)
                if use_adv:
                    discriminators_active = cast(nn.ModuleList, discriminators)
                    set_requires_grad(discriminators_active, False)
                    for disc, fake_wav, fake_ri in zip(
                        discriminators_active, pred_wavs, pred_stages
                    ):
                        adv, fm = generator_lsgan_fm_loss(
                            cast(MultiScaleDiscriminator, disc),
                            hr,
                            target_ri,
                            fake_wav,
                            fake_ri,
                            fm_weight=args.fm_weight,
                        )
                        adv_val = adv_val + adv
                        fm_val = fm_val + fm
                    adv_val = adv_val / len(pred_wavs)
                    fm_val = fm_val / len(pred_wavs)
                    loss_g = loss_g + args.adv_weight * (adv_val + fm_val)

            cast(torch.Tensor, scaler.scale(loss_g)).backward()
            scaler.unscale_(optimizer_g)
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            g_step_ok = gradients_are_finite(model.parameters())
            scaler.step(optimizer_g)
            scaler.update()
            if g_step_ok:
                scheduler_g.step()
            if use_adv:
                discriminators_active = cast(nn.ModuleList, discriminators)
                set_requires_grad(discriminators_active, True)

            global_step += 1
            step_time = time.perf_counter() - step_started
            epoch_elapsed = max(1e-6, time.perf_counter() - epoch_started)
            remaining_steps = max(0, train_steps_per_epoch - step_in_epoch)
            eta_seconds = remaining_steps * (epoch_elapsed / max(1, step_in_epoch))
            batch_samples = int(lr.shape[0])
            pbar.set_postfix(
                recon=f"{logs['recon_loss']:.4f}",
                g=f"{float(loss_g.detach().cpu()):.4f}",
                d=f"{d_loss_val:.4f}",
                lr=f"{scheduler_g.get_last_lr()[0]:.2e}",
                mem=gpu_memory_summary(device),
            )
            if (
                tracker is not None
                and args.trackio_log_every > 0
                and global_step % args.trackio_log_every == 0
            ):
                train_metrics: dict[str, TrackMetricValue] = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train/recon_loss": scalar_float(logs["recon_loss"]),
                    "train/g_loss": float(loss_g.detach().cpu()),
                    "train/d_loss": d_loss_val,
                    "train/adv_loss": float(adv_val.detach().cpu()),
                    "train/fm_loss": float(fm_val.detach().cpu()),
                    "train/lr_g": scalar_float(scheduler_g.get_last_lr()[0]),
                    "train/use_adv": use_adv,
                }
                if scheduler_d is not None:
                    train_metrics["train/lr_d"] = scalar_float(
                        scheduler_d.get_last_lr()[0]
                    )
                for key, value in logs.items():
                    if key != "recon_loss":
                        train_metrics[f"train/{key}"] = scalar_float(value)
                tracker.log(train_metrics)
            if not printed_kernel_runtime:
                print_runtime_kernel_activations()
                printed_kernel_runtime = True
            if args.train_log_every > 0 and (
                global_step == 1 or global_step % args.train_log_every == 0
            ):
                stage_losses = train_stage_loss_summary(logs)
                print(
                    "train "
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step_in_epoch}/{train_steps_per_epoch} "
                    f"global_step={global_step} "
                    f"samples={batch_samples} "
                    f"recon={scalar_float(logs['recon_loss']):.4f} "
                    f"g={float(loss_g.detach().cpu()):.4f} "
                    f"d={d_loss_val:.4f} "
                    f"adv={float(adv_val.detach().cpu()):.4f} "
                    f"fm={float(fm_val.detach().cpu()):.4f} "
                    f"lr={scalar_float(scheduler_g.get_last_lr()[0]):.2e} "
                    f"data_s={data_time:.3f} "
                    f"step_s={step_time:.3f} "
                    f"steps_per_sec={step_in_epoch / epoch_elapsed:.3f} "
                    f"samples_per_sec={batch_samples / max(1e-6, step_time):.2f} "
                    f"eta_min={eta_seconds / 60.0:.1f} "
                    f"{gpu_memory_summary(device)} "
                    f"{stage_losses}",
                    flush=True,
                )
            data_started = time.perf_counter()
            if (
                valid_loader is not None
                and args.eval_steps > 0
                and global_step % args.eval_steps == 0
            ):
                best_valid_recon_loss = run_validation(
                    model=model,
                    recon_loss_fn=recon_loss_fn,
                    valid_loader=valid_loader,
                    device=device,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    optional_eval_metrics=optional_eval_metrics,
                    tracker=tracker,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                    out_path=out_path,
                    optimizer_g=optimizer_g,
                    scheduler_g=scheduler_g,
                    windows=windows,
                    best_valid_recon_loss=best_valid_recon_loss,
                    best_metrics_path=best_metrics_path,
                    discriminators=discriminators,
                    optimizer_d=optimizer_d,
                    scheduler_d=scheduler_d,
                )

        if (
            valid_loader is not None
            and args.eval_steps <= 0
            and (epoch + 1) % args.valid_every == 0
        ):
            best_valid_recon_loss = run_validation(
                model=model,
                recon_loss_fn=recon_loss_fn,
                valid_loader=valid_loader,
                device=device,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
                optional_eval_metrics=optional_eval_metrics,
                tracker=tracker,
                epoch=epoch,
                global_step=global_step,
                args=args,
                out_path=out_path,
                optimizer_g=optimizer_g,
                scheduler_g=scheduler_g,
                windows=windows,
                best_valid_recon_loss=best_valid_recon_loss,
                best_metrics_path=best_metrics_path,
                discriminators=discriminators,
                optimizer_d=optimizer_d,
                scheduler_d=scheduler_d,
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_path / f"checkpoint_epoch_{epoch + 1:04d}.safetensors",
                model,
                optimizer_g,
                scheduler_g,
                epoch,
                global_step,
                args,
                windows,
                discriminators=discriminators,
                optimizer_d=optimizer_d,
                scheduler_d=scheduler_d,
            )
            prune_numbered_checkpoints(out_path, args.keep_n_checkpoints)
            save_checkpoint(
                out_path / "last.safetensors",
                model,
                optimizer_g,
                scheduler_g,
                epoch,
                global_step,
                args,
                windows,
                discriminators=discriminators,
                optimizer_d=optimizer_d,
                scheduler_d=scheduler_d,
            )
            if tracker is not None:
                tracker.log(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "checkpoint/epoch": epoch + 1,
                        "checkpoint/global_step": global_step,
                    }
                )
    if tracker is not None:
        tracker.finish()


if __name__ == "__main__":
    run(main)
