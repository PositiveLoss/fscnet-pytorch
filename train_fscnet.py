#!/usr/bin/env python3
"""Train an FSC-Net style speech bandwidth extension model.

Example:
  python train_fscnet.py \
    --train_manifest train.jsonl --valid_manifest valid.jsonl \
    --input_sr 4000 --target_sr 48000 --epochs 100 --batch_size 8
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence, cast

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fscnet_pytorch.audio import complex_to_ri, stft_complex
from fscnet_pytorch.config import (
    get_model_preset,
    model_preset_names,
    resolve_model_config,
)
from fscnet_pytorch.data import BandwidthExtensionDataset
from fscnet_pytorch.discriminator import (
    MultiScaleDiscriminator,
    discriminator_lsgan_loss,
    generator_lsgan_fm_loss,
    set_requires_grad,
)
from fscnet_pytorch.losses import StageLossWeights, StageReconstructionLoss
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train FSC-Net for speech bandwidth extension"
    )
    p.add_argument(
        "--train_manifest",
        default=None,
        help="jsonl/csv/txt manifest for training audio",
    )
    p.add_argument(
        "--valid_manifest", default=None, help="optional validation manifest"
    )
    p.add_argument("--out_dir", default="runs/fscnet", help="checkpoint/log directory")
    p.add_argument(
        "--model_size",
        choices=model_preset_names(),
        default="compact",
        help="model size preset; explicit architecture flags override it",
    )
    p.add_argument(
        "--list_model_sizes",
        action="store_true",
        help="print available model size presets and exit",
    )

    p.add_argument("--target_sr", type=int, default=None)
    p.add_argument("--input_sr", type=int, default=None)
    p.add_argument("--segment_seconds", type=float, default=2.0)
    p.add_argument("--n_fft", type=int, default=None)
    p.add_argument("--win_length", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=None)
    p.add_argument("--subbands", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--num_blocks", type=int, default=None)
    p.add_argument("--rnn_hidden", type=int, default=None)
    p.add_argument("--attention_heads", type=int, default=None)
    p.add_argument("--time_attention", choices=("v1", "v2"), default=None)
    p.add_argument(
        "--time_attention_qk_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable QK normalization for --time_attention v2",
    )
    p.add_argument(
        "--time_attention_rope",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable rotary time positions for --time_attention v2",
    )
    p.add_argument("--ffc_ratio", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument(
        "--progressive_windows",
        default=None,
        help="comma-separated windows; defaults to the selected model size preset",
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="defaults to the selected model size preset recommendation",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--clip_grad_norm", type=float, default=5.0)
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")

    p.add_argument("--trackio", action="store_true", help="enable Trackio logging")
    p.add_argument("--trackio_project", default="fscnet", help="Trackio project name")
    p.add_argument("--trackio_name", default=None, help="optional Trackio run name")
    p.add_argument("--trackio_group", default=None, help="optional Trackio run group")
    p.add_argument(
        "--trackio_space_id",
        default=None,
        help="optional Hugging Face Space id for hosted Trackio logs",
    )
    p.add_argument(
        "--trackio_server_url",
        default=None,
        help="optional self-hosted Trackio server URL",
    )
    p.add_argument(
        "--trackio_log_every",
        type=int,
        default=1,
        help="log training metrics every N optimizer steps; <=0 disables step logs",
    )

    p.add_argument("--mrstft_weight", type=float, default=1.0)
    p.add_argument("--lsd_weight", type=float, default=0.1)
    p.add_argument("--complex_l1_weight", type=float, default=1.0)
    p.add_argument("--mrstft_fft_sizes", default="512,1024,2048")

    p.add_argument(
        "--adv_weight", type=float, default=0.0, help="set >0 to enable per-stage LSGAN"
    )
    p.add_argument("--fm_weight", type=float, default=10.0)
    p.add_argument("--adv_start_step", type=int, default=0)
    p.add_argument("--disc_scales", type=int, default=3)
    p.add_argument("--disc_channels", type=int, default=16)

    p.add_argument("--save_every", type=int, default=1, help="save every N epochs")
    p.add_argument("--valid_every", type=int, default=1, help="validate every N epochs")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="CPU intra-op threads; 1 avoids oversubscription on many machines",
    )
    p.add_argument("--resume", default=None, help="checkpoint to resume")
    return p


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
    args: argparse.Namespace,
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


def save_checkpoint(
    path: Path,
    model: FSCNet,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    windows: Sequence[int],
    discriminators: nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scheduler_d=None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "config": model.cfg.to_dict(),
        "windows": tuple(windows),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "optimizer_g": optimizer_g.state_dict(),
        "scheduler_g": scheduler_g.state_dict() if scheduler_g is not None else None,
    }
    if discriminators is not None:
        payload["discriminators"] = discriminators.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    if scheduler_d is not None:
        payload["scheduler_d"] = scheduler_d.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def validate(
    model: FSCNet,
    loss_fn: StageReconstructionLoss,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="valid", leave=False):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
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
        total += float(loss.detach().cpu()) * lr.shape[0]
        count += lr.shape[0]
    model.train()
    return {"valid_recon_loss": total / max(1, count)}


def main() -> None:
    args = build_arg_parser().parse_args()
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {field: getattr(args, field) for field in MODEL_OVERRIDE_FIELDS}
    cfg, windows = resolve_model_config(
        args.model_size,
        overrides=overrides,
        progressive_windows=args.progressive_windows,
    )
    if args.batch_size is None:
        args.batch_size = get_model_preset(args.model_size).suggested_batch_size
    mrstft_fft_sizes = tuple(int(x) for x in args.mrstft_fft_sizes.split(",") if x)
    model = FSCNet(cfg).to(device)
    parameter_count = count_parameters(model)
    print(f"Model preset: {args.model_size}; windows={windows}; config={cfg.to_dict()}")
    print(f"Generator parameters: {parameter_count / 1e6:.3f} M")

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
        shuffle=True,
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    weights = StageLossWeights(
        args.mrstft_weight, args.lsd_weight, args.complex_l1_weight
    )
    recon_loss_fn = StageReconstructionLoss(
        cfg, windows, weights=weights, mrstft_fft_sizes=mrstft_fft_sizes
    ).to(device)

    discriminators = None
    optimizer_d = None
    scheduler_d = None
    if args.adv_weight > 0:
        discriminators = nn.ModuleList(
            [
                MultiScaleDiscriminator(
                    in_channels=2,
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
    total_steps = max(1, len(train_loader) * args.epochs)
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
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

    (out_dir / "config.json").write_text(
        json.dumps(
            {"model": cfg.to_dict(), "windows": windows, "args": vars(args)}, indent=2
        ),
        encoding="utf-8",
    )

    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    model.train()
    tracker = init_trackio(args, cfg.to_dict(), windows, parameter_count)

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", enabled=args.amp and device.type == "cuda"
            ):
                pred_stages, input_ri = model(lr, return_all=True)
                target_ri = complex_to_ri(
                    stft_complex(
                        hr, cfg.n_fft, cfg.hop_length, cfg.win_length, center=cfg.center
                    )
                )
                recon_loss, logs, pred_wavs, target_wavs = recon_loss_fn(
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
                    device_type="cuda", enabled=args.amp and device.type == "cuda"
                ):
                    d_loss = lr.new_tensor(0.0)
                    for disc, fake_wav, real_wav in zip(
                        discriminators_active, pred_wavs, target_wavs
                    ):
                        d_loss = d_loss + discriminator_lsgan_loss(
                            cast(MultiScaleDiscriminator, disc),
                            lr,
                            real_wav,
                            fake_wav.detach(),
                        )
                    d_loss = d_loss / len(pred_wavs)
                cast(torch.Tensor, scaler.scale(d_loss)).backward()
                scaler.unscale_(optimizer_d)
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        discriminators_active.parameters(), args.clip_grad_norm
                    )
                scaler.step(optimizer_d)
                if scheduler_d is not None:
                    scheduler_d.step()
                d_loss_val = float(d_loss.detach().cpu())

            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", enabled=args.amp and device.type == "cuda"
            ):
                loss_g = recon_loss
                adv_val = lr.new_tensor(0.0)
                fm_val = lr.new_tensor(0.0)
                if use_adv:
                    discriminators_active = cast(nn.ModuleList, discriminators)
                    set_requires_grad(discriminators_active, False)
                    for disc, fake_wav, real_wav in zip(
                        discriminators_active, pred_wavs, target_wavs
                    ):
                        adv, fm = generator_lsgan_fm_loss(
                            cast(MultiScaleDiscriminator, disc),
                            lr,
                            real_wav,
                            fake_wav,
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
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()
            if use_adv:
                discriminators_active = cast(nn.ModuleList, discriminators)
                set_requires_grad(discriminators_active, True)

            global_step += 1
            pbar.set_postfix(
                recon=f"{logs['recon_loss']:.4f}",
                g=f"{float(loss_g.detach().cpu()):.4f}",
                d=f"{d_loss_val:.4f}",
                lr=f"{scheduler_g.get_last_lr()[0]:.2e}",
            )
            if (
                tracker is not None
                and args.trackio_log_every > 0
                and global_step % args.trackio_log_every == 0
            ):
                train_metrics: dict[str, TrackMetricValue] = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train/recon_loss": logs["recon_loss"],
                    "train/g_loss": float(loss_g.detach().cpu()),
                    "train/d_loss": d_loss_val,
                    "train/adv_loss": float(adv_val.detach().cpu()),
                    "train/fm_loss": float(fm_val.detach().cpu()),
                    "train/lr_g": scheduler_g.get_last_lr()[0],
                    "train/use_adv": use_adv,
                }
                if scheduler_d is not None:
                    train_metrics["train/lr_d"] = scheduler_d.get_last_lr()[0]
                for key, value in logs.items():
                    if key != "recon_loss":
                        train_metrics[f"train/{key}"] = value
                tracker.log(train_metrics)

        if valid_loader is not None and (epoch + 1) % args.valid_every == 0:
            metrics = validate(model, recon_loss_fn, valid_loader, device, args.amp)
            print(metrics)
            if tracker is not None:
                tracker.log(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "valid/recon_loss": metrics["valid_recon_loss"],
                    }
                )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
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
            save_checkpoint(
                out_dir / "last.pt",
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
    main()
