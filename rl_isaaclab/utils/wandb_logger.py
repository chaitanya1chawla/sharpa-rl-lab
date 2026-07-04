from __future__ import annotations

from typing import Any

import os
import tempfile

import imageio.v2 as imageio
import numpy as np
import torch
import wandb


def _to_python(value: Any) -> Any:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item') and getattr(value, 'shape', ()) == ():
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist') and getattr(value, 'shape', ()) == ():
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.float().mean().item()
    return value


def init_wandb_run(*, project: str | None, entity: str | None, name: str | None, mode: str | None, dir: str, config: dict[str, Any] | None = None):
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        dir=dir,
        config=config or {},
        reinit=True,
    )
    return run


def log_wandb_metrics(run: Any, metrics: dict[str, Any], step: int | None = None):
    if run is None:
        return
    run.log({k: _to_python(v) for k, v in metrics.items()}, step=step)


def finish_wandb_run(run: Any):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


def log_wandb_gif(run: Any, key: str, frames: list[Any], step: int | None = None, fps: int = 20):
    if run is None or not frames:
        return
    tmp_dir = tempfile.mkdtemp(prefix="wandb_gif_")
    gif_path = os.path.join(tmp_dir, "rollout.gif")
    normalized_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        normalized_frames.append(frame)
    imageio.mimsave(gif_path, normalized_frames, fps=fps)
    run.log({key: wandb.Video(gif_path, format="gif", fps=fps)}, step=step)
