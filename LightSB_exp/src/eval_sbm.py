"""Standalone GEARS-style evaluation for SBM checkpoints."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from scldm.evaluations import gears_aggregate, gears_per_condition_metrics
from scldm.logger import logger

DEFAULT_CHECKPOINT_DIR = pathlib.Path("/projects/p32572/Luke/outputs/checkpoints/sbm_replogle")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GEARS-style evaluation for a trained SBM checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=None,
        help="Checkpoint directory containing config.yaml and last.ckpt",
    )
    group.add_argument(
        "--ckpt_path",
        type=pathlib.Path,
        default=None,
        help="Path to a .ckpt file (config.yaml should be in same directory)",
    )
    parser.add_argument(
        "--ctrl_mean_path",
        type=pathlib.Path,
        required=True,
        help="Path to precomputed control mean .npy vector.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-K DE genes used in GEARS metrics.",
    )
    parser.add_argument(
        "--max_cells_per_condition",
        type=int,
        default=2048,
        help="Maximum number of cells to evaluate per condition.",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        default=pathlib.Path("gears_results.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def _resolve_config_and_ckpt(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.resolve()
        config_path = checkpoint_dir / "config.yaml"
        ckpt_path = checkpoint_dir / "last.ckpt"
    elif args.ckpt_path is not None:
        ckpt_path = args.ckpt_path.resolve()
        checkpoint_dir = ckpt_path.parent
        config_path = checkpoint_dir / "config.yaml"
    else:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR.resolve()
        config_path = checkpoint_dir / "config.yaml"
        ckpt_path = checkpoint_dir / "last.ckpt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return config_path, ckpt_path


def run_eval(
    config_path: pathlib.Path,
    ckpt_path: pathlib.Path,
    ctrl_mean_path: pathlib.Path,
    top_k: int,
    max_cells_per_condition: int,
    out_path: pathlib.Path,
) -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    cfg = OmegaConf.load(config_path)
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    ctrl_mean_np = np.load(ctrl_mean_path)
    if ctrl_mean_np.ndim != 1:
        raise ValueError(f"ctrl_mean must be 1D, got shape={ctrl_mean_np.shape}")
    ctrl_mean = torch.from_numpy(ctrl_mean_np.astype(np.float32))
    logger.info(f"Loaded ctrl_mean from {ctrl_mean_path}, shape={ctrl_mean_np.shape}")

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    datamodule.setup()
    module = hydra.utils.instantiate(cfg.model.module)

    trainer = Trainer(
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    logger.info("Running SBM predict for GEARS evaluation...")
    predictions = trainer.predict(module, datamodule=datamodule, ckpt_path=str(ckpt_path))
    logger.info(f"Collected {len(predictions)} prediction batches.")

    accum: dict[int, dict[str, Any]] = defaultdict(lambda: {"pred": [], "true": [], "n": 0})
    for batch_out in predictions:
        pred = batch_out["predicted_counts"]
        true = batch_out["tgt_counts"]
        cond = batch_out["condition_gene"].long().squeeze(-1)
        for c in torch.unique(cond):
            c_int = int(c.item())
            idx = (cond == c).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            state = accum[c_int]
            remain = int(max_cells_per_condition) - int(state["n"])
            if remain <= 0:
                continue
            kept = idx[:remain]
            state["pred"].append(pred[kept].cpu())
            state["true"].append(true[kept].cpu())
            state["n"] += int(kept.numel())

    per_condition: dict[str, dict[str, float]] = {}
    metrics_list: list[dict[str, float]] = []
    for c_int in sorted(accum.keys()):
        state = accum[c_int]
        if not state["pred"]:
            continue
        pred_c = torch.cat(state["pred"], dim=0)
        true_c = torch.cat(state["true"], dim=0)
        m = gears_per_condition_metrics(
            pred_counts=pred_c,
            true_counts=true_c,
            ctrl_mean=ctrl_mean,
            top_k=top_k,
        )
        per_condition[str(c_int)] = m
        metrics_list.append(m)

    aggregate = gears_aggregate(metrics_list)
    result = {
        "checkpoint": str(ckpt_path),
        "ctrl_mean_path": str(ctrl_mean_path),
        "top_k": int(top_k),
        "max_cells_per_condition": int(max_cells_per_condition),
        "aggregate": aggregate,
        "per_condition": per_condition,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved GEARS evaluation results to {out_path}")


def main() -> None:
    args = _parse_args()
    config_path, ckpt_path = _resolve_config_and_ckpt(args)
    run_eval(
        config_path=config_path,
        ckpt_path=ckpt_path,
        ctrl_mean_path=args.ctrl_mean_path,
        top_k=args.top_k,
        max_cells_per_condition=args.max_cells_per_condition,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
