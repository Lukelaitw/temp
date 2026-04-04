"""Standalone inference script for VAE models. Loads config and checkpoint, runs test or predict."""
# 必須在所有 import 之前
import torch
_orig_load = torch.load
def _load_weights_only_false(*args, **kwargs):
    kwargs["weights_only"] = False 
    return _orig_load(*args, **kwargs)
torch.load = _load_weights_only_false


import argparse
import pathlib

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from scldm.logger import logger
import typing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VAE inference (test or predict) from a trained checkpoint."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default="/projects/p32572/Luke/outputs/checkpoints/my_vae_experiment",
        help="Checkpoint directory containing config.yaml and last.ckpt",
    )
    group.add_argument(
        "--ckpt_path",
        type=pathlib.Path,
        default="/projects/p32572/Luke/outputs/checkpoints/my_vae_experiment/epoch449.ckpt",
        help="Path to a single .ckpt file (config.yaml must be in the same directory)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predict instead of test (outputs AnnData etc.)",
        default=False,
    )
    return parser.parse_args()


def _resolve_config_and_ckpt(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolve config path and checkpoint path from CLI args."""
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.resolve()
        config_path = checkpoint_dir / "config.yaml"
        ckpt_path = checkpoint_dir / "last.ckpt"
    else:
        ckpt_path = args.ckpt_path.resolve()
        checkpoint_dir = ckpt_path.parent
        config_path = checkpoint_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Ensure config.yaml exists in the checkpoint directory, or use --checkpoint_dir "
            "pointing to the training output directory."
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return config_path, ckpt_path


def run_inference(config_path: pathlib.Path, ckpt_path: pathlib.Path, predict: bool) -> None:
    """Load config and checkpoint, then run test or predict."""
    # Register eval resolver (used in some configs)
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    cfg = OmegaConf.load(config_path)
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # Setup datamodule (no setup_datamodule_and_steps - that mutates cfg for training)
    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    datamodule.setup()
    logger.info("Datamodule setup complete")

    # Instantiate model
    module = hydra.utils.instantiate(cfg.model.module)
    logger.info("Model instantiated")

    # Minimal Trainer for inference (no fit, no checkpointing, no logger)
    trainer = Trainer(
        devices="auto",
        accelerator="gpu",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    if predict:
        logger.info("Running predict...")
        trainer.predict(module, datamodule=datamodule, ckpt_path=str(ckpt_path))
        logger.info("Predict complete!")
    else:
        logger.info("Running test...")
        results = trainer.test(module, datamodule=datamodule, ckpt_path=str(ckpt_path))
        logger.info("Test complete!")
        if results:
            metrics = results[0]
            # Lightning may use "test_llh" or "test/test_llh" etc.
            def _get(suffix: str):
                for k, v in metrics.items():
                    if k.endswith(suffix) or suffix in k:
                        return v.item() if hasattr(v, "item") else v
                return None

            re_val = _get("llh") or _get("loss")
            pcc_val = _get("pcc")
            mse_val = _get("mse")
            logger.info("--- Test metrics ---")
            logger.info(f"  RE (reconstruction loss / -LLH): {re_val}")
            logger.info(f"  PCC (Pearson): {pcc_val}")
            logger.info(f"  MSE: {mse_val}")


def main() -> None:
    args = _parse_args()
    config_path, ckpt_path = _resolve_config_and_ckpt(args)
    run_inference(config_path, ckpt_path, predict=args.predict)


if __name__ == "__main__":
    main()
