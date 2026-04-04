"""
Training script for Schrödinger Bridge Matching (SBM) perturbation model.

Usage
-----
From the repo root (LightSB_exp/):

    python src/train_sbm.py \
        --config-name sbm_training \
        paths.base_data_path=/path/to/data \
        experiment_name=sbm_replogle

The script:
1. Loads the SBMDataModule (paired control / perturbed cells).
2. Loads a frozen VAE from checkpoint (same as LDM training).
3. Instantiates SchrodingerBridgeMatching and trains it.
"""
import os
import pathlib
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from scldm._utils import load_validate_statedict_config
from scldm.logger import logger

os.environ["HYDRA_FULL_ERROR"] = "1"

# Patch torch.load to be compatible with older checkpoints
_orig_load = torch.load


def _load_weights_only_false(*args: Any, **kwargs: Any) -> Any:
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _load_weights_only_false


def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    logger.info("Instantiating SBM DataModule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    datamodule.setup()

    # ------------------------------------------------------------------
    # Load VAE checkpoint
    # ------------------------------------------------------------------
    vae_state_dict = None
    is_vae_checkpoint = (
        hasattr(cfg.model.module, "vae_as_tokenizer")
        and "load_from_checkpoint" in cfg.model.module.vae_as_tokenizer
    )

    if is_vae_checkpoint:
        ckpt_cfg = cfg.model.module.vae_as_tokenizer.load_from_checkpoint
        job_path = pathlib.Path(f"{ckpt_cfg.ckpt_path}/{ckpt_cfg.job_name}")

        checkpoint_file = (
            f"epoch={ckpt_cfg.epoch}.ckpt"
            if ckpt_cfg.epoch is not None and isinstance(ckpt_cfg.epoch, int)
            else "last.ckpt"
        )
        logger.info(f"Loading VAE checkpoint: {job_path / checkpoint_file}")

        vae_checkpoints = torch.load(job_path / checkpoint_file, weights_only=False)
        vae_config = OmegaConf.load(job_path / "config.yaml")
        vae_state_dict, cfg = load_validate_statedict_config(vae_checkpoints, cfg, vae_config)
        logger.info("VAE config merged from checkpoint.")

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
    logger.info("Instantiating SchrodingerBridgeMatching model...")
    module = hydra.utils.instantiate(cfg.model.module)

    if is_vae_checkpoint and vae_state_dict is not None:
        module.vae_model.load_state_dict(vae_state_dict)
        logger.info("VAE weights loaded from checkpoint.")

        if not cfg.model.module.vae_as_tokenizer.train:
            for param in module.vae_model.parameters():
                param.requires_grad = False
            module.vae_model.eval()
            logger.info("VAE frozen.")

    # ------------------------------------------------------------------
    # Scale LR for multi-GPU
    # ------------------------------------------------------------------
    if world_size > 1 and hasattr(cfg.model.module, "sbm_optimizer"):
        orig_lr = cfg.model.module.sbm_optimizer.lr
        cfg.model.module.sbm_optimizer.lr = orig_lr * world_size
        logger.info(
            f"Scaled SBM LR: {orig_lr} -> {cfg.model.module.sbm_optimizer.lr}"
        )

    # ------------------------------------------------------------------
    # Callbacks & loggers
    # ------------------------------------------------------------------
    callbacks = []
    for cb_name, cb_cfg in cfg.training.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb_cfg))
        logger.info(f"Added callback: {cb_name}")

    loggers = []
    for lg_name, lg_cfg in cfg.training.logger.items():
        if lg_name == "wandb":
            if local_rank == 0:
                wandb_partial = hydra.utils.instantiate(lg_cfg)
                loggers.append(wandb_partial(id=None))
                logger.info(f"Added logger: {lg_name}")
        else:
            loggers.append(hydra.utils.instantiate(lg_cfg))
            logger.info(f"Added logger: {lg_name}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_partial = hydra.utils.instantiate(cfg.training.trainer)
    strategy = DDPStrategy(find_unused_parameters=True) if world_size > 1 else "auto"

    trainer: Trainer = trainer_partial(
        devices="auto",
        strategy=strategy,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        use_distributed_sampler=False,
    )

    # ------------------------------------------------------------------
    # Save config & resume
    # ------------------------------------------------------------------
    checkpoint_dir = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        OmegaConf.save(cfg, checkpoint_dir / "config.yaml")
        logger.info(f"Saved config to {checkpoint_dir / 'config.yaml'}")

    last_ckpt = checkpoint_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting SBM training...")
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("SBM training complete!")


@hydra.main(config_path="../configs", config_name="sbm_training", version_base="1.2")
def main(cfg: DictConfig) -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    train(cfg)


if __name__ == "__main__":
    main()
