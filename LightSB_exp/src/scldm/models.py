from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from ema_pytorch import EMA
from hydra.core.config_store import DictConfig
from pytorch_lightning import LightningModule
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution, Normal
from torch.utils._pytree import tree_map
from torchmetrics.functional.regression import mean_squared_error, pearson_corrcoef, r2_score

from scldm.constants import LossEnum, ModelEnum
from scldm.evaluations import (
    BrayCurtisKernel,
    MMDLoss,
    RBFKernel,
    RuzickaKernel,
    TanimotoKernel,
    gears_aggregate,
    gears_per_condition_metrics,
    wasserstein,
)
from scldm.lightsbm import ConditionedLightSBM
from scldm.logger import logger
from scldm.nnets import DiT
from scldm.transport import Sampler, Transport
from scldm.vae import TransformerVAE


def _log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: Callable[[torch.Tensor], torch.Tensor] = torch.log,
    lgamma_fn: Callable[[torch.Tensor], torch.Tensor] = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood of a minibatch under a negative binomial (positive support)."""
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    return (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )


REGRESSION_METRICS = {
    "mse": mean_squared_error,
    "pcc": pearson_corrcoef,
    # "scc": spearman_corrcoef,
    # "r2": partial(r2_score, multioutput="raw_values"),
}

MMD_METRICS = {
  #  "mmd_braycurtis_counts": MMDLoss(kernel=BrayCurtisKernel()),
  #  "mmd_tanimoto": MMDLoss(kernel=TanimotoKernel()),
  #  "mmd_ruzicka_counts": MMDLoss(kernel=RuzickaKernel()),
    "mmd_rbf": MMDLoss(kernel=RBFKernel()),
}

WASSERSTEIN_METRICS = {
  #  "wasserstein1_sinkhorn": partial(wasserstein, method="sinkhorn", power=1),
    "wasserstein2_sinkhorn": partial(wasserstein, method="sinkhorn", power=2),
}


R2_METRICS = {
    "r2_mean": lambda preds, target: r2_score(preds.mean(0), target.mean(0)),
    "r2_var": lambda preds, target: r2_score(preds.var(0), target.var(0)),
}


class BaseModel(LightningModule, ABC):
    """Abstract base class for VAE-based models."""

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        """Sample from the model."""
        pass

    @abstractmethod
    def inference(self, *args, **kwargs) -> dict[str, Any]:
        """Inference from the model."""
        pass

    def validation_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        metrics = self.shared_step(batch, batch_idx, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        metrics = self.shared_step(batch, batch_idx, "val", ema=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        metrics = self.shared_step(batch, batch_idx, "test")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        metrics = self.shared_step(batch, batch_idx, "test", ema=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Now the parameters have been updated by the optimizer
        # This is the right place to update the EMA
        if hasattr(self, "ema_model"):
            self.ema_model.update()

    def on_train_epoch_start(self) -> None:
        # from cellarium-ml
        combined_loader = self.trainer.fit_loop._combined_loader
        assert combined_loader is not None
        dataloaders = combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)

    def on_validation_start(self):
        pass
        """Reset datasets before validation to ensure consistent state"""
        # Add logging to debug
       # if dist.is_initialized():
       #     rank = dist.get_rank()
       #     world_size = dist.get_world_size()
       #     logger.info(f"Rank {rank}/{world_size} - Validation starting")
            # train_dataloader = self.trainer.train_dataloader()
      #      val_dataloader = (
      #          self.trainer.val_dataloaders[0]
      #          if isinstance(self.trainer.val_dataloaders, list)
      #          else self.trainer.val_dataloaders
      #      )
       #     if val_dataloader is not None:
      #          logger.info(f"Rank {rank}/{world_size} - Val dataset size: {len(val_dataloader.dataset)}")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        fit_loop = self.trainer.fit_loop
        epoch_loop = fit_loop.epoch_loop
        batch_progress = epoch_loop.batch_progress
        if batch_progress.current.completed < batch_progress.current.processed:  # type: ignore[attr-defined]
            # Checkpointing is done before these attributes are updated. So, we need to update them manually.
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] += 1
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"] += 1
            if not epoch_loop._should_accumulate():
                checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"] += 1
            if batch_progress.is_last_batch:
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]["processed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["processed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]["completed"] += 1
                checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] += 1

    def _compute_gradient_norms(self, modules: dict[str, nn.Module]) -> dict[str, float]:
        """Compute gradient norms for each module."""
        grad_norms = {}

        # Compute norms for each module and their submodules
        for name, module in modules.items():
            if module is None or not any(p.requires_grad for p in module.parameters()):
                continue

            # Total norm for the module
            grad_norms[f"grad_norm/{name}"] = self._calculate_grad_norm(module.parameters())

            # Compute norms for each submodule
            for submodule_name, submodule in module.named_children():
                # Compute norms for each sub-submodule
                for sub_submodule_name, sub_submodule in submodule.named_children():
                    if not any(p.requires_grad for p in sub_submodule.parameters()):
                        continue

                    # Include class name in the logging key for better identification
                    class_name = sub_submodule.__class__.__name__
                    grad_norms[f"grad_norm/{name}/{submodule_name}/{sub_submodule_name}_{class_name}"] = (
                        self._calculate_grad_norm(sub_submodule.parameters())
                    )

        return grad_norms

    @staticmethod
    def _calculate_grad_norm(parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm**2
        return total_norm**0.5


class VAE(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        vae_scheduler: Callable[[int], float] | None = None,
        calculate_grad_norms: bool = False,
        ortho_loss_weight: float = 0.0,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model
        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS

        self.calculate_grad_norms = calculate_grad_norms
        self.ortho_loss_weight = ortho_loss_weight

        self.generation_args = generation_args
        self.inference_args = inference_args

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def configure_optimizers(self):
        vae_params = [p for p in self.vae_model.parameters() if p.requires_grad]
        vae_config = (
            {"optimizer": self.vae_optimizer(vae_params)} if vae_params else {}
        )  # empty dict is the case when vae is frozen

        if self.vae_scheduler is not None and vae_config:
            vae_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(vae_config["optimizer"], self.vae_scheduler),
                "interval": "step",
            }

        return vae_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hasattr(self, "vae_model_compiled"):
            return self.vae_model_compiled(counts, genes, library_size, counts_subset, genes_subset)
        else:
            return self.vae_model(counts, genes, library_size, counts_subset, genes_subset)

    def loss(
        self,
        counts: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ) -> dict[str, Any]:
        recon_loss = -_log_nb_positive(counts, mu, theta)
        output = {
            LossEnum.LLH_LOSS.value: recon_loss.sum(dim=1).mean(),
        }
        if self.ortho_loss_weight > 0:
            inducing_points = getattr(
                getattr(self.vae_model.encoder, "ca_layer", None),
                "inducing_points",
                None,
            )
            if inducing_points is not None:
                Q = inducing_points  # (M, D)
                Q_norm = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-8)
                M = Q_norm.shape[0]
                I_M = torch.eye(M, device=Q.device, dtype=Q.dtype)
                ortho_loss = (Q_norm @ Q_norm.T - I_M).pow(2).sum()
                output[LossEnum.ORTHO_LOSS.value] = self.ortho_loss_weight * ortho_loss
        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]

        mu, theta, _ = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            mu=mu,
            theta=theta,
        )

        self.log("train_theta", theta.mean(), on_step=True, on_epoch=True, sync_dist=True)

        loss = sum(loss_output.values())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_output.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            modules = {
                "encoder": self.vae_model.encoder,
                "decoder": self.vae_model.decoder,
                "encoder_head": self.vae_model.encoder_head,
                "decoder_head": self.vae_model.decoder_head,
            }
            if hasattr(self.vae_model, "input_layer"):
                modules["input_layer"] = self.vae_model.input_layer
            grad_norms = self._compute_gradient_norms(modules=modules)
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]

        mu, theta, _ = self.vae_model(
            counts,
            genes,
            library_size,
            counts_subset,
            genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            mu=mu,
            theta=theta,
        )

        loss = sum(loss_output.values())
        metrics = {}
        metrics[f"{stage}_loss"] = loss
        for k, v in loss_output.items():
            metrics[f"{stage}_{k}"] = v

        counts_pred = NegativeBinomialSCVI(mu=mu, theta=theta).sample()

        counts_pred_scaled = torch.log1p((counts_pred / counts_pred.sum(dim=1, keepdim=True)) * 10_000)
        counts_true_scaled = torch.log1p((counts / counts.sum(dim=1, keepdim=True)) * 10_000)

        counts_pred_zeros = (counts_pred == 0).float()
        counts_true_zeros = (counts == 0).float()

        metrics[f"{stage}_zeros_accuracy"] = (counts_pred_zeros == counts_true_zeros).float().mean()

        for k, fn in self.metric_fns.items():
            output = fn(counts_pred_scaled, counts_true_scaled)
            metrics[f"{stage}_{k}"] = torch.nanmean(output)

        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        outputs = self.inference(batch)
        return outputs

    @torch.no_grad()
    def sample(
        self,
        library_size: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Sampling is not implemented for VAE")

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        from scldm._utils import create_anndata_from_inference_output

        # encode_kwargs = {str(k): v for k, v in self.inference_args.items()} if self.inference_args else {}

        mu, theta, z = self.forward(
            counts=batch[ModelEnum.COUNTS.value],
            genes=batch[ModelEnum.GENES.value],
            library_size=batch[ModelEnum.LIBRARY_SIZE.value],
            counts_subset=batch.get(ModelEnum.COUNTS_SUBSET.value),
            genes_subset=batch.get(ModelEnum.GENES_SUBSET.value),
        )
        counts_pred = NegativeBinomialSCVI(mu=mu, theta=theta).sample()
        inference_outputs: dict[str, torch.Tensor] = {
            "reconstructed_counts": counts_pred.cpu(),
            "z": z.cpu(),
        }
        inference_outputs.update({k: batch[k].cpu().numpy() for k in tree_map(lambda x: x.cpu(), batch)})
        adata = create_anndata_from_inference_output(inference_outputs, self.trainer.datamodule)
        return adata


class LatentDiffusion(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        # diffusion
        diffusion_model: DiT,
        transport: Transport,
        diffusion_scheduler: Callable[[int], float],
        diffusion_optimizer: Callable[[], Any],
        # more vae
        vae_scheduler: Callable[[int], float] | None = None,
        # ema
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
        update_after_step: int = 1000,
        allow_different_devices: bool = True,
        use_foreach: bool = True,
        calculate_grad_norms: bool = False,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        vae_as_tokenizer: DictConfig | None = None,
        # generation evaluation
        eval_generation: DictConfig = DictConfig({"enabled": False}),
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS
        self.mmd_metric_fns = MMD_METRICS
        self.wasserstein_metric_fns = WASSERSTEIN_METRICS
        self.r2_metric_fns = R2_METRICS

        self.generation_args = generation_args
        self.inference_args = inference_args

        self.diffusion_scheduler = diffusion_scheduler
        self.diffusion_optimizer = diffusion_optimizer

        self.vae_as_tokenizer = vae_as_tokenizer
        if self.vae_as_tokenizer is not None and not getattr(self.vae_as_tokenizer, "train", False):
            logger.info("VAE model is frozen")
            self.freeze()
            self.vae_model.eval()

        self.diffusion_model = diffusion_model

        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.transport = transport
        self.transport_sampler = Sampler(self.transport)
        self.mse_loss = nn.MSELoss()

        self.ema_model = EMA(
            model=self.diffusion_model,
            beta=ema_decay,  # exponential moving average factor
            update_every=ema_update_every,  # how often to update
            allow_different_devices=allow_different_devices,
            use_foreach=use_foreach,
            update_after_step=update_after_step,
        )
        self.mmd_metric_fns = MMD_METRICS
        self.calculate_grad_norms = calculate_grad_norms

        # Initialize attributes for generation evaluation
        self.eval_generation: DictConfig = eval_generation
        self.accumulated_generated_batches: list[torch.Tensor] = []
        self.accumulated_samples = 0  # number of samples accumulated for generation evaluation
        self.is_generation_eval_epoch = False  # used in on_validation*** to decide if it is a generation eval epoch

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )
            self.diffusion_model_compiled = torch.compile(
                self.diffusion_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def _sample_log_size_factors(self, condition: dict[str, torch.Tensor] | None, batch_size: int) -> torch.Tensor:
        """Sample log size factors using joint or independent condition stats.

        Behavior:
        - If `self.diffusion_model.condition_strategy == "joint"` and the vocabulary
          encoder provides `joint_idx_2_classes` and a valid `joint_key` present in
          both `mu_size_factor` and `sd_size_factor`, build a joint class key per
          sample and draw from the corresponding Normal distribution.
        - Otherwise, fall back to independent sampling based on a single condition key.
          The key is resolved by `vocab.size_factor_condition_key` if available,
          otherwise inferred from the intersection of `condition.keys()` with
          `mu_size_factor`/`sd_size_factor` keys.

        If statistics are missing, return zeros for the affected samples and log a
        warning once, to keep generation running.
        """
        vocab_encoder = self.trainer.datamodule.vocabulary_encoder
        mu_size_factor = getattr(vocab_encoder, "mu_size_factor", None)
        sd_size_factor = getattr(vocab_encoder, "sd_size_factor", None)

        log_size_factors = torch.zeros(batch_size, device=self.device)

        # Early exit when stats or condition are unavailable
        if condition is None or mu_size_factor is None or sd_size_factor is None:
            return log_size_factors

        # Decide whether to use joint sampling
        use_joint = False
        joint_idx_2_classes = getattr(vocab_encoder, "joint_idx_2_classes", None)
        joint_key = getattr(vocab_encoder, "joint_key", None)
        if getattr(self.diffusion_model, "condition_strategy", None) == "joint" and joint_idx_2_classes is not None:
            # Safe casts after early None-check above
            mu_map = cast(dict, mu_size_factor)
            sd_map = cast(dict, sd_size_factor)
            if joint_key is not None and joint_key in mu_map and joint_key in sd_map:
                use_joint = True

        if use_joint:
            components = getattr(vocab_encoder, "joint_components", None)
            if components is not None:
                component_keys = [k for k in components if k in condition]
            else:
                component_keys = list(condition.keys())

            # Validate lengths for all component keys
            for k in component_keys:
                if len(condition[k]) != batch_size:
                    if not hasattr(self, "_warned_joint_len_mismatch"):
                        logger.warning(
                            "Length mismatch for joint components; expected %d, got different sizes. Using zeros.",
                            batch_size,
                        )
                        self._warned_joint_len_mismatch = True
                    return log_size_factors

            mu_map = cast(dict, mu_size_factor)
            sd_map = cast(dict, sd_size_factor)
            for bch_idx in range(batch_size):
                indices = [int(condition[k][bch_idx].item()) for k in component_keys]
                key = "_".join(str(i) for i in indices)
                if key not in joint_idx_2_classes:
                    if not hasattr(self, "_warned_missing_joint_key"):
                        logger.warning(
                            "Joint key '%s' not found in vocabulary_encoder.joint_idx_2_classes; using zero.", key
                        )
                        self._warned_missing_joint_key = True
                    continue
                class_idx = joint_idx_2_classes[key]
                mu_vec = cast(dict, mu_map[joint_key])  # type: ignore[index]
                sd_vec = cast(dict, sd_map[joint_key])  # type: ignore[index]
                mean_val = mu_vec.get(class_idx)
                std_val = sd_vec.get(class_idx)
                if mean_val is None or std_val is None:
                    if not hasattr(self, "_warned_missing_stats"):
                        logger.warning("Missing mean/std for joint size factor; using zero for affected samples.")
                        self._warned_missing_stats = True
                    continue
                log_size_factors[bch_idx] = Normal(loc=mean_val, scale=std_val).sample()
            return log_size_factors

        # Independent sampling path
        size_factor_condition_key = getattr(vocab_encoder, "size_factor_condition_key", None)
        selected_key: str | None = None
        if (
            size_factor_condition_key
            and size_factor_condition_key in condition
            and size_factor_condition_key in mu_size_factor
            and size_factor_condition_key in sd_size_factor
        ):
            selected_key = size_factor_condition_key
        else:
            cond_keys = set(condition.keys())
            mu_keys = set(mu_size_factor.keys())
            sd_keys = set(sd_size_factor.keys())
            inter = sorted(cond_keys & mu_keys & sd_keys)
            if inter:
                selected_key = inter[0]
                if not hasattr(self, "_warned_inferred_condition_key"):
                    logger.warning("Inferred size-factor condition key '%s' for independent sampling.", selected_key)
                    self._warned_inferred_condition_key = True
            else:
                if not hasattr(self, "_warned_no_condition_key"):
                    logger.warning("No matching condition key found in mu/sd for size-factor sampling; using zeros.")
                    self._warned_no_condition_key = True
                return log_size_factors

        assert selected_key is not None
        labels = condition[selected_key]
        if len(labels) != batch_size:
            raise ValueError(f"Condition '{selected_key}' length ({len(labels)}) must match batch size ({batch_size})")
        mu_map = cast(dict, mu_size_factor)
        sd_map = cast(dict, sd_size_factor)
        for i in range(batch_size):
            class_idx = int(labels[i].item())
            mu_vec = cast(dict, mu_map[selected_key])
            sd_vec = cast(dict, sd_map[selected_key])
            mean_val = mu_vec.get(class_idx)
            std_val = sd_vec.get(class_idx)
            if mean_val is None or std_val is None:
                if not hasattr(self, "_warned_missing_stats"):
                    logger.warning("Missing mean/std for independent size factor; using zero for affected samples.")
                    self._warned_missing_stats = True
                continue
            log_size_factors[i] = Normal(loc=mean_val, scale=std_val).sample()
        return log_size_factors

    def configure_optimizers(self):
        diffusion_params = [p for p in self.diffusion_model.parameters() if p.requires_grad]
        diffusion_config = {"optimizer": self.diffusion_optimizer(diffusion_params)}

        if self.diffusion_scheduler is not None:
            diffusion_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(diffusion_config["optimizer"], self.diffusion_scheduler),
                "interval": "step",
            }

        return diffusion_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if self.model_is_compiled:
            z = self.vae_model_compiled.encode(
                counts,
                genes,
                counts_subset,
                genes_subset,
            )
        else:
            z = self.vae_model.encode(
                counts,
                genes,
                counts_subset,
                genes_subset,
            )
        return z

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (ModelEnum.COUNTS.value, ModelEnum.GENES.value, ModelEnum.LIBRARY_SIZE.value)

        z = self.forward(
            counts=counts,
            genes=genes,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )

        # Prepare all available conditions for CFG dropout (similar to SiT approach)
        condition_keys = [k for k in batch.keys() if k not in exclude_keys]
        condition = {k: batch[k] for k in condition_keys}
        model_kwargs = {"condition": condition}

        if self.model_is_compiled:
            loss_dict = self.transport.training_losses(self.diffusion_model_compiled, z, model_kwargs)
        else:
            loss_dict = self.transport.training_losses(self.diffusion_model, z, model_kwargs)

        loss_output = {"train_loss": loss_dict["loss"].mean()}

        self.log("train_loss", loss_output["train_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            grad_norms = self._compute_gradient_norms({"diffusion": self.diffusion_model})
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)

        return loss_output["train_loss"]

    def freeze(self):
        """Freeze the vae model parameters"""
        logger.info("Freezing the vae model parameters")
        for param in self.vae_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]

        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        model = self.ema_model if ema else self.diffusion_model
        stage = stage + "_ema" if ema else stage

        z = self.forward(
            counts=counts,
            genes=genes,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        loss_dict = self.transport.training_losses(model, z, {"condition": condition})

        metrics = {}
        metrics[f"{stage}_loss"] = loss_dict["loss"].mean()
        metrics[f"{stage}_{LossEnum.DIFF_LOSS.value}"] = loss_dict["loss"].mean()
        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor] | None:
        if self.generation_args is not None:
            generation_kwargs = {str(k): v for k, v in self.generation_args.items()} if self.generation_args else {}
            guidance_weight = generation_kwargs.get("guidance_weight", None)
            timesteps = generation_kwargs.get("timesteps", 50)

            exclude_keys = (
                ModelEnum.COUNTS.value,
                ModelEnum.GENES.value,
                ModelEnum.COUNTS_SUBSET.value,
                ModelEnum.GENES_SUBSET.value,
                ModelEnum.LIBRARY_SIZE.value,
            )

            condition = {k: v for k, v in batch.items() if k not in exclude_keys}
            size_factors = batch[ModelEnum.LIBRARY_SIZE.value]
            genes = batch[ModelEnum.GENES.value]

            nb_outputs, z_outputs = self.sample(
                condition=condition,
                guidance_weight=guidance_weight,
                batch_size=len(size_factors),
                genes=genes,
                timesteps=timesteps,
            )
            batch_size_single = len(size_factors)
            # first half is unconditional, second half is conditional
            batch[f"{ModelEnum.COUNTS.value}_generated_unconditional"] = nb_outputs[:batch_size_single]
            batch[f"{ModelEnum.COUNTS.value}_generated_conditional"] = nb_outputs[batch_size_single:]
            batch["z_generated_unconditional"] = z_outputs[:batch_size_single].flatten(start_dim=1)
            batch["z_generated_conditional"] = z_outputs[batch_size_single:].flatten(start_dim=1)
            return tree_map(lambda x: x.cpu(), batch)
        elif self.inference_args is not None:
            from scldm._utils import create_anndata_from_inference_output

            logger.info("Running inference")
            encode_kwargs = {str(k): v for k, v in self.inference_args.items()} if self.inference_args else {}

            inference_outputs: dict[str, torch.Tensor] = self.inference(
                batch=batch,
                **encode_kwargs,
            )
            excluded_keys = (
                ModelEnum.COUNTS.value,
                ModelEnum.GENES.value,
                ModelEnum.LIBRARY_SIZE.value,
                ModelEnum.COUNTS_SUBSET.value,
                ModelEnum.GENES_SUBSET.value,
            )
            inference_outputs.update({k: batch[k].cpu().numpy() for k in batch if k not in excluded_keys})
            adata = create_anndata_from_inference_output(inference_outputs, self.trainer.datamodule)
            return adata
        else:
            raise ValueError("No generation or encode args provided")

    @torch.no_grad()
    def sample(
        self,
        condition: dict[str, torch.Tensor] | None,
        guidance_weight: dict[str, float] | None,
        batch_size: int,
        genes: torch.Tensor,
        timesteps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Validate inputs
        if len(genes) != batch_size:
            raise ValueError(f"genes batch dimension ({genes.shape[0]}) must match batch_size ({batch_size})")

        if condition is not None:
            for key, values in condition.items():
                if len(values) != batch_size:
                    raise ValueError(f"Condition '{key}' length ({len(values)}) must match batch size ({batch_size})")

        # Sample size factors from normal distributions based on condition labels
        size_factors = self._sample_log_size_factors(condition, batch_size)

        # Initial latent noise
        z = torch.randn(
            (batch_size, self.diffusion_model.seq_len, self.vae_model.encoder.latent_embedding),
            device=self.device,
        )

        sample_fn = self.transport_sampler.sample_ode()

        # Validate guidance_weight keys match condition keys (only if both are not None)
        if guidance_weight is not None and condition is not None:
            assert set(guidance_weight.keys()) == set(condition.keys()), (
                f"Guidance weight keys {set(guidance_weight.keys())} must match condition keys {set(condition.keys())}"
            )

        z_cfg = torch.cat([z, z], dim=0)

        # Duplicate conditions for CFG
        condition_cfg = {}
        if condition is not None:
            for key, values in condition.items():
                condition_cfg[key] = torch.cat([values, values], dim=0)

        model_fn = lambda x, t, **kwargs: self.diffusion_model.forward_with_cfg(
            x, t, **kwargs, cfg_scale=guidance_weight
        )
        samples = sample_fn(z_cfg, model_fn, **{"condition": condition_cfg})[-1]

        genes = torch.cat([genes, genes], dim=0)

        size_factors_actual = torch.exp(size_factors).view(-1, 1)  # shape: (batch_size, 1)
        size_factors_cfg = torch.cat([size_factors_actual, size_factors_actual], dim=0)
        nb = self.vae_model.decode(samples, genes, size_factors_cfg)
        return nb.sample(), samples

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value)

        mu, theta, z = self.vae_model.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        counts_pred = NegativeBinomialSCVI(mu=mu, theta=theta).sample()
        output: dict[str, torch.Tensor] = {
            "reconstructed_counts": counts_pred.cpu(),
            "z": z.cpu(),
            "z_mean_flat": z.flatten(start_dim=1).cpu(),
        }
        return output

    def on_validation_epoch_start(self) -> None:
        """Check if this is a generation evaluation epoch and initialize accumulation."""
        super().on_validation_start()

        if (
            self.eval_generation.enabled
            and self.current_epoch % self.eval_generation.freq == 0
            and self.current_epoch > self.eval_generation.warmup_epochs
            and self.current_epoch > 0
        ):
            self.is_generation_eval_epoch = True
            self.accumulated_generated_batches = []
            if dist.is_initialized():
                rank = dist.get_rank()
                logger.info(f"Rank {rank} - Starting generation evaluation at epoch {self.current_epoch}")
        else:
            self.is_generation_eval_epoch = False

    def validation_step(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch_idx: int) -> None:
        """Override validation_step to accumulate batches during generation evaluation epochs."""
        super().validation_step(batch, batch_idx)

        if self.is_generation_eval_epoch:
            if self.accumulated_samples < self.eval_generation.sample_size:
                # Cast batch to the expected type for sample method
                timesteps = self.generation_args.get("timesteps", 50)
                genes = batch[ModelEnum.GENES.value]
                logger.info("Generating samples.")
                counts_generated, _ = self.sample(
                    condition=None,
                    guidance_weight=None,
                    batch_size=len(batch[ModelEnum.COUNTS.value]),
                    genes=genes,
                    timesteps=timesteps,
                )
                bs = len(batch[ModelEnum.COUNTS.value])
                batch[f"{ModelEnum.COUNTS.value}_generated_unconditional"] = counts_generated[:bs]
                batch[f"{ModelEnum.COUNTS.value}_generated_conditional"] = counts_generated[bs:]
                self.accumulated_generated_batches.append(tree_map(lambda x: x.cpu(), batch))
                self.accumulated_samples += len(batch[ModelEnum.COUNTS.value])

    def on_validation_epoch_end(self) -> None:
        """Process accumulated batches for generation evaluation."""
        if self.is_generation_eval_epoch and len(self.accumulated_generated_batches) > 0:
            # Concatenate all accumulated batches
            counts = torch.cat([b[ModelEnum.COUNTS.value] for b in self.accumulated_generated_batches], dim=0)
            counts_gen_u = torch.cat(
                [b[f"{ModelEnum.COUNTS.value}_generated_unconditional"] for b in self.accumulated_generated_batches],
                dim=0,
            )
            counts_gen_c = torch.cat(
                [b[f"{ModelEnum.COUNTS.value}_generated_conditional"] for b in self.accumulated_generated_batches],
                dim=0,
            )
            library_size = torch.cat(
                [b[ModelEnum.LIBRARY_SIZE.value] for b in self.accumulated_generated_batches], dim=0
            )
            counts_true_scaled = torch.log1p((counts / library_size) * 10_000)
            logger.info("Computing generation evaluation metrics.")
            for branch, counts_generated in (
                ("unconditional", counts_gen_u),
                ("conditional", counts_gen_c),
            ):
                counts_generated_scaled = torch.log1p((counts_generated / library_size) * 10_000)
                for k, fn in self.mmd_metric_fns.items():
                    if "counts" in k:
                        mmd = fn(counts_true_scaled, counts_generated_scaled)
                    else:
                        mmd = fn(counts, counts_generated)
                    self.log(
                        f"generation_eval/{k}_{branch}",
                        torch.nanmean(mmd),
                        on_epoch=True,
                        sync_dist=True,
                    )
                for k, fn in self.wasserstein_metric_fns.items():
                    wdist = fn(counts_true_scaled, counts_generated_scaled)
                    self.log(
                        f"generation_eval/{k}_{branch}",
                        wdist,
                        on_epoch=True,
                        sync_dist=True,
                    )
            #    for k, fn in self.r2_metric_fns.items():
            #        r2 = fn(counts_true_scaled, counts_generated_scaled)
            #        self.log(
            #            f"generation_eval/{k}_{branch}",
            #            r2,
            #            on_epoch=True,
            #            sync_dist=True,
            #        )

            #self.log("generation_eval/total_samples", counts.shape[0], on_epoch=True, sync_dist=True)

            if dist.is_initialized():
                rank = dist.get_rank()
                logger.info(f"Rank {rank} - Generation evaluation completed with {counts.shape[0]} total samples")

            # Clear accumulated batches to free memory
            self.accumulated_generated_batches = []
            self.accumulated_samples = 0
            self.is_generation_eval_epoch = False


class VAEScvi(BaseModel):
    def __init__(
        self,
        # vae
        vae_model: TransformerVAE,
        vae_optimizer: Callable[[], Any],
        vae_scheduler: Callable[[int], float] | None = None,
        # loss
        kl_weight: float = 1.0,
        cr_weight: float = 0.0,
        masking_prop: float = 0.0,
        mask_token_idx: int = 0,
        # ema
        ema_decay: float = 0.999,
        ema_update_every: int = 1,
        update_after_step: int = 1000,
        allow_different_devices: bool = True,
        use_foreach: bool = True,
        calculate_grad_norms: bool = False,
        # generation
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        compile: bool = False,
        compile_mode: str = "default",
    ):
        super().__init__()

        self.vae_model = vae_model

        self.model_is_compiled = compile
        self.compile_mode = compile_mode

        self.vae_scheduler = vae_scheduler
        self.vae_optimizer = vae_optimizer

        self.metric_fns = REGRESSION_METRICS

        self.kl_weight = kl_weight
        self.masking_prop = masking_prop
        self.cr_weight = cr_weight
        self.mask_token_idx = mask_token_idx
        if self.masking_prop > 0:
            assert cr_weight is not None, "cr_weight must be provided if masking_prop is greater than 0"
        self.calculate_grad_norms = calculate_grad_norms

        self.generation_args = generation_args
        self.inference_args = inference_args

    def on_fit_start(self) -> None:
        if self.model_is_compiled:
            logger.info(f"Compiling model with {self.compile_mode} mode.")
            self.vae_model_compiled = torch.compile(
                self.vae_model, mode=self.compile_mode, dynamic=True, fullgraph=False
            )

    def configure_optimizers(self):
        vae_params = [p for p in self.vae_model.parameters() if p.requires_grad]
        vae_config = (
            {"optimizer": self.vae_optimizer(vae_params)} if vae_params else {}
        )  # empty dict is the case when vae is frozen

        if self.vae_scheduler is not None and vae_config:
            vae_config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(vae_config["optimizer"], self.vae_scheduler),
                "interval": "step",
            }

        return vae_config

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
        masking_prop: float = 0.0,
        mask_token_idx: int = 0,
    ) -> tuple[Distribution, Distribution, torch.Tensor]:
        if self.model_is_compiled:
            return self.vae_model_compiled(
                counts, genes, library_size, condition, counts_subset, genes_subset, masking_prop, mask_token_idx
            )
        else:
            return self.vae_model(
                counts, genes, library_size, condition, counts_subset, genes_subset, masking_prop, mask_token_idx
            )

    def loss(
        self,
        counts: torch.Tensor,
        conditional_likelihood: Distribution,
        variational_posterior: Distribution,
        z_sample: torch.Tensor,
        conditional_likelihood_masked: Distribution | None = None,
        variational_posterior_masked: Distribution | None = None,
        z_sample_masked: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        recon_loss = -conditional_likelihood.log_prob(counts)
        kl_loss = self.kl_weight * (variational_posterior.log_prob(z_sample) - self.vae_model.prior.log_prob(z_sample))

        output = {
            LossEnum.LLH_LOSS.value: recon_loss.sum(dim=1).mean(),
            LossEnum.KL_LOSS.value: kl_loss.sum(dim=1).mean(),
        }

        for k, v in output.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN values detected in {k}")

        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        # Forward returns (conditional_likelihood, variational_posterior, z), we only need z
        conditional_likelihood, variational_posterior, z = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            condition=condition,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        conditional_likelihood_masked = None
        variational_posterior_masked = None
        z_masked = None

        loss_output = self.loss(
            counts=counts,
            conditional_likelihood=conditional_likelihood,
            variational_posterior=variational_posterior,
            z_sample=z,
            conditional_likelihood_masked=conditional_likelihood_masked,
            variational_posterior_masked=variational_posterior_masked,
            z_sample_masked=z_masked,
        )

        if hasattr(conditional_likelihood, "theta"):
            self.log("train_theta", conditional_likelihood.theta.mean(), on_step=True, on_epoch=True, sync_dist=True)
        if hasattr(conditional_likelihood, "scale") and conditional_likelihood.scale is not None:
            self.log("train_scale", conditional_likelihood.scale.mean(), on_step=True, on_epoch=True, sync_dist=True)

        loss = sum(loss_output.values())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        for k, v in loss_output.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            modules = {
                "encoder": self.vae_model.encoder,
                "decoder": self.vae_model.decoder,
                "encoder_head": self.vae_model.encoder_head,
                "decoder_head": self.vae_model.decoder_head,
            }
            if hasattr(self.vae_model, "input_layer"):
                modules["input_layer"] = self.vae_model.input_layer
            grad_norms = self._compute_gradient_norms(modules=modules)
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def shared_step(self, batch, batch_idx, stage: str, ema: bool = False) -> dict[str, Any]:
        counts, genes = batch[ModelEnum.COUNTS.value], batch[ModelEnum.GENES.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
            ModelEnum.LIBRARY_SIZE.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        model = self.vae_model
        model.eval()

        conditional_likelihood, variational_posterior, z = model(
            counts,
            genes,
            library_size,
            condition,
            counts_subset,
            genes_subset,
        )

        loss_output = self.loss(
            counts=counts,
            conditional_likelihood=conditional_likelihood,
            variational_posterior=variational_posterior,
            z_sample=z,
        )

        loss = sum(loss_output.values())
        metrics = {}
        metrics[f"{stage}_loss"] = loss
        for k, v in loss_output.items():
            metrics[f"{stage}_{k}"] = v

        counts_pred = conditional_likelihood.sample()

        real_library_size = counts.sum(dim=1, keepdim=True)
        counts_pred_scaled = torch.log1p((counts_pred / real_library_size) * 10_000)
        counts_true_scaled = torch.log1p((counts / real_library_size) * 10_000)

        counts_pred_zeros = (counts_pred == 0).float()
        counts_true_zeros = (counts == 0).float()

        metrics[f"{stage}_zeros_accuracy"] = (counts_pred_zeros == counts_true_zeros).float().mean()

        for k, fn in self.metric_fns.items():
            output = fn(counts_pred_scaled, counts_true_scaled)
            metrics[f"{stage}_{k}"] = torch.nanmean(output)

        return metrics

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        from scldm._utils import create_anndata_from_inference_output

        outputs = self.inference(batch)
        batch.update(outputs)
        if self.generation_args is not None:
            return tree_map(lambda x: x.cpu(), batch)
        else:
            return create_anndata_from_inference_output(tree_map(lambda x: x.cpu(), batch), self.trainer.datamodule)

    @torch.no_grad()
    def sample(
        self,
        library_size: torch.Tensor,
        genes: torch.Tensor,
    ) -> torch.Tensor:
        z_sample_prior = self.vae_model.prior.sample(n_samples=len(library_size))
        nb = self.vae_model.decode(z_sample_prior, genes, library_size, None)
        return nb.sample()

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        counts = batch[ModelEnum.COUNTS.value]
        genes = batch[ModelEnum.GENES.value]
        library_size = batch[ModelEnum.LIBRARY_SIZE.value]
        counts_subset = batch.get(ModelEnum.COUNTS_SUBSET.value, None)
        genes_subset = batch.get(ModelEnum.GENES_SUBSET.value, None)
        exclude_keys = (
            ModelEnum.COUNTS.value,
            ModelEnum.GENES.value,
            ModelEnum.LIBRARY_SIZE.value,
            ModelEnum.COUNTS_SUBSET.value,
            ModelEnum.GENES_SUBSET.value,
        )
        condition = {k: batch[k] for k in batch if k not in exclude_keys}

        nb, variational_posterior, _ = self.forward(
            counts=counts,
            genes=genes,
            library_size=library_size,
            condition=condition,
            counts_subset=counts_subset,
            genes_subset=genes_subset,
        )
        z = variational_posterior.sample((n_samples,))
        output: dict[str, torch.Tensor] = {
            "z_mean_flat": z.flatten(start_dim=1),
        }
        return output


# ---------------------------------------------------------------------------
# Schrödinger Bridge Matching for Cell Perturbation
# ---------------------------------------------------------------------------


class SchrodingerBridgeMatching(BaseModel):
    """
    Schrödinger Bridge Matching (SBM) for cell perturbation prediction.

    Architecture
    ------------
    source cell counts  ─┐
                          ├─► frozen VAE encoder ─► z0 ─┐
    target cell counts  ─┘                               │
                                                         ▼
    perturbation label ──────────────────────► ConditionedLightSBM
                                                         │
                                               (bridge matching training)
                                                         │
                                          z1' ◄──────────┘
                                           │
                          frozen VAE decoder ─► predicted perturbed counts

    Training objective
    ------------------
    Bridge Matching MSE loss (Gushchin et al., ICML 2024):
        t  ~ Uniform[0, 1 - safe_t]
        xt = t·z1 + (1-t)·z0 + √(ε·t·(1-t))·ξ
        loss = ‖ ConditionedLightSBM.get_drift(xt, t, c)
                 − (z1 - xt)/(1-t) ‖²

    z0 and z1 are sampled **independently** within each batch (the SBM
    potential itself learns the optimal coupling).

    Args:
        vae_model:            frozen TransformerVAE (encoder + decoder)
        sbm_model:            ConditionedLightSBM to be trained
        sbm_optimizer:        partial optimizer for SBM parameters
        vae_as_tokenizer:     config used to mark/freeze the VAE
        sbm_scheduler:        optional LR scheduler factory
        safe_t:               upper limit for t sampling (1 - safe_t)
        epsilon:              entropic regularisation ε (synced to sbm_model)
        euler_maruyama_steps: number of EM steps used at inference time
        calculate_grad_norms: whether to log gradient norms
        generation_args:      generation / inference config overrides
        inference_args:       inference mode config
    """

    def __init__(
        self,
        vae_model: TransformerVAE,
        sbm_model: ConditionedLightSBM,
        sbm_optimizer: Callable[[], Any],
        vae_as_tokenizer: DictConfig | None = None,
        sbm_scheduler: Callable[[int], float] | None = None,
        safe_t: float = 0.01,
        epsilon: float = 1.0,
        euler_maruyama_steps: int = 100,
        calculate_grad_norms: bool = False,
        generation_args: DictConfig | None = None,
        inference_args: DictConfig | None = None,
        gears_eval_freq: int = 50,
        gears_eval_sample_size: int = 512,
        gears_top_k_de: int = 20,
        ctrl_mean_path: str | None = None,
    ):
        super().__init__()

        self.vae_model = vae_model
        self.sbm_model = sbm_model
        self.sbm_optimizer = sbm_optimizer
        self.sbm_scheduler = sbm_scheduler
        self.safe_t = safe_t
        self.epsilon = epsilon
        self.euler_maruyama_steps = euler_maruyama_steps
        self.calculate_grad_norms = calculate_grad_norms
        self.generation_args = generation_args
        self.inference_args = inference_args
        self.vae_as_tokenizer = vae_as_tokenizer
        self.gears_eval_freq = gears_eval_freq
        self.gears_eval_sample_size = gears_eval_sample_size
        self.gears_top_k_de = gears_top_k_de
        self.ctrl_mean_path = ctrl_mean_path
        self.ctrl_mean: torch.Tensor | None = None
        self.is_gears_eval_epoch = False
        self._gears_accum: dict[int, dict[str, Any]] = {}

        if ctrl_mean_path is not None:
            ctrl_mean_np = np.load(Path(ctrl_mean_path))
            if ctrl_mean_np.ndim != 1:
                raise ValueError(
                    f"ctrl_mean loaded from {ctrl_mean_path} must be 1D, got shape {ctrl_mean_np.shape}."
                )
            self.ctrl_mean = torch.from_numpy(ctrl_mean_np.astype(np.float32))
            logger.info(f"Loaded GEARS ctrl_mean from {ctrl_mean_path} with shape {ctrl_mean_np.shape}.")

        # Sync epsilon to the SBM model buffer
        self.sbm_model.set_epsilon(epsilon)

        # Freeze VAE by default
        if vae_as_tokenizer is not None and not getattr(vae_as_tokenizer, "train", False):
            logger.info("VAE is frozen for SBM training.")
            for param in self.vae_model.parameters():
                param.requires_grad = False
            self.vae_model.eval()

        self.mse_loss = nn.MSELoss()
        self.mmd_metric_fns = MMD_METRICS

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        sbm_params = [p for p in self.sbm_model.parameters() if p.requires_grad]
        config: dict[str, Any] = {"optimizer": self.sbm_optimizer(sbm_params)}
        if self.sbm_scheduler is not None:
            config["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    config["optimizer"], self.sbm_scheduler
                ),
                "interval": "step",
            }
        return config

    # ------------------------------------------------------------------
    # VAE encode helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        counts_subset: torch.Tensor | None = None,
        genes_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode counts to latent z via the frozen VAE encoder."""
        z = self.vae_model.encode(counts, genes, counts_subset, genes_subset)
        return z  # (B, seq_len, latent_dim)

    def _flatten_z(self, z: torch.Tensor) -> torch.Tensor:
        return z.flatten(start_dim=1)  # (B, seq_len * latent_dim)

    def _unflatten_z(self, z_flat: torch.Tensor) -> torch.Tensor:
        seq_len = self.vae_model.encoder.latent_dim
        latent_dim = self.vae_model.encoder.latent_embedding
        return z_flat.view(-1, seq_len, latent_dim)

    def _stack_condition_indices(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = self.sbm_model.condition_label_order
        missing = [label for label in labels if f"condition_{label}" not in batch]
        if missing:
            raise KeyError(
                f"Missing condition keys in batch: {missing}. "
                f"Expected keys {[f'condition_{label}' for label in labels]}."
            )
        return torch.stack(
            [batch[f"condition_{label}"].long().squeeze(-1) for label in labels],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # ---- encode source (control) cells ----
        z0 = self._flatten_z(
            self._encode(
                batch["src_counts"],
                batch["src_genes"],
                batch.get("src_counts_subset"),
                batch.get("src_genes_subset"),
            )
        )  # (B, D)

        # ---- encode target (perturbed) cells ----
        z1 = self._flatten_z(
            self._encode(
                batch["tgt_counts"],
                batch["tgt_genes"],
                batch.get("tgt_counts_subset"),
                batch.get("tgt_genes_subset"),
            )
        )  # (B, D)

        condition_indices = self._stack_condition_indices(batch)  # (B, K_labels)

        B = z0.shape[0]
        device = z0.device

        # ---- sample t ----
        t = torch.rand(B, device=device) * (1.0 - self.safe_t)  # (B,)

        # ---- interpolate: x_t = t*z1 + (1-t)*z0 + sqrt(ε*t*(1-t))*ξ ----
        noise = torch.randn_like(z0)
        z_t = (
            t[:, None] * z1
            + (1 - t)[:, None] * z0
            + torch.sqrt(self.epsilon * t * (1 - t))[:, None] * noise
        )  # (B, D)

        # ---- bridge matching target drift ----
        drift_target = (z1 - z_t) / (1 - t[:, None])  # (B, D)

        # ---- predicted drift ----
        drift_pred = self.sbm_model.get_drift(z_t, t, condition_indices)  # (B, D)

        loss = self.mse_loss(drift_pred, drift_target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.calculate_grad_norms:
            grad_norms = self._compute_gradient_norms({"sbm": self.sbm_model})
            self.log_dict(grad_norms, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    # ------------------------------------------------------------------
    # Validation / shared step
    # ------------------------------------------------------------------

    def shared_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        stage: str,
        ema: bool = False,
    ) -> dict[str, Any]:
        # Encode under no_grad (frozen VAE). Drift uses autograd inside get_drift — do not wrap in no_grad.
        with torch.no_grad():
            z0 = self._flatten_z(
                self._encode(
                    batch["src_counts"],
                    batch["src_genes"],
                    batch.get("src_counts_subset"),
                    batch.get("src_genes_subset"),
                )
            )
            z1 = self._flatten_z(
                self._encode(
                    batch["tgt_counts"],
                    batch["tgt_genes"],
                    batch.get("tgt_counts_subset"),
                    batch.get("tgt_genes_subset"),
                )
            )
        condition_indices = self._stack_condition_indices(batch)
        B = z0.shape[0]
        t = torch.rand(B, device=z0.device) * (1.0 - self.safe_t)
        z_t = (
            t[:, None] * z1
            + (1 - t)[:, None] * z0
            + torch.sqrt(self.epsilon * t * (1 - t))[:, None] * torch.randn_like(z0)
        )
        drift_target = (z1 - z_t) / (1 - t[:, None])
        with torch.enable_grad():
            drift_pred = self.sbm_model.get_drift(z_t, t, condition_indices)
        loss = self.mse_loss(drift_pred, drift_target)
        return {f"{stage}_loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        should_eval = (
            self.ctrl_mean is not None
            and self.gears_eval_freq > 0
            and self.current_epoch > 0
            and self.current_epoch % self.gears_eval_freq == 0
        )
        self.is_gears_eval_epoch = bool(should_eval)
        self._gears_accum = {}
        if self.is_gears_eval_epoch:
            logger.info(
                "Starting GEARS evaluation at epoch "
                f"{self.current_epoch} (sample_size={self.gears_eval_sample_size}, top_k={self.gears_top_k_de})."
            )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        # Keep default validation loss logging from BaseModel.
        super().validation_step(batch, batch_idx)

        if not self.is_gears_eval_epoch or self.ctrl_mean is None:
            return

        counts_pred, _ = self.sample(
            src_counts=batch["src_counts"],
            src_genes=batch["src_genes"],
            condition_indices=self._stack_condition_indices(batch),
            tgt_genes=batch["tgt_genes"],
            tgt_library_size=batch["tgt_library_size"],
            src_counts_subset=batch.get("src_counts_subset"),
            src_genes_subset=batch.get("src_genes_subset"),
        )

        true_counts = batch["tgt_counts"]
        if "condition_gene" in batch:
            cond_idx = batch["condition_gene"].long().squeeze(-1)
        else:
            default_label = self.sbm_model.condition_label_order[0]
            cond_idx = batch[f"condition_{default_label}"].long().squeeze(-1)
        unique_conditions = torch.unique(cond_idx)
        for cond in unique_conditions:
            cond_int = int(cond.item())
            mask = cond_idx == cond
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue

            state = self._gears_accum.setdefault(cond_int, {"pred": [], "true": [], "n": 0})
            remain = int(self.gears_eval_sample_size) - int(state["n"])
            if remain <= 0:
                continue
            kept = indices[:remain]
            state["pred"].append(counts_pred[kept].detach().cpu())
            state["true"].append(true_counts[kept].detach().cpu())
            state["n"] += int(kept.numel())

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if not self.is_gears_eval_epoch or self.ctrl_mean is None:
            return

        per_condition_metrics: list[dict[str, float]] = []
        for condition, state in self._gears_accum.items():
            if len(state["pred"]) == 0:
                continue
            pred = torch.cat(state["pred"], dim=0)
            true = torch.cat(state["true"], dim=0)
            cond_metrics = gears_per_condition_metrics(
                pred_counts=pred,
                true_counts=true,
                ctrl_mean=self.ctrl_mean,
                top_k=self.gears_top_k_de,
            )
            cond_metrics["condition"] = float(condition)
            per_condition_metrics.append(cond_metrics)

        aggregated = gears_aggregate(per_condition_metrics)
        top_k = int(self.gears_top_k_de)
        self.log("gears_eval/pearson_all", aggregated["pearson_all"], on_epoch=True, sync_dist=True)
        self.log(
            f"gears_eval/pearson_top{top_k}_de",
            aggregated["pearson_top_de"],
            on_epoch=True,
            sync_dist=True,
        )
        self.log("gears_eval/mse_all", aggregated["mse_all"], on_epoch=True, sync_dist=True)
        self.log(
            f"gears_eval/mse_top{top_k}_de",
            aggregated["mse_top_de"],
            on_epoch=True,
            sync_dist=True,
        )
        self.log("gears_eval/n_conditions", aggregated["n_conditions"], on_epoch=True, sync_dist=True)

        logger.info(
            "GEARS evaluation completed: "
            f"n_conditions={aggregated['n_conditions']}, "
            f"pearson_all={aggregated['pearson_all']:.4f}, "
            f"pearson_top{top_k}_de={aggregated['pearson_top_de']:.4f}."
        )
        self._gears_accum = {}
        self.is_gears_eval_epoch = False

    # ------------------------------------------------------------------
    # Inference / sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        src_counts: torch.Tensor,
        src_genes: torch.Tensor,
        condition_indices: torch.Tensor,
        src_counts_subset: torch.Tensor | None = None,
        src_genes_subset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for inference: encode source → bridge → latent z1'.
        Returns flat latent tensor (B, D).
        """
        z0 = self._flatten_z(
            self._encode(src_counts, src_genes, src_counts_subset, src_genes_subset)
        )
        c_idx = condition_indices.long()
        # Run Euler-Maruyama to get final latent
        trajectory = self.sbm_model.sample_euler_maruyama(
            z0, c_idx, n_steps=self.euler_maruyama_steps
        )
        return trajectory[:, -1, :]  # (B, D) — final step

    @torch.no_grad()
    def sample(
        self,
        src_counts: torch.Tensor,
        src_genes: torch.Tensor,
        condition_indices: torch.Tensor,
        tgt_genes: torch.Tensor,
        tgt_library_size: torch.Tensor,
        src_counts_subset: torch.Tensor | None = None,
        src_genes_subset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full inference pipeline:
          source cell → bridge → latent z1' → VAE decode → perturbed counts.

        Returns
        -------
        counts_pred : (B, G)  sampled predicted perturbed counts
        z1_pred     : (B, D)  predicted perturbed latent (flat)
        """
        z1_flat = self.forward(
            src_counts, src_genes, condition_indices,
            src_counts_subset, src_genes_subset,
        )
        z1 = self._unflatten_z(z1_flat)  # (B, seq_len, latent_dim)

        size_factors = tgt_library_size.view(-1, 1).float()
        nb = self.vae_model.decode(z1, tgt_genes, size_factors)
        return nb.sample(), z1_flat

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        counts_pred, z1_flat = self.sample(
            src_counts=batch["src_counts"],
            src_genes=batch["src_genes"],
            condition_indices=self._stack_condition_indices(batch),
            tgt_genes=batch["tgt_genes"],
            tgt_library_size=batch["tgt_library_size"],
            src_counts_subset=batch.get("src_counts_subset"),
            src_genes_subset=batch.get("src_genes_subset"),
        )
        condition_key = "condition_gene"
        if condition_key not in batch:
            condition_key = f"condition_{self.sbm_model.condition_label_order[0]}"
        return {
            "predicted_counts": counts_pred.cpu(),
            "z1_predicted": z1_flat.cpu(),
            condition_key: batch[condition_key].cpu(),
            "tgt_counts": batch["tgt_counts"].cpu(),
        }

    @torch.no_grad()
    def inference(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.predict_step(batch, batch_idx=0)
