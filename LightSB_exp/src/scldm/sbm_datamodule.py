"""
DataModule for Schrödinger Bridge Matching (SBM) cell perturbation experiments.

This module provides paired (source, target, condition) batches where:
  - source cells = unperturbed / control cells  (P0)
  - target cells = perturbed cells for a given condition (P1(c))

The source and target within each batch are sampled **independently** (no
OT coupling at data-loading time); LightSBM learns the optimal coupling.

Usage example
-------------
In the Hydra config (configs/datamodule/sbm.yaml):

    _target_: scldm.sbm_datamodule.SBMDataModule
    source_adata_path: /path/to/source.h5ad
    train_adata_path: /path/to/train.h5ad
    test_adata_path:  /path/to/test.h5ad
    control_label_key: gene           # obs column that holds perturbation id
    control_label_value: non-targeting  # kept for config compatibility
    perturbation_label_keys: [gene]   # condition labels included in the batch
    val_fraction: 0.1                 # split validation from train perturbations
    vocabulary_encoder: ...
    batch_size: 128
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from scldm.constants import ModelEnum
from scldm.datamodule import tokenize_cells
from scldm.encoder import VocabularyEncoderSimplified
from scldm.logger import logger


# ---------------------------------------------------------------------------
# Map-style Dataset
# ---------------------------------------------------------------------------

class PairedPerturbationDataset(Dataset):
    """
    A map-style dataset that returns (source, target, condition) triples.

    For each index `i` the dataset:
      1. Picks a random control cell as source.
      2. Picks a random condition from the available conditions.
      3. Picks a random perturbed cell matching that condition as target.

    The randomness is seeded per-index so that batches are reproducible
    within an epoch when using a fixed `epoch_seed`.

    Args:
        control_adata:    AnnData with only control cells.
        perturbed_adata:  AnnData with only perturbed cells (all conditions).
        condition_col:    obs column that contains the perturbation condition.
        vocabulary_encoder: maps gene names and label strings to integer indices.
        adata_attr:       attribute of AnnData holding count matrix (``"X"``
                          or ``"layers"``).
        adata_key:        key within the attribute (``None`` for sparse X).
        genes_seq_len:    maximum gene sequence length for tokenisation.
        sample_genes:     gene sub-sampling strategy.
        n_samples:        length of the dataset (virtual; controls steps/epoch).
        epoch_seed:       base seed; incremented each epoch via ``set_epoch``.
    """

    def __init__(
        self,
        control_adata: ad.AnnData,
        perturbed_adata: ad.AnnData,
        condition_col: str,
        emb_condition_labels: list[str] | None,
        vocabulary_encoder: VocabularyEncoderSimplified,
        condition_obs_column_map: dict[str, str] | None = None,
        adata_attr: str = "X",
        adata_key: str | None = None,
        genes_seq_len: int = 2000,
        sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"] = "none",
        n_samples: int = 100_000,
        epoch_seed: int = 42,
    ):
        super().__init__()
        self.control_adata = control_adata
        self.perturbed_adata = perturbed_adata
        self.condition_col = condition_col
        self.emb_condition_labels = emb_condition_labels or [condition_col]
        self.vocabulary_encoder = vocabulary_encoder
        self.condition_obs_column_map = condition_obs_column_map or {}
        self.adata_attr = adata_attr
        self.adata_key = adata_key
        self.genes_seq_len = genes_seq_len
        self.sample_genes = sample_genes
        self.n_samples = n_samples
        self.epoch_seed = epoch_seed

        # Build per-condition index lists once
        perturb_obs = perturbed_adata.obs[condition_col]
        self.conditions: list[str] = sorted(perturb_obs.unique().tolist())
        self.condition_to_indices: dict[str, np.ndarray] = {
            cond: np.where(perturb_obs == cond)[0]
            for cond in self.conditions
        }
        self.n_control = len(control_adata)
        logger.info(
            f"PairedPerturbationDataset: {self.n_control} control cells, "
            f"{len(perturbed_adata)} perturbed cells across "
            f"{len(self.conditions)} conditions; embedding labels={self.emb_condition_labels}."
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch_seed = epoch

    def __len__(self) -> int:
        return self.n_samples

    def _get_counts(self, adata: ad.AnnData, idx: int) -> np.ndarray:
        """Return a (1, G) dense count matrix for the i-th cell."""
        sub = adata[idx]
        if self.adata_attr == "X":
            x = sub.X
        else:
            x = getattr(sub, self.adata_attr)[self.adata_key]
        if hasattr(x, "toarray"):
            x = x.toarray()
        return x.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed=self.epoch_seed * 2_000_000 + idx)

        # --- source: random control cell ---
        src_idx = int(rng.integers(0, self.n_control))
        src_counts = self._get_counts(self.control_adata, src_idx)  # (1, G)

        # --- pick a random condition, then a random perturbed cell ---
        cond_name = self.conditions[int(rng.integers(0, len(self.conditions)))]
        cond_pool = self.condition_to_indices[cond_name]
        tgt_idx = int(rng.choice(cond_pool))
        tgt_counts = self._get_counts(self.perturbed_adata, tgt_idx)  # (1, G)

        # --- encode selected condition labels to integer indices ---
        encoded_conditions: dict[str, np.ndarray] = {}
        for label in self.emb_condition_labels:
            if label == self.condition_col:
                label_value = cond_name
            else:
                obs_col = self.condition_obs_column_map.get(label, label)
                label_value = self.perturbed_adata.obs[obs_col].iloc[tgt_idx]
            encoded = self.vocabulary_encoder.encode_metadata(
                np.array([label_value]), label=label
            )
            if encoded is None or np.any(encoded == None):  # noqa: E711
                raise ValueError(
                    f"Unknown metadata value for label='{label}' value='{label_value}'. "
                    "Please ensure metadata_json/class_vocab_sizes include this label value."
                )
            encoded_conditions[label] = encoded

        # --- tokenise genes (both cells share the same var_names) ---
        src_var_names = self.control_adata.var_names.tolist()
        tgt_var_names = self.perturbed_adata.var_names.tolist()

        src_tok = tokenize_cells(
            cell=src_counts,
            var_names=src_var_names,
            encoder=self.vocabulary_encoder,
            genes_seq_len=self.genes_seq_len,
            sample_genes=self.sample_genes,
            gene_tokens_key="src_genes",
            counts_key="src_counts",
        )
        tgt_tok = tokenize_cells(
            cell=tgt_counts,
            var_names=tgt_var_names,
            encoder=self.vocabulary_encoder,
            genes_seq_len=self.genes_seq_len,
            sample_genes=self.sample_genes,
            gene_tokens_key="tgt_genes",
            counts_key="tgt_counts",
        )

        sample: dict[str, Any] = {
            "src_counts": src_tok["src_counts"],        # (1, G)
            "src_genes": src_tok["src_genes"],          # (1, G)
            "src_library_size": src_tok["library_size"], # (1, 1)
            "tgt_counts": tgt_tok["tgt_counts"],        # (1, G)
            "tgt_genes": tgt_tok["tgt_genes"],          # (1, G)
            "tgt_library_size": tgt_tok["library_size"],# (1, 1)
        }
        for label, encoded in encoded_conditions.items():
            sample[f"condition_{label}"] = encoded
        # Backward-compatibility path for legacy single-condition consumers.
        if len(self.emb_condition_labels) == 1 and self.emb_condition_labels[0] == self.condition_col:
            sample["condition"] = encoded_conditions[self.condition_col]

        # Propagate subset arrays if produced by the tokeniser
        for prefix, tok in [("src", src_tok), ("tgt", tgt_tok)]:
            for extra_key in (ModelEnum.GENES_SUBSET.value, ModelEnum.COUNTS_SUBSET.value):
                if extra_key in tok:
                    sample[f"{prefix}_{extra_key}"] = tok[extra_key]

        return sample


def _sbm_collate(
    batch: list[dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Collate a list of single-cell samples into a batched dict of tensors."""
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        arr = np.concatenate([item[key] for item in batch], axis=0)
        out[key] = torch.from_numpy(arr)
    return out


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class SBMDataModule(LightningDataModule):
    """
    LightningDataModule for Schrödinger Bridge Matching.

    Loads source, train, and test AnnData files where:
      - source contains control cells (P0)
      - train and test contain perturbed cells (P1)
    Each training batch contains independently sampled (source, target,
    condition) triples for bridge matching.

    Args:
        source_adata_path:     path to source/control .h5ad
        train_adata_path:      path to perturbed training .h5ad
        test_adata_path:       path to perturbed test .h5ad
        control_label_key:     obs column used to identify perturbation
                               (e.g. ``"gene"``)
        control_label_value:   value in that column marking control cells
                               (e.g. ``"non-targeting"``)
        perturbation_label_keys: list of obs columns to include as condition
                               labels in the batch (default: same as
                               ``control_label_key``)
        vocabulary_encoder:    shared VocabularyEncoderSimplified
        adata_attr:            AnnData attribute holding counts (``"X"`` or
                               ``"layers"``)
        adata_key:             key within layers (``None`` for X)
        batch_size:            training batch size
        test_batch_size:       validation / test batch size
        num_workers:           DataLoader worker count
        seed:                  base random seed
        genes_seq_len:         gene sequence length after tokenisation
        sample_genes:          gene sub-sampling strategy
        n_samples_per_epoch:   virtual dataset length (controls steps / epoch)
        val_fraction:          fraction of train perturbations used for validation
    """

    def __init__(
        self,
        source_adata_path: Path | str,
        train_adata_path: Path | str,
        test_adata_path: Path | str,
        control_label_key: str,
        control_label_value: str,
        vocabulary_encoder: VocabularyEncoderSimplified,
        perturbation_label_keys: list[str] | None = None,
        emb_condition_labels: list[str] | None = None,
        condition_obs_column_map: dict[str, str] | None = None,
        adata_attr: str = "X",
        adata_key: str | None = None,
        batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 42,
        genes_seq_len: int = 2000,
        sample_genes: Literal[
            "random", "weighted", "expressed", "expressed_zero", "none"
        ] = "none",
        n_samples_per_epoch: int = 100_000,
        val_fraction: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.source_adata_path = Path(source_adata_path)
        self.train_adata_path = Path(train_adata_path)
        self.test_adata_path = Path(test_adata_path)
        self.control_label_key = control_label_key
        self.control_label_value = control_label_value
        self.perturbation_label_keys = perturbation_label_keys or [control_label_key]
        self.emb_condition_labels = emb_condition_labels or self.perturbation_label_keys
        self.condition_obs_column_map = condition_obs_column_map or {}
        self.vocabulary_encoder = vocabulary_encoder
        self.adata_attr = adata_attr
        self.adata_key = adata_key
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.genes_seq_len = genes_seq_len
        self.sample_genes = sample_genes
        self.n_samples_per_epoch = n_samples_per_epoch
        self.val_fraction = val_fraction

        # Set by setup()
        self.n_cells: int = 0
        self.train_dataset: PairedPerturbationDataset | None = None
        self.val_dataset: PairedPerturbationDataset | None = None
        self.test_dataset: PairedPerturbationDataset | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dataset(
        self,
        ctrl: ad.AnnData,
        pert: ad.AnnData,
        n_samples: int,
        epoch_seed: int,
    ) -> PairedPerturbationDataset:
        return PairedPerturbationDataset(
            control_adata=ctrl,
            perturbed_adata=pert,
            condition_col=self.control_label_key,
            emb_condition_labels=self.emb_condition_labels,
            vocabulary_encoder=self.vocabulary_encoder,
            condition_obs_column_map=self.condition_obs_column_map,
            adata_attr=self.adata_attr,
            adata_key=self.adata_key,
            genes_seq_len=self.genes_seq_len,
            sample_genes=self.sample_genes,
            n_samples=n_samples,
            epoch_seed=epoch_seed,
        )

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        logger.info("SBMDataModule.setup() called")

        source_adata = ad.read_h5ad(self.source_adata_path)
        train_adata = ad.read_h5ad(self.train_adata_path)
        test_adata = ad.read_h5ad(self.test_adata_path)

        n_train = len(train_adata)
        if n_train < 2:
            raise ValueError(
                f"train_adata must contain at least 2 cells to split train/val, got {n_train}."
            )
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}.")

        val_n = int(round(n_train * self.val_fraction))
        val_n = min(max(val_n, 1), n_train - 1)
        rng = np.random.default_rng(seed=self.seed)
        perm = rng.permutation(n_train)
        val_indices = perm[:val_n]
        train_indices = perm[val_n:]
        train_pert = train_adata[train_indices].copy()
        val_pert = train_adata[val_indices].copy()

        self.n_cells = len(train_adata)
        logger.info(
            "Using source/train/test with split: source=%d, train_pert=%d, val_pert=%d, test_pert=%d",
            len(source_adata),
            len(train_pert),
            len(val_pert),
            len(test_adata),
        )

        self.train_dataset = self._make_dataset(
            source_adata, train_pert,
            n_samples=self.n_samples_per_epoch,
            epoch_seed=self.seed,
        )
        self.val_dataset = self._make_dataset(
            source_adata, val_pert,
            n_samples=max(1024, len(val_pert)),
            epoch_seed=self.seed + 1,
        )
        self.test_dataset = self._make_dataset(
            source_adata,
            test_adata,
            n_samples=max(1024, len(test_adata)),
            epoch_seed=self.seed + 2,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_sbm_collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_sbm_collate,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_sbm_collate,
            pin_memory=True,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Epoch management (called by BaseModel.on_train_epoch_start)
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)
