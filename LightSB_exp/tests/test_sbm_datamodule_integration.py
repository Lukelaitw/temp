"""
End-to-end: synthetic h5ad + metadata.json → SBMDataModule → one training batch → SBM loss.

Requires ``cellarium-ml`` (see ``scldm.datamodule``) and the rest of project deps.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from tests.tiny_models import (
    TEST_LATENT_FLAT_DIM,
    TEST_N_GENES,
    TEST_SEQ_LEN,
    make_tiny_transformer_vae,
)


@pytest.mark.integration
def test_sbm_datamodule_batch_and_training_step(tmp_path: Path) -> None:
    pytest.importorskip("cellarium.ml.data", reason="cellarium-ml required for SBMDataModule")

    from scldm.lightsbm import ConditionedLightSBM
    from scldm.models import SchrodingerBridgeMatching
    from scldm.sbm_datamodule import SBMDataModule
    from scldm.encoder import VocabularyEncoderSimplified

    n_genes = TEST_N_GENES
    genes = [f"g{i}" for i in range(n_genes)]
    gene_labels = ["non-targeting", "PertA", "PertB", "PertC"]
    cell_line_labels = ["A549", "K562"]
    n_gene_conds = len(gene_labels)
    n_cell_lines = len(cell_line_labels)

    meta = {
        "genes": genes,
        "labels": {"gene": gene_labels, "cell_line": cell_line_labels},
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    rng = np.random.default_rng(0)

    def _make_adata(n_ctrl: int, n_per_pert: int) -> ad.AnnData:
        rows = []
        cell_lines = []
        x_rows = []
        # control
        for _ in range(n_ctrl):
            rows.append("non-targeting")
            cell_lines.append(cell_line_labels[0])
            x_rows.append(rng.poisson(1.5, n_genes).astype(np.float32))
        for lab in gene_labels[1:]:
            for _ in range(n_per_pert):
                rows.append(lab)
                cell_lines.append(cell_line_labels[int(rng.integers(0, n_cell_lines))])
                x_rows.append(rng.poisson(2.0, n_genes).astype(np.float32))
        x = np.stack(x_rows, axis=0)
        obs = pd.DataFrame({"gene": rows, "cell_line": cell_lines})
        var = pd.DataFrame(index=genes)
        return ad.AnnData(X=x, obs=obs, var=var)

    train_path = tmp_path / "train.h5ad"
    test_path = tmp_path / "test.h5ad"
    _make_adata(n_ctrl=6, n_per_pert=4).write_h5ad(train_path)
    _make_adata(n_ctrl=2, n_per_pert=2).write_h5ad(test_path)

    enc = VocabularyEncoderSimplified(
        adata_path=None,
        class_vocab_sizes={"gene": n_gene_conds, "cell_line": n_cell_lines},
        mask_token="<MASK>",
        mask_token_idx=0,
        n_genes=n_genes,
        guidance_weight={"gene": 1.0, "cell_line": 1.0},
        mu_size_factor=None,
        sd_size_factor=None,
        condition_strategy="mutually_exclusive",
        metadata_genes=None,
        metadata_json=meta_path,
    )

    dm = SBMDataModule(
        source_adata_path=train_path,
        train_adata_path=train_path,
        test_adata_path=test_path,
        control_label_key="gene",
        control_label_value="non-targeting",
        perturbation_label_keys=["gene", "cell_line"],
        emb_condition_labels=["gene", "cell_line"],
        vocabulary_encoder=enc,
        adata_attr="X",
        adata_key=None,
        batch_size=4,
        test_batch_size=4,
        num_workers=0,
        seed=0,
        genes_seq_len=TEST_SEQ_LEN,
        sample_genes="none",
        n_samples_per_epoch=32,
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert (
        "src_counts" in batch
        and "tgt_counts" in batch
        and "condition_gene" in batch
        and "condition_cell_line" in batch
    )
    assert batch["src_counts"].shape[0] == 4

    vae = make_tiny_transformer_vae()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    sbm = ConditionedLightSBM(
        dim=TEST_LATENT_FLAT_DIM,
        n_potentials=4,
        condition_label_order=["gene", "cell_line"],
        condition_vocab_sizes={"gene": n_gene_conds, "cell_line": n_cell_lines},
        cond_embed_dim=16,
        epsilon=1.0,
        sampling_batch_size=8,
        S_diagonal_init=0.1,
        cond_hidden_dim=32,
    )

    module = SchrodingerBridgeMatching(
        vae_model=vae,
        sbm_model=sbm,
        sbm_optimizer=partial(torch.optim.AdamW, lr=1e-3),
        vae_as_tokenizer=OmegaConf.create({"train": False}),
        sbm_scheduler=None,
        safe_t=0.01,
        epsilon=1.0,
        euler_maruyama_steps=5,
        calculate_grad_norms=False,
        generation_args=None,
        inference_args=None,
    )
    module.train()
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.integration
def test_trainer_fast_dev_run_cpu(tmp_path: Path) -> None:
    pytest.importorskip("cellarium.ml.data", reason="cellarium-ml required for SBMDataModule")

    import pytorch_lightning as pl

    from scldm.encoder import VocabularyEncoderSimplified
    from scldm.lightsbm import ConditionedLightSBM
    from scldm.models import SchrodingerBridgeMatching
    from scldm.sbm_datamodule import SBMDataModule

    n_genes = TEST_N_GENES
    genes = [f"g{i}" for i in range(n_genes)]
    gene_labels = ["non-targeting", "PertA", "PertB", "PertC"]
    cell_line_labels = ["A549", "K562"]
    n_gene_conds = len(gene_labels)
    n_cell_lines = len(cell_line_labels)

    meta = {"genes": genes, "labels": {"gene": gene_labels, "cell_line": cell_line_labels}}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    rng = np.random.default_rng(1)

    def _make_adata(n_ctrl: int, n_per_pert: int) -> ad.AnnData:
        rows, cell_lines, x_rows = [], [], []
        for _ in range(n_ctrl):
            rows.append("non-targeting")
            cell_lines.append(cell_line_labels[0])
            x_rows.append(rng.poisson(1.5, n_genes).astype(np.float32))
        for lab in gene_labels[1:]:
            for _ in range(n_per_pert):
                rows.append(lab)
                cell_lines.append(cell_line_labels[int(rng.integers(0, n_cell_lines))])
                x_rows.append(rng.poisson(2.0, n_genes).astype(np.float32))
        x = np.stack(x_rows, axis=0)
        obs = pd.DataFrame({"gene": rows, "cell_line": cell_lines})
        var = pd.DataFrame(index=genes)
        return ad.AnnData(X=x, obs=obs, var=var)

    train_path = tmp_path / "tr.h5ad"
    test_path = tmp_path / "te.h5ad"
    _make_adata(4, 2).write_h5ad(train_path)
    _make_adata(2, 1).write_h5ad(test_path)

    enc = VocabularyEncoderSimplified(
        adata_path=None,
        class_vocab_sizes={"gene": n_gene_conds, "cell_line": n_cell_lines},
        mask_token="<MASK>",
        mask_token_idx=0,
        n_genes=n_genes,
        guidance_weight={"gene": 1.0, "cell_line": 1.0},
        mu_size_factor=None,
        sd_size_factor=None,
        condition_strategy="mutually_exclusive",
        metadata_genes=None,
        metadata_json=meta_path,
    )

    dm = SBMDataModule(
        source_adata_path=train_path,
        train_adata_path=train_path,
        test_adata_path=test_path,
        control_label_key="gene",
        control_label_value="non-targeting",
        perturbation_label_keys=["gene", "cell_line"],
        emb_condition_labels=["gene", "cell_line"],
        vocabulary_encoder=enc,
        batch_size=2,
        test_batch_size=2,
        num_workers=0,
        genes_seq_len=TEST_SEQ_LEN,
        sample_genes="none",
        n_samples_per_epoch=8,
    )
    dm.setup()

    vae = make_tiny_transformer_vae()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    sbm = ConditionedLightSBM(
        dim=TEST_LATENT_FLAT_DIM,
        n_potentials=4,
        condition_label_order=["gene", "cell_line"],
        condition_vocab_sizes={"gene": n_gene_conds, "cell_line": n_cell_lines},
        cond_embed_dim=16,
        epsilon=1.0,
        sampling_batch_size=4,
        S_diagonal_init=0.1,
        cond_hidden_dim=32,
    )

    module = SchrodingerBridgeMatching(
        vae_model=vae,
        sbm_model=sbm,
        sbm_optimizer=partial(torch.optim.AdamW, lr=1e-3),
        vae_as_tokenizer=OmegaConf.create({"train": False}),
        epsilon=1.0,
        euler_maruyama_steps=3,
    )

    # inference_mode=True (Lightning default) blocks autograd inside ConditionedLightSBM.get_drift during val.
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=False,
    )
    trainer.fit(module, datamodule=dm)
