"""Tests for SchrodingerBridgeMatching training_step and encode subset wiring."""

from functools import partial

import torch
from omegaconf import OmegaConf

from scldm.models import SchrodingerBridgeMatching
from scldm.vae import TransformerVAE
from tests.tiny_models import (
    TEST_LATENT_FLAT_DIM,
    make_tiny_conditioned_lightsbm,
    make_tiny_transformer_vae,
)


def _make_sbm_module(
    vae: TransformerVAE,
) -> SchrodingerBridgeMatching:
    sbm = make_tiny_conditioned_lightsbm()
    return SchrodingerBridgeMatching(
        vae_model=vae,
        sbm_model=sbm,
        sbm_optimizer=partial(torch.optim.AdamW, lr=1e-3),
        vae_as_tokenizer=OmegaConf.create({"train": False}),
        sbm_scheduler=None,
        safe_t=0.01,
        epsilon=float(sbm.epsilon),
        euler_maruyama_steps=10,
        calculate_grad_norms=False,
        generation_args=None,
        inference_args=None,
    )


def test_training_step_finite_and_backward(tiny_vae: TransformerVAE, sbm_batch):
    m = _make_sbm_module(tiny_vae)
    m.train()
    loss = m.training_step(sbm_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None for p in m.sbm_model.parameters() if p.requires_grad
    )


def test_training_step_optimizer_step(tiny_vae: TransformerVAE, sbm_batch):
    m = _make_sbm_module(tiny_vae)
    opt_cfg = m.configure_optimizers()
    opt = opt_cfg["optimizer"]
    m.train()
    opt.zero_grad()
    loss = m.training_step(sbm_batch, 0)
    loss.backward()
    opt.step()


def test_encode_subsets_match_vae_encode(tiny_vae: TransformerVAE, sbm_batch_with_subsets):
    """SchrodingerBridgeMatching._encode must pass (counts_subset, genes_subset) order to VAE.encode."""
    m = _make_sbm_module(tiny_vae)
    b = sbm_batch_with_subsets["src_counts"].shape[0]
    z_direct = tiny_vae.encode(
        sbm_batch_with_subsets["src_counts"],
        sbm_batch_with_subsets["src_genes"],
        sbm_batch_with_subsets["src_counts_subset"],
        sbm_batch_with_subsets["src_genes_subset"],
    )
    z_module = m._encode(
        sbm_batch_with_subsets["src_counts"],
        sbm_batch_with_subsets["src_genes"],
        sbm_batch_with_subsets["src_counts_subset"],
        sbm_batch_with_subsets["src_genes_subset"],
    )
    torch.testing.assert_close(z_module, z_direct)
    assert z_module.shape == (b, tiny_vae.encoder.latent_dim, tiny_vae.encoder.latent_embedding)


def test_flatten_unflatten_roundtrip(tiny_vae: TransformerVAE):
    m = _make_sbm_module(tiny_vae)
    b = 2
    s = tiny_vae.encoder.latent_dim
    e = tiny_vae.encoder.latent_embedding
    z = torch.randn(b, s, e)
    flat = m._flatten_z(z)
    assert flat.shape == (b, TEST_LATENT_FLAT_DIM)
    back = m._unflatten_z(flat)
    torch.testing.assert_close(back, z)
