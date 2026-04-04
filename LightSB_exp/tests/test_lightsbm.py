"""Tests for ConditionedLightSBM (drift, forward sampling)."""

import torch
import torch.nn.functional as F

from tests.tiny_models import (
    TEST_BATCH_SIZE,
    TEST_LATENT_FLAT_DIM,
    make_tiny_conditioned_lightsbm,
)


def test_conditioned_lightsbm_get_drift_shape():
    m = make_tiny_conditioned_lightsbm()
    b = TEST_BATCH_SIZE
    x = torch.randn(b, TEST_LATENT_FLAT_DIM)
    t = torch.rand(b) * 0.99
    gene_idx = torch.randint(0, m.condition_vocab_sizes["gene"], (b,), dtype=torch.long)
    cell_line_idx = torch.randint(0, m.condition_vocab_sizes["cell_line"], (b,), dtype=torch.long)
    c = torch.stack([gene_idx, cell_line_idx], dim=1)
    drift = m.get_drift(x, t, c)
    assert drift.shape == (b, TEST_LATENT_FLAT_DIM)


def test_conditioned_lightsbm_get_drift_backward():
    m = make_tiny_conditioned_lightsbm()
    b = TEST_BATCH_SIZE
    x = torch.randn(b, TEST_LATENT_FLAT_DIM, requires_grad=True)
    t = torch.rand(b) * 0.99
    gene_idx = torch.randint(0, m.condition_vocab_sizes["gene"], (b,), dtype=torch.long)
    cell_line_idx = torch.randint(0, m.condition_vocab_sizes["cell_line"], (b,), dtype=torch.long)
    c = torch.stack([gene_idx, cell_line_idx], dim=1)
    drift = m.get_drift(x.detach(), t, c)
    loss = drift.pow(2).mean()
    loss.backward()
    # x was detached inside get_drift; grads flow through SBM params
    assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_conditioned_lightsbm_forward_sample_shape():
    m = make_tiny_conditioned_lightsbm()
    m.eval()
    b = TEST_BATCH_SIZE
    x = torch.randn(b, TEST_LATENT_FLAT_DIM)
    gene_idx = torch.randint(0, m.condition_vocab_sizes["gene"], (b,), dtype=torch.long)
    cell_line_idx = torch.randint(0, m.condition_vocab_sizes["cell_line"], (b,), dtype=torch.long)
    c = torch.stack([gene_idx, cell_line_idx], dim=1)
    with torch.no_grad():
        y = m(x, c)
    assert y.shape == (b, TEST_LATENT_FLAT_DIM)


def test_bridge_matching_mse_is_finite():
    m = make_tiny_conditioned_lightsbm()
    b = TEST_BATCH_SIZE
    d = TEST_LATENT_FLAT_DIM
    z0 = torch.randn(b, d)
    z1 = torch.randn(b, d)
    t = torch.rand(b) * 0.99
    noise = torch.randn_like(z0)
    eps = float(m.epsilon)
    z_t = t[:, None] * z1 + (1 - t)[:, None] * z0 + torch.sqrt(eps * t * (1 - t))[:, None] * noise
    drift_target = (z1 - z_t) / (1 - t[:, None])
    gene_idx = torch.randint(0, m.condition_vocab_sizes["gene"], (b,), dtype=torch.long)
    cell_line_idx = torch.randint(0, m.condition_vocab_sizes["cell_line"], (b,), dtype=torch.long)
    c = torch.stack([gene_idx, cell_line_idx], dim=1)
    drift_pred = m.get_drift(z_t, t, c)
    loss = F.mse_loss(drift_pred, drift_target)
    assert torch.isfinite(loss)
