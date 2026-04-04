"""Tests for TransformerVAE encode / forward shapes."""

import torch

from scldm.vae import TransformerVAE
from tests.tiny_models import (
    TEST_BATCH_SIZE,
    TEST_N_EMBED_LATENT,
    TEST_N_INDUCING,
    TEST_SEQ_LEN,
    make_tiny_transformer_vae,
)


def test_encode_output_shape(tiny_vae: TransformerVAE):
    b, s = TEST_BATCH_SIZE, TEST_SEQ_LEN
    counts = torch.rand(b, s)
    genes = torch.randint(1, tiny_vae.input_layer.gene_embedding.num_embeddings, (b, s))
    # FlexAttention on CPU does not support grad; inference matches frozen-VAE encode path.
    with torch.inference_mode():
        z = tiny_vae.encode(counts, genes)
    assert z.shape == (b, TEST_N_INDUCING, TEST_N_EMBED_LATENT)


def test_forward_output_shapes(tiny_vae: TransformerVAE):
    b, s = TEST_BATCH_SIZE, TEST_SEQ_LEN
    counts = torch.rand(b, s)
    genes = torch.randint(1, tiny_vae.input_layer.gene_embedding.num_embeddings, (b, s))
    library_size = torch.rand(b, 1) * 50.0 + 5.0
    with torch.inference_mode():
        mu, theta, h_z = tiny_vae.forward(counts, genes, library_size)
    assert h_z.shape == (b, TEST_N_INDUCING, TEST_N_EMBED_LATENT)
    assert mu.shape[0] == b and theta.shape[0] == b


def test_forward_without_explicit_subsets_matches_encode_path(tiny_vae: TransformerVAE):
    """forward() should fall back to full counts/genes when subsets are None (vae.py)."""
    b, s = TEST_BATCH_SIZE, TEST_SEQ_LEN
    counts = torch.rand(b, s)
    genes = torch.randint(1, tiny_vae.input_layer.gene_embedding.num_embeddings, (b, s))
    library_size = torch.rand(b, 1) * 50.0 + 5.0
    with torch.inference_mode():
        _, _, h_z_fwd = tiny_vae.forward(counts, genes, library_size)
        h_z_enc = tiny_vae.encode(counts, genes)
    torch.testing.assert_close(h_z_fwd, h_z_enc)


def test_factory_builds_module():
    m = make_tiny_transformer_vae()
    assert isinstance(m, TransformerVAE)
