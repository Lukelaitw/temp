"""Tiny TransformerVAE / ConditionedLightSBM builders shared by tests and conftest."""

from __future__ import annotations

import sys
from types import ModuleType

# Allow importing `scldm` when `scvi` is not installed.
try:
    import scvi.distributions as _scvi_dist_check  # noqa: F401
except ImportError:
    if "scvi.distributions" not in sys.modules:
        scvi_pkg = ModuleType("scvi")
        dist = ModuleType("scvi.distributions")

        class _NegativeBinomial:
            pass

        dist.NegativeBinomial = _NegativeBinomial  # type: ignore[misc, assignment]
        scvi_pkg.distributions = dist  # type: ignore[attr-defined]
        sys.modules["scvi"] = scvi_pkg
        sys.modules["scvi.distributions"] = dist

import torch

from scldm.layers import InputTransformerVAE
from scldm.lightsbm import ConditionedLightSBM
from scldm.nnets import Decoder, Encoder
from scldm.stochastic_layers import NegativeBinomialTransformerLayer
from scldm.vae import TransformerVAE

TEST_N_GENES = 32
TEST_SEQ_LEN = 32
TEST_BATCH_SIZE = 4
TEST_N_INDUCING = 4
TEST_N_EMBED_LATENT = 8
TEST_LATENT_FLAT_DIM = TEST_N_INDUCING * TEST_N_EMBED_LATENT
TEST_N_GENE_CONDITIONS = 8
TEST_N_CELL_LINES = 4
TEST_N_POTENTIALS = 4


def make_tiny_transformer_vae() -> TransformerVAE:
    n_embed = 32
    encoder = Encoder(
        n_layer=1,
        n_inducing_points=TEST_N_INDUCING,
        n_embed=n_embed,
        n_embed_latent=TEST_N_EMBED_LATENT,
        n_head=4,
        n_head_cross=2,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-8,
        norm_layer="layernorm",
        positional_encoding=False,
    )
    decoder = Decoder(
        n_genes=TEST_N_GENES,
        n_embed=n_embed,
        n_embed_latent=TEST_N_EMBED_LATENT,
        n_head=4,
        n_head_cross=2,
        n_layer=1,
        n_inducing_points=TEST_N_INDUCING,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-8,
        norm_layer="layernorm",
        shared_embedding=True,
        use_adaln=False,
    )
    input_layer = InputTransformerVAE(
        n_genes=TEST_N_GENES,
        n_embed=n_embed,
        agg_func="log1p",
    )
    decoder_head = NegativeBinomialTransformerLayer(
        n_genes=TEST_N_GENES,
        shared_theta=True,
        n_embed=n_embed,
        norm_layer="layernorm",
        layernorm_eps=1e-8,
    )
    return TransformerVAE(
        encoder=encoder,
        decoder=decoder,
        decoder_head=decoder_head,
        input_layer=input_layer,
    )


def make_tiny_conditioned_lightsbm() -> ConditionedLightSBM:
    return ConditionedLightSBM(
        dim=TEST_LATENT_FLAT_DIM,
        n_potentials=TEST_N_POTENTIALS,
        condition_label_order=["gene", "cell_line"],
        condition_vocab_sizes={
            "gene": TEST_N_GENE_CONDITIONS,
            "cell_line": TEST_N_CELL_LINES,
        },
        cond_embed_dim=16,
        epsilon=1.0,
        sampling_batch_size=8,
        S_diagonal_init=0.1,
        cond_hidden_dim=32,
    )


def make_sbm_batch_dict() -> dict[str, torch.Tensor]:
    b, g = TEST_BATCH_SIZE, TEST_SEQ_LEN
    torch.manual_seed(0)
    src_counts = torch.rand(b, g, dtype=torch.float32) * 3.0
    tgt_counts = torch.rand(b, g, dtype=torch.float32) * 3.0
    genes = torch.randint(
        low=1,
        high=TEST_N_GENES + 1,
        size=(b, g),
        dtype=torch.long,
    )
    lib = torch.rand(b, 1, dtype=torch.float32) * 100.0 + 10.0
    cond_gene = torch.randint(0, TEST_N_GENE_CONDITIONS, size=(b, 1), dtype=torch.long)
    cond_cell_line = torch.randint(0, TEST_N_CELL_LINES, size=(b, 1), dtype=torch.long)
    return {
        "src_counts": src_counts,
        "src_genes": genes,
        "src_library_size": lib,
        "tgt_counts": tgt_counts,
        "tgt_genes": genes.clone(),
        "tgt_library_size": lib.clone(),
        "condition_gene": cond_gene,
        "condition_cell_line": cond_cell_line,
    }


def make_sbm_batch_with_subsets(
    base: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    out = dict(base or make_sbm_batch_dict())
    b, sub = TEST_BATCH_SIZE, TEST_SEQ_LEN // 2
    torch.manual_seed(1)
    out["src_counts_subset"] = torch.rand(b, sub, dtype=torch.float32)
    out["src_genes_subset"] = torch.randint(1, TEST_N_GENES + 1, (b, sub), dtype=torch.long)
    out["tgt_counts_subset"] = torch.rand(b, sub, dtype=torch.float32)
    out["tgt_genes_subset"] = torch.randint(1, TEST_N_GENES + 1, (b, sub), dtype=torch.long)
    return out
