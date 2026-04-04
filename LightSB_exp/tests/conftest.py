"""
Shared fixtures for TransformerVAE + ConditionedLightSBM tests.

Dimensions are kept tiny for fast CPU runs.
"""
from __future__ import annotations

import torch
from pytest import fixture

from tests.tiny_models import (
    make_sbm_batch_dict,
    make_sbm_batch_with_subsets,
    make_tiny_conditioned_lightsbm,
    make_tiny_transformer_vae,
)


@fixture
def tiny_vae():
    return make_tiny_transformer_vae()


@fixture
def tiny_sbm():
    return make_tiny_conditioned_lightsbm()


@fixture
def sbm_batch() -> dict[str, torch.Tensor]:
    return make_sbm_batch_dict()


@fixture
def sbm_batch_with_subsets(sbm_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return make_sbm_batch_with_subsets(sbm_batch)
