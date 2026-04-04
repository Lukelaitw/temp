import math
from collections.abc import Sequence
from functools import partial
from typing import Literal

import numpy as np
import ot
import torch
from scipy import linalg
from sklearn.decomposition import PCA
from torch import nn


class RBFKernel(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = (x**2).sum(dim=1, keepdim=True)  # Bx x 1
        y_norm = (y**2).sum(dim=1, keepdim=True)  # By x 1
        squared_ell_2 = x_norm - 2 * x @ y.T + y_norm.T  # Bx x By

        return torch.exp(-self.scale * squared_ell_2)


class BrayCurtisKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.abs(x - y).sum(dim=2)  # Bx x By
        denominator = torch.abs(x + y).sum(dim=2) + 1e-8

        return 1 - numerator / denominator


class TanimotoKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = (x * y).sum(dim=2)  # Bx x By
        denominator = (x + y - x * y).sum(dim=2) + 1e-8

        return numerator / denominator


class RuzickaKernel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Bx x 1 x D
        y = y.unsqueeze(0)  # 1 x By x D

        numerator = torch.min(x, y).sum(dim=2)  # Bx x By
        denominator = torch.max(x, y).sum(dim=2) + 1e-8

        return numerator / denominator


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self.kernel(x, x)
        k_yy = self.kernel(y, y)
        k_xy = self.kernel(x, y)

        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Literal["emd", "sinkhorn"] = "emd",
    reg: float = 0.05,
    power: int = 2,
) -> float:
    assert power == 1 or power == 2

    if method == "emd" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0], type_as=x0), ot.unif(x1.shape[0], type_as=x1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M, numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def scale_counts_log1p_cpm(counts: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
    """log1p( counts / library * 1e4 ), broadcast-safe (matches training generation_eval)."""
    lib = library_size
    if lib.dim() == 1:
        lib = lib.unsqueeze(-1)
    c = counts.float()
    return torch.log1p((c / lib.clamp(min=1e-8)) * 10_000.0)


def frechet_distance_gaussian_pca(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    n_components: int = 64,
    ridge: float = 1e-6,
) -> float:
    """Fréchet distance between Gaussians fit in PCA space (PCA fit on real only).

    Common for single-cell / tabular generative metrics (cf. Fréchet distance in feature space).
    """
    xr = x_real.detach().float().cpu().numpy()
    fr = x_fake.detach().float().cpu().numpy()
    d = xr.shape[1]
    n_r, n_f = xr.shape[0], fr.shape[0]
    n_comp = int(min(n_components, n_r - 1, n_f - 1, d))
    n_comp = max(n_comp, 1)

    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(xr)
    r = pca.transform(xr)
    f = pca.transform(fr)

    mu1, mu2 = r.mean(0), f.mean(0)
    if n_comp == 1:
        cov1 = np.array([[np.var(r, ddof=1)]])
        cov2 = np.array([[np.var(f, ddof=1)]])
    else:
        cov1 = np.cov(r, rowvar=False)
        cov2 = np.cov(f, rowvar=False)
        if cov1.ndim == 0:
            cov1 = np.array([[float(cov1)]])
        if cov2.ndim == 0:
            cov2 = np.array([[float(cov2)]])

    cov1 = cov1 + np.eye(n_comp) * ridge
    cov2 = cov2 + np.eye(n_comp) * ridge

    diff = mu1 - mu2
    s1_sqrt = linalg.sqrtm(cov1)
    if np.iscomplexobj(s1_sqrt):
        s1_sqrt = np.real(s1_sqrt)
    product = s1_sqrt @ cov2 @ s1_sqrt
    product = (product + product.T) / 2.0
    covmean = linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    tr = np.trace(cov1 + cov2 - 2.0 * covmean)
    return float(np.sum(diff**2) + tr)


def mmd2_rbf(x: torch.Tensor, y: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Squared MMD with RBF kernel (same construction as training MMD_METRICS)."""
    return MMDLoss(kernel=RBFKernel(scale=scale))(x, y)


def _pearson_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pearson correlation for two 1D tensors, returns NaN if variance is zero."""
    xx = x.float().reshape(-1)
    yy = y.float().reshape(-1)
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy))
    if torch.isnan(denom) or denom <= eps:
        return torch.tensor(float("nan"), device=xx.device)
    return torch.sum(xx * yy) / denom


def gears_per_condition_metrics(
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    ctrl_mean: torch.Tensor,
    top_k: int = 20,
) -> dict[str, float]:
    """Compute GEARS-style per-condition pseudo-bulk metrics.

    Metrics are computed on log1p-CPM normalised expression:
      - pearson_all: Pearson(pred_mean, true_mean)
      - pearson_top_de: Pearson on top-k DE genes by |true_mean - ctrl_mean|
      - mse_all: MSE(pred_mean, true_mean)
      - mse_top_de: MSE on top-k DE genes
    """
    if pred_counts.ndim != 2 or true_counts.ndim != 2:
        raise ValueError("pred_counts and true_counts must both be 2D tensors of shape (N, G).")
    if pred_counts.shape != true_counts.shape:
        raise ValueError("pred_counts and true_counts must have the same shape.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    n_cells, n_genes = pred_counts.shape
    pred_lib = pred_counts.sum(dim=1, keepdim=True)
    true_lib = true_counts.sum(dim=1, keepdim=True)
    pred_norm = scale_counts_log1p_cpm(pred_counts, pred_lib)
    true_norm = scale_counts_log1p_cpm(true_counts, true_lib)

    pred_mean = pred_norm.mean(dim=0)
    true_mean = true_norm.mean(dim=0)

    ctrl = ctrl_mean.float().reshape(-1).to(device=true_mean.device)
    if ctrl.numel() != n_genes:
        raise ValueError(f"ctrl_mean must have shape ({n_genes},), but got ({ctrl.numel()},).")

    top_k_eff = min(int(top_k), int(n_genes))
    top_idx = torch.topk(torch.abs(true_mean - ctrl), k=top_k_eff, largest=True).indices

    pearson_all = _pearson_1d(pred_mean, true_mean)
    pearson_top = _pearson_1d(pred_mean[top_idx], true_mean[top_idx])
    mse_all = torch.mean((pred_mean - true_mean) ** 2)
    mse_top = torch.mean((pred_mean[top_idx] - true_mean[top_idx]) ** 2)

    return {
        "pearson_all": float(pearson_all.detach().cpu()),
        "pearson_top_de": float(pearson_top.detach().cpu()),
        "mse_all": float(mse_all.detach().cpu()),
        "mse_top_de": float(mse_top.detach().cpu()),
        "n_cells": float(n_cells),
    }


def gears_aggregate(per_condition_metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average GEARS per-condition metrics over conditions."""
    if len(per_condition_metrics) == 0:
        return {
            "pearson_all": float("nan"),
            "pearson_top_de": float("nan"),
            "mse_all": float("nan"),
            "mse_top_de": float("nan"),
            "n_conditions": 0.0,
        }

    keys = ("pearson_all", "pearson_top_de", "mse_all", "mse_top_de")
    out: dict[str, float] = {}
    for key in keys:
        vals = [m.get(key, float("nan")) for m in per_condition_metrics]
        vals_arr = np.asarray(vals, dtype=np.float64)
        out[key] = float(np.nanmean(vals_arr))
    out["n_conditions"] = float(len(per_condition_metrics))
    return out
