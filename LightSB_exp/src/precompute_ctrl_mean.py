"""Precompute control mean expression for GEARS-style evaluation.

This script computes mean log1p-CPM expression over control cells and saves
it as a 1D numpy array (shape: [n_genes]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute control mean log1p-CPM vector from an h5ad file."
    )
    parser.add_argument("--adata_path", type=Path, required=True, help="Input .h5ad path.")
    parser.add_argument(
        "--control_label_key",
        type=str,
        required=True,
        help="obs column containing perturbation / condition labels.",
    )
    parser.add_argument(
        "--control_label_value",
        type=str,
        required=True,
        help="obs value used to identify control cells.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Output .npy path for control mean vector.",
    )
    parser.add_argument(
        "--adata_attr",
        type=str,
        default="X",
        help='AnnData attribute that stores counts ("X" or "layers").',
    )
    parser.add_argument(
        "--adata_key",
        type=str,
        default=None,
        help='Key under `layers` when --adata_attr is "layers".',
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Number of rows processed per chunk.",
    )
    return parser.parse_args()


def _get_matrix(adata: ad.AnnData, adata_attr: str, adata_key: str | None):
    if adata_attr == "X":
        return adata.X
    mat = getattr(adata, adata_attr)
    if adata_key is None:
        raise ValueError("`--adata_key` is required when --adata_attr is not X.")
    return mat[adata_key]


def _chunked_mean_log1p_cpm(x, chunk_size: int) -> np.ndarray:
    n_cells, n_genes = x.shape
    if n_cells == 0:
        raise ValueError("No control cells found; cannot compute control mean.")

    if sp.issparse(x):
        x = x.tocsr()
        lib = np.asarray(x.sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        lib = np.asarray(x.sum(axis=1)).reshape(-1).astype(np.float64)
    lib = np.clip(lib, 1e-8, None)

    running_sum = np.zeros((n_genes,), dtype=np.float64)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        scales = (10000.0 / lib[start:end]).astype(np.float64)
        block = x[start:end]
        if sp.issparse(block):
            block = block.multiply(scales[:, None]).toarray().astype(np.float64, copy=False)
        else:
            block = np.asarray(block, dtype=np.float64) * scales[:, None]
        np.log1p(block, out=block)
        running_sum += block.sum(axis=0)
    return (running_sum / float(n_cells)).astype(np.float32)


def main() -> None:
    args = _parse_args()
    adata = ad.read_h5ad(args.adata_path)

    if args.control_label_key not in adata.obs.columns:
        raise KeyError(f"obs key not found: {args.control_label_key}")

    ctrl_mask = adata.obs[args.control_label_key] == args.control_label_value
    ctrl = adata[ctrl_mask]

    x_ctrl = _get_matrix(ctrl, args.adata_attr, args.adata_key)
    ctrl_mean = _chunked_mean_log1p_cpm(x_ctrl, chunk_size=args.chunk_size)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, ctrl_mean)
    print(f"[precompute_ctrl_mean] saved: {args.out_path}")
    print(f"[precompute_ctrl_mean] shape: {ctrl_mean.shape}, n_control_cells: {ctrl.shape[0]}")


if __name__ == "__main__":
    main()
