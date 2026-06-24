from __future__ import annotations

from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def load_reference_genes(path: Union[str, Path]) -> list[str]:
    ref = sc.read_h5ad(str(path))
    return ref.var_names.astype(str).tolist()


def load_gene_file(path: Union[str, Path]) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def save_gene_file(
    genes: list[str],
    path: Union[str, Path],
):
    with open(path, "w") as f:
        for gene in genes:
            f.write(gene + "\n")


def align_to_reference(
    adata: ad.AnnData,
    reference_genes: list[str],
    *,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Align query genes to Panglao/scBERT gene space.

    Missing genes -> filled with 0
    Extra genes   -> dropped
    """

    src_genes = adata.var_names.astype(str).tolist()
    src_lookup = {g: i for i, g in enumerate(src_genes)}

    rows = adata.n_obs
    cols = len(reference_genes)

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()

    row_idx = []
    col_idx = []
    values = []
    matched = 0

    for target_col, gene in enumerate(reference_genes):
        source_col = src_lookup.get(gene)

        if source_col is None:
            continue

        matched += 1

        col = X[:, source_col]

        if sparse.issparse(col):
            col = col.toarray().reshape(-1)
        else:
            col = np.asarray(col).reshape(-1)

        nz = col != 0

        if np.any(nz):
            row_idx.append(np.where(nz)[0])
            col_idx.append(np.full(np.sum(nz), target_col, dtype=np.int64))
            values.append(col[nz])

    if verbose:
        print(
            f"Original genes: {len(src_genes)} | "
            f"Reference genes: {len(reference_genes)} | "
            f"Matched genes: {matched:,} "
            f"({matched / len(reference_genes) * 100:.1f}%)"
        )

    if len(values) > 0:
        aligned = sparse.coo_matrix(
            (
                np.concatenate(values),
                (
                    np.concatenate(row_idx),
                    np.concatenate(col_idx),
                ),
            ),
            shape=(rows, cols),
            dtype=np.float32,
        ).tocsr()
    else:
        aligned = sparse.csr_matrix((rows, cols), dtype=np.float32)

    out = ad.AnnData(X=aligned)
    out.var_names = reference_genes
    out.obs_names = adata.obs_names.copy()
    out.obs = adata.obs.copy()

    return out


def preprocess_adata(
    adata: ad.AnnData,
    reference_genes: list[str],
    *,
    filter_cells: bool = True,
    min_genes: int = 200,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    log1p: bool = True,
    log_base: float = 2,
    verbose=True,
) -> ad.AnnData:
    """
    scBERT preprocessing.

    Based on official preprocess.py:

        align genes
        filter_cells(min_genes=200)
        normalize_total(target_sum=1e4)
        log1p(base=2)
    """

    out = align_to_reference(
        adata,
        reference_genes,
    )

    if filter_cells:
      sc.pp.filter_cells(
          out,
          min_genes=min_genes,
      )

      print(
          f"Cells after filtering: {out.n_obs:,} | "
          f"Genes: {out.n_vars:,}"
      )

    if normalize_total:
        sc.pp.normalize_total(
            out,
            target_sum=target_sum,
        )

    if log1p:
        sc.pp.log1p(
            out,
            base=log_base,
        )

    return out