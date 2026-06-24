from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np
import scanpy as sc
from scipy import sparse

from preprocessing import (
    load_reference_genes,
    load_gene_file,
    save_gene_file,
)

from registry import get_pretrained


class scBERTTokenizer:
    """
    scBERT tokenizer.

    Input:
        aligned expression matrix

    Output:
        integer tokens in range [0, 5]

    Notes:
    -----
    Exact Tencent binning was not released.

    Current compatibility mode follows the description of the tokenizer from the scBERT paper,
    its public implementation (https://github.com/TencentAILabHealthcare/scBERT) 
    and my understanding:
        normalize_total
        log1p(base=2)
        astype(int)
        clip(0, 5)
    """

    def __init__(
        self,
        genes: list[str],
        *,
        bin_num: int = 5,
        append_cls: bool = True,
    ):
        self.genes = list(genes)
        self.bin_num = bin_num
        self.append_cls = append_cls

    @classmethod
    def from_panglao_h5ad(
        cls,
        path: Union[str, Path],
        *,
        bin_num: int = 5,
        append_cls: bool = True,
    ):
        return cls(
            load_reference_genes(path),
            bin_num=bin_num,
            append_cls=append_cls,
        )

    @classmethod
    def from_gene_file(
        cls,
        path: Union[str, Path],
        *,
        bin_num: int = 5,
        append_cls: bool = True,
    ):
        return cls(
            load_gene_file(path),
            bin_num=bin_num,
            append_cls=append_cls,
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str = "scbert-human-panglao",
        *,
        bin_num: int = 5,
        append_cls: bool = True,
    ):
        info = get_pretrained(name)
        return cls.from_gene_file(
            info["gene_order"],
            bin_num=bin_num,
            append_cls=append_cls,
        )

    def save_gene_file(
        self,
        path: Union[str, Path],
    ):
        save_gene_file(
            self.genes,
            path,
        )

    def encode(
        self,
        X,
        verbose = True,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        X:
            aligned expression matrix

        Returns
        -------
        tokens:
            shape [N, 16906]
            or [N, 16907] if append_cls=True
        """

        if sparse.issparse(X):
            X = X.toarray()

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        tokens = X.astype(np.int64)

        tokens[tokens < 0] = 0
        tokens[tokens > self.bin_num] = self.bin_num

        if self.append_cls:
            cls_token = np.zeros(
                (tokens.shape[0], 1),
                dtype=np.int64,
            )

            tokens = np.concatenate(
                [tokens, cls_token],
                axis=1,
            )

        if verbose:
          print(
              f"Token range: "
              f"[{tokens.min()}, {tokens.max()}]"
          )

        return tokens.astype(np.int64)

    def encode_adata(
        self,
        adata,
    ) -> np.ndarray:
        return self.encode(adata.X)