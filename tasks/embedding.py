from __future__ import annotations

import argparse
import numpy as np
import scanpy as sc
import torch

from .model import PerformerLM
from .preprocessing import preprocess_adata
from .registry import get_pretrained
from .scBERT_tokenizer import scBERTTokenizer


@torch.no_grad()
def fix_performer_projection_matrices(model, seed=0):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for module in model.modules():
        if module.__class__.__name__ == "FastAttention":
            device = module.projection_matrix.device
            projection = module.create_projection(device=device)
            module.projection_matrix.copy_(projection)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--output",
        required=True,
        help="Path to save the output embeddings.",
    )
    p.add_argument(
        "--model",
        default="scbert-human-panglao",
    )
    p.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "preprocessed"],
        help=(
            "Input format. "
            "'raw': raw AnnData (.h5ad). Run the full scBERT pipeline: "
            "gene alignment -> filter_cells(min_genes=200) -> "
            "normalize_total(1e4) -> log1p(base=2) -> "
            "tokenization -> embedding. "
            "'preprocessed': input has already been processed by the "
            "official scBERT preprocessing pipeline. "
            "Skip preprocessing and run "
            "tokenization -> embedding only."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing the data.",
    )
    p.add_argument(
        "--pool",
        default="gene_mean",
        choices=["mean", "gene_mean", "cls"],
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    p.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile.",
    )
    args = p.parse_args(argv)

    use_amp = not args.no_amp
    use_compile = not args.no_compile

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    info = get_pretrained(args.model)

    print("Loading data...")
    adata = sc.read_h5ad(args.input)
    print(f"Input shape: {adata.shape}")

    tokenizer = scBERTTokenizer.from_pretrained(
        args.model,
        append_cls=True,
    )

    if args.mode == "raw":
        print("Running scBERT preprocessing...")
        adata = preprocess_adata(
            adata,
            tokenizer.genes,
        )
        print(f"Preprocessed shape: {adata.shape}")

    else:
        print("Skipping preprocessing (already preprocessed).")

    print("Tokenizing data...")
    tokens = tokenizer.encode_adata(adata)

    x = torch.from_numpy(tokens).long()

    print("Loading nano-scBERT...")

    model = PerformerLM.from_pretrained(
        info["checkpoint"],
        strict=True,
        **info["config"],
    ).to(device)

    fix_performer_projection_matrices(
        model,
        seed=0,
    )

    model = model.optimize_for_inference(
        compile_model=use_compile,
    )

    print(
        f"use_amp={use_amp}, "
        f"use_compile={use_compile}"
    )

    with torch.inference_mode():
        emb = model.encode(
            x,
            batch_size=args.batch_size,
            device=device,
            append_cls=False,
            pool=args.pool,
            use_amp=use_amp,
            compile_model=False,   # already handled above
        )

    emb = emb.cpu().numpy()

    np.save(
        args.output,
        emb,
    )

    print(
        f"saved {emb.shape} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()