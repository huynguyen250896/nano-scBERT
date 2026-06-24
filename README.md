# nano-scBERT

A minimal, fast, and faithful reimplementation of [scBERT](https://github.com/TencentAILabHealthcare/scBERT) for single-cell foundation model inference, with upcoming support for fine-tuning and training from scratch. 

nano-scBERT is designed to make scBERT easier to understand, modify, and run while preserving the behavior of the original model.

![figure1](assets/umap_nano_scbert_vs_scbert.png)

## Why nano-scBERT
Single-cell foundation models are one of the most promising directions in AI for biology, but many existing repositories remain difficult to read, extend, benchmark, or use as educational resources.

nano-scBERT aims to provide:
- A clean and minimal implementation
- Faithful reproduction of the original scBERT architecture
- Faster inference with modern PyTorch optimizations
- A codebase suitable for experimentation, fine-tuning, and future training from scratch

## Benchmark 
I carefully benchmarked nano-scBERT across different settings to give future users confidence in adopting nano-scBERT as a drop-in alternative to the official implementation. Full benchmark details are available in [benchmark_scbert_vs_nano.ipynb](benchmark_scbert_vs_nano.ipynb).

#### Inference Runtime
nano-scBERT achieves roughly **2.2× faster inference** than the original implementation.

| Model      | Total (4,146 cells) | Per cell | Throughput  | Speedup |
| ---------- | -------------------- | -------- | ----------- | ------- |
| nano-scBERT | **53.71 s**         | **12.955 ms**  | **77.19 cells/s** | **2.2×**   |
| scBERT      | 132.64 s            | 31.992 ms| 31.25 cells/s | 1.00×   |

#### Cell-level Embedding Reproducibility
nano-scBERT reproduces the original scBERT embedding space almost exactly, preserving both local and global structure.

![figure2](assets/umap_overlay_nano_scbert_vs_scbert.png)

| Metric | Value |
|----------|----------:|
| Mean cosine similarity | **1.0000** |
| Median cosine similarity | **1.0000** |
| Minimum cosine similarity | **0.9999998** |
| Mean absolute difference | **2.74e-06** |
| Distance correlation | **0.9999998** |

The PCA spectrum and pairwise distance structure are nearly identical between nano-scBERT and the original implementation.

> Benchmarked on a single NVIDIA A100 (80 GB) GPU with batch size 64 on the Pancreas dataset (4,146 cells).

## Install
```bash
git clone https://github.com/huynguyen250896/nano-scBERT.git
cd nano-scBERT

pip install -r requirements.txt
```

## Quick Start

## Task: Generate Cell Embeddings from Raw-count `.h5ad`
```sh
python tasks/embedding.py \
    --input pancreas.h5ad \
    --output nano_scbert_embeddings.npy \
    --mode raw
```

## Task: Generate Cell Embeddings from Preprocessed `.h5ad`
```sh
python tasks/embedding.py \
    --input pancreas_scbert_preprocessed.h5ad \
    --output nano_scbert_embeddings.npy \
    --mode preprocessed
```

## Roadmap
- [X] Embedding .h5ad scRNA data
- [ ] Finetuning
- [ ] Training from scratch

Let me know what tasks you'd like to see next!

## Acknowledgments
1. If you find this repo interesting and/or use nano-scBERT in your work, please cite the original paper:
>Yang, F., Wang, W., Wang, F. et al. scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. Nat Mach Intell (2022). https://doi.org/10.1038/s42256-022-00534-z

and STAR⭐ my repo. Thanks!

2. nano-scBERT is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanogpt), Chris Hayduk's [minAlphaFold2](https://github.com/ChrisHayduk/minAlphaFold2), and especially Danqi Liao's [nano-scGPT](https://github.com/Danqi7/nano-scGPT).

## License
[MIT LICENSE](LICENSE)