from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

PRETRAINED_MODELS = {
    "scbert-human-panglao": {
        "checkpoint": PACKAGE_ROOT
        / "assets"
        / "scbert-human-panglao"
        / "panglao_pretrain_nano.pth",

        "gene_order": PACKAGE_ROOT
        / "assets"
        / "scbert-human-panglao"
        / "gene_order.txt",

        "config": {
            "num_tokens": 7,
            "dim": 200,
            "depth": 6,
            "max_seq_len": 16907,
            "heads": 10,
            "local_attn_heads": 0,
            "g2v_position_emb": True,
        },
    },
}


def get_pretrained(name: str):
    if name not in PRETRAINED_MODELS:
        available = ", ".join(PRETRAINED_MODELS.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {available}"
        )

    return PRETRAINED_MODELS[name]

