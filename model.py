from __future__ import annotations

import math
from pathlib import Path
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from scBERT_tokenizer import get_pretrained

def exists(x):
    return x is not None


def default(x, d):
    return x if exists(x) else d


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def empty(x):
    return x.numel() == 0


def get_module_device(module: nn.Module):
    return next(module.parameters()).device


def find_modules(module: nn.Module, module_type):
    return [m for m in module.modules() if isinstance(m, module_type)]


def orthogonal_matrix_chunk(cols, device=None):
    block = torch.randn((cols, cols), device=device)
    q, _ = torch.linalg.qr(block.cpu(), mode="reduced")
    return q.to(device).t()


def gaussian_orthogonal_random_matrix(
    nb_rows,
    nb_columns,
    scaling=0,
    device=None,
):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        block_list.append(orthogonal_matrix_chunk(nb_columns, device=device))

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        block_list.append(
            orthogonal_matrix_chunk(nb_columns, device=device)[:remaining_rows]
        )

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(
    data,
    *,
    projection_matrix,
    is_query,
    normalize_data=True,
    eps=1e-4,
    device=None,
):
    b, h, *_ = data.shape

    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5

    projection = projection_matrix[None, None, :, :].expand(b, h, -1, -1)
    projection = projection.type_as(data)

    data_dash = torch.einsum(
        "...id,...jd->...ij",
        data_normalizer * data,
        projection,
    )

    diag_data = (data ** 2).sum(dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True).values
            )
            + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps
        )

    return data_dash.type_as(data)


# def linear_attention(q, k, v):
#     k_cumsum = k.sum(dim=-2)
#     d_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))
#     context = torch.einsum("...nd,...ne->...de", k, v)
#     out = torch.einsum("...de,...nd,...n->...ne", context, q, d_inv)
#     return out
def linear_attention(q, k, v, eps=1e-6):
    # q, k, v: [B, H, N, D]

    k_sum = k.sum(dim=-2)  # [B, H, D]

    denom = torch.matmul(
        q,
        k_sum.unsqueeze(-1),
    ).squeeze(-1)  # [B, H, N]

    denom = denom.clamp_min(eps)

    context = torch.matmul(
        k.transpose(-2, -1),
        v,
    )  # [B, H, D, D]

    out = torch.matmul(
        q,
        context,
    )  # [B, H, N, D]

    out = out / denom.unsqueeze(-1)

    return out


class PerformerAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        no_projection=False,
    ):
        super().__init__()

        if causal:
            raise NotImplementedError("nano-scBERT only supports non-causal Performer")
        if generalized_attention:
            raise NotImplementedError("generalized_attention not used in scBERT MVP")
        if no_projection:
            raise NotImplementedError("no_projection not used in scBERT MVP")

        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
        )

        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projection = self.create_projection(device=device)
        self.projection_matrix.copy_(projection)

    def forward(self, q, k, v, output_attentions=False):
        create_kernel = partial(
            softmax_kernel,
            projection_matrix=self.projection_matrix,
            device=q.device,
        )

        q = create_kernel(q, is_query=True)
        k = create_kernel(k, is_query=False)

        out = linear_attention(q, k, v)

        if output_attentions:
            raise NotImplementedError("output_attentions removed from nano-scBERT MVP")

        return out


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()

        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.0,
        no_projection=False,
        qkv_bias=False,
    ):
        super().__init__()

        if local_heads != 0:
            raise NotImplementedError("local attention is not used in scBERT checkpoint")
        if causal:
            raise NotImplementedError("scBERT uses non-causal attention")

        assert dim % heads == 0, "dimension must be divisible by heads"

        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads

        self.fast_attention = PerformerAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=no_projection,
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = None

        # self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        # self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        # self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        if exists(context):
            raise NotImplementedError("cross attention is not used in scBERT")

        b, n, _ = x.shape
        h = self.heads

        # q = self.to_q(x)
        # k = self.to_k(x)
        # v = self.to_v(x)

        # q = q.view(b, n, h, -1).transpose(1, 2)
        # k = k.view(b, n, h, -1).transpose(1, 2)
        # v = v.view(b, n, h, -1).transpose(1, 2)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, n, h, -1).transpose(1, 2)
        v = v.view(b, n, h, -1).transpose(1, 2)

        if exists(context_mask):
            global_mask = context_mask[:, None, :, None]
            v.masked_fill_(~global_mask, 0.0)

        out = self.fast_attention(q, k, v, output_attentions=output_attentions)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out = self.to_out(out)
        return self.dropout(out)


class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len,
        gene2vec_path: Optional[Union[str, Path]] = None,
        gene2vec_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if gene2vec_weight is None:
            if gene2vec_path is None:
                weight = torch.zeros(max_seq_len, dim)
            else:
                arr = np.load(str(gene2vec_path))
                arr = np.concatenate(
                    [arr, np.zeros((1, arr.shape[1]), dtype=arr.dtype)],
                    axis=0,
                )
                weight = torch.from_numpy(arr).float()
        else:
            weight = gene2vec_weight.float()

        if weight.shape[0] < max_seq_len:
            raise ValueError(f"gene2vec rows {weight.shape[0]} < {max_seq_len}")
        if weight.shape[1] != dim:
            raise ValueError(f"gene2vec dim {weight.shape[1]} != {dim}")

        self.emb = nn.Embedding.from_pretrained(weight[:max_seq_len], freeze=False)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class SequentialSequence(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for f, g in self.layers:
            x = x + f(x)
            x = x + g(x)

        return x


class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.0,
        attn_dropout=0.0,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
    ):
        super().__init__()

        if reversible:
            raise NotImplementedError("reversible layers are not used in scBERT MVP")
        if cross_attend:
            raise NotImplementedError("cross attention is not used in scBERT")
        if use_scalenorm or use_rezero:
            raise NotImplementedError("scBERT checkpoint uses PreLayerNorm")

        layers = nn.ModuleList([])

        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = (
            local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        )

        assert len(local_attn_heads) == depth

        wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(
                nn.ModuleList(
                    [
                        wrapper_fn(
                            SelfAttention(
                                dim,
                                causal=causal,
                                heads=heads,
                                dim_head=dim_head,
                                local_heads=local_heads,
                                local_window_size=local_window_size,
                                nb_features=nb_features,
                                generalized_attention=generalized_attention,
                                kernel_fn=kernel_fn,
                                dropout=attn_dropout,
                                no_projection=no_projection,
                                qkv_bias=qkv_bias,
                            )
                        ),
                        wrapper_fn(
                          FeedForward(
                              dim,
                              mult=ff_mult,
                              dropout=ff_dropout,
                              glu=ff_glu,
                          )
                      ),
                    ]
                )
            )

        self.net = SequentialSequence(layers)

        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if (
            exists(self.feature_redraw_interval)
            and self.calls_since_last_redraw >= self.feature_redraw_interval
        ):
            device = get_module_device(self)

            for fast_attention in find_modules(self, PerformerAttention):
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions=False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()

        return self.net(x)


class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head=64,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False, #kept for future training-from-scratch support; 
                          #current pretrained scBERT checkpoint does not use reversible layers
        ff_chunks=1,
        ff_glu=False,
        emb_dropout=0.0,
        ff_dropout=0.0,
        attn_dropout=0.0,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        cross_attend=False,
        no_projection=False,
        tie_embed=False,
        g2v_position_emb=True,
        auto_check_redraw=True,
        qkv_bias=False,
        gene2vec_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(
                dim,
                max_seq_len,
                gene2vec_path=gene2vec_path,
            )
        else:
            self.pos_emb = torch.zeros_like

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(
            dim,
            depth,
            heads,
            dim_head,
            local_attn_heads,
            local_window_size,
            causal,
            ff_mult,
            nb_features,
            feature_redraw_interval,
            reversible,
            ff_chunks,
            generalized_attention,
            kernel_fn,
            use_scalenorm,
            use_rezero,
            ff_glu,
            ff_dropout,
            attn_dropout,
            cross_attend,
            no_projection,
            auto_check_redraw,
            qkv_bias,
        )

        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def optimize_for_inference(self, compile_model: bool = True):
        for p in self.parameters():
            p.requires_grad_(False)
    
        if compile_model and next(self.parameters()).device.type == "cuda":
            return torch.compile(self)
    
        return self

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, output_attentions=False, **kwargs):
        _, n = x.shape
        assert n <= self.max_seq_len, (
            f"sequence length {n} must be less than max_seq_len {self.max_seq_len}"
        )

        x = self.token_emb(x.long())

        if output_attentions:
            x.requires_grad_()

        x = x + self.pos_emb(x)
        x = self.dropout(x)

        x = self.performer(x)

        x = self.norm(x)

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()

    @torch.inference_mode()
    def encode(
        self,
        x,
        *,
        batch_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        append_cls: bool = True,
        pool: str = "gene_mean",
        use_amp: bool = True,
        compile_model: bool = False,
        verbose: bool = False,
    ):
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
            self.to(device)
    
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.long)
    
        if append_cls and x.shape[1] == self.max_seq_len - 1:
            zero = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            x = torch.cat([x, zero], dim=1)
    
        if compile_model and device.type == "cuda":
            model = torch.compile(self)
        else:
            model = self
    
        outs = []
    
        autocast_enabled = use_amp and device.type == "cuda"
    
        if batch_size is None:
            xb = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                h = model.forward(xb, return_encodings=True)
                outs.append(_pool_encodings(h, pool).cpu())
        else:
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                if verbose:
                    print(f"Encoding cells {start:,}:{end:,} / {x.shape[0]:,}")
    
                xb = x[start:end].to(device, non_blocking=True)
    
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    h = model.forward(xb, return_encodings=True)
                    out = _pool_encodings(h, pool)
    
                outs.append(out.cpu())
    
        out = torch.cat(outs, dim=0)
    
        return out

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        map_location="cpu",
        strict=True,
        **kwargs,
    ):
        model = cls(**kwargs)
    
        ckpt = torch.load(str(checkpoint_path), map_location=map_location)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        state = _strip_module_prefix(state)
    
        missing, unexpected = model.load_state_dict(state, strict=strict)
    
        if not strict:
            print(f"[nano-scBERT] missing keys: {len(missing)}")
            print(f"[nano-scBERT] unexpected keys: {len(unexpected)}")
    
        return model    

    @classmethod
    def from_pretrained_name(
        cls,
        name: str = "scbert-human-panglao",
        *,
        map_location="cpu",
        strict=True,
    ):
        info = get_pretrained(name)
    
        return cls.from_pretrained(
            info["checkpoint"],
            map_location=map_location,
            strict=strict,
            **info["config"],
        )

def _pool_encodings(h, pool: str):
    if pool == "mean":
        return h.mean(dim=1)
    if pool == "cls":
        return h[:, -1]
    if pool == "gene_mean":
        return h[:, :-1].mean(dim=1)

    raise ValueError(f"unknown pool: {pool}")


def _strip_module_prefix(state_dict):
    out = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v

    out = _convert_qkv_state_dict(out)
    return out


def _convert_qkv_state_dict(state_dict):
    out = dict(state_dict)
    prefixes = set()

    for k in list(out.keys()):
        if k.endswith(".to_q.weight"):
            prefixes.add(k[: -len(".to_q.weight")])

    for prefix in prefixes:
        out[prefix + ".to_qkv.weight"] = torch.cat(
            [
                out.pop(prefix + ".to_q.weight"),
                out.pop(prefix + ".to_k.weight"),
                out.pop(prefix + ".to_v.weight"),
            ],
            dim=0,
        )

        q_b = prefix + ".to_q.bias"
        k_b = prefix + ".to_k.bias"
        v_b = prefix + ".to_v.bias"

        if q_b in out:
            out[prefix + ".to_qkv.bias"] = torch.cat(
                [
                    out.pop(q_b),
                    out.pop(k_b),
                    out.pop(v_b),
                ],
                dim=0,
            )

    return out