import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import XfmrWeights, LayerWeights
from entropix.torch_stats import AttnStats

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

from typing import Tuple, Optional

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps)).to(device) 

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2).to(device) 
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2).to(device) 
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1]).to(device) 
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1]).to(device) 
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2).to(device) 
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2).to(device) 
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1).to(device) 
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1).to(device) 
    return xq_out.to(dtype=dtype, device=device), xk_out.to(dtype=dtype, device=device)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

    x = x.to(device)
    layer_weights = layer_weights.to(device)
    attn_mask = attn_mask.to(device) if attn_mask is not None else None

    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3)).to(device)   # (bs, n_heads, seqlen, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1)).to(device)   # (bs, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(values, (0, 2, 1, 3)).to(device)  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.to(device)).to(device) 
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(dtype=torch.float32, device=device)  # Always do attention softmax at float32
    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE).to(device) 
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE).to(device) 
    scores = F.softmax(padded_logits, dim=-1).to(device=device, dtype=torch.float32)
    output = torch.matmul(scores, values).to(device) 
    output = output.transpose(1, 2).reshape(xq.shape[0], xq.shape[2], -1).to(device) 
    out = F.linear(output, layer_weights.wo).to(device) 
    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2).to(device) 

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    xfmr_weights = xfmr_weights.to(device)
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm).to(device) 
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:,:,-1,:], i)
        h = h + h_attn.to(device) 
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i]).to(device) 
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output).to(device) 
    return logits, kvcache, scores, attn_stats