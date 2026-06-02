#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# KV Cache Per-Token Per-Head Dynamic Quantization
#

from typing import Tuple

import torch


def per_token_per_head_quant(
    x: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token per-head symmetric int8 quantization (pure torch native).

    For each (token, head) vector along the head_size dim:
        scale = absmax(x[t, h, :])               # shape [num_tokens, num_kv_heads]
        x_int8 = round(x / (scale / 127.0))      # clamp to [-127, 127]

    Args:
        x: Input tensor [num_tokens, num_kv_heads, head_size], bf16/fp16

    Returns:
        x_int8: [num_tokens, num_kv_heads, head_size], dtype=int8
        scales: [num_tokens, num_kv_heads], dtype=float32 (absmax)
    """
    x_f32 = x.to(torch.float32)
    # absmax over head_size: [num_tokens, num_kv_heads]
    scales = x_f32.abs().amax(dim=-1)
    # avoid div-by-zero for all-zero rows
    safe = scales.clamp(min=1e-8).unsqueeze(-1) / 127.0
    x_int8 = torch.round(x_f32 / safe).clamp_(-127, 127).to(torch.int8)
    return x_int8, scales


def per_token_per_head_dequant(
    x_int8: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    scales: torch.Tensor,  # [num_tokens, num_kv_heads]
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-token per-head dequantization (pure torch native):
        x_fp = x_int8 * (absmax / 127.0)

    Args:
        x_int8: Quantized tensor [num_tokens, num_kv_heads, head_size], int8
        scales: Per-token per-head absmax [num_tokens, num_kv_heads], float32
        out_dtype: Output dtype (bf16 or fp16)

    Returns:
        x_dequant: [num_tokens, num_kv_heads, head_size], dtype=out_dtype
    """
    if x_int8.numel() == 0:
        return torch.empty_like(x_int8, dtype=out_dtype)

    scale = scales.to(torch.float32).unsqueeze(-1) / 127.0
    return (x_int8.to(torch.float32) * scale).to(out_dtype)


def reconstruction_check(
    name: str,
    orig: torch.Tensor,         # [num_tokens, num_kv_heads, head_size], bf16/fp16
    x_int8: torch.Tensor,       # same leading shape, int8
    scales: torch.Tensor,       # [num_tokens, num_kv_heads], float32
    log_fn=None,
) -> dict:
    """
    Compare original BF16 KV against dequantized BF16 KV.

    Reports:
      - cosine similarity (mean / min) over per-(token, head) vectors
      - absolute diff (max / mean)
      - relative diff (max / mean), normalized by per-row absmax

    Returns a dict of metrics; also logs a one-line summary if log_fn is given.
    """
    if orig.numel() == 0:
        return {}

    deq = per_token_per_head_dequant(x_int8, scales, out_dtype=orig.dtype)

    a = orig.to(torch.float32)
    b = deq.to(torch.float32)

    # cosine similarity per (token, head) vector along head_size
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)
    diff = (a - b).abs()
    denom = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    rel = diff / denom

    metrics = {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "abs_max": diff.max().item(),
        "abs_mean": diff.mean().item(),
        "rel_max": rel.max().item(),
        "rel_mean": rel.mean().item(),
        "n_vec": int(cos.numel()),
    }
    if log_fn is not None:
        log_fn(
            f"[KV-quant recon][{name}] n={metrics['n_vec']} "
            f"cos(mean/min)={metrics['cos_mean']:.6f}/{metrics['cos_min']:.6f} "
            f"abs(max/mean)={metrics['abs_max']:.4e}/{metrics['abs_mean']:.4e} "
            f"rel(max/mean)={metrics['rel_max']:.4e}/{metrics['rel_mean']:.4e}"
        )
    return metrics


def reshape_and_cache_int8_with_scales(
    key_int8: torch.Tensor,  # [num_tokens, num_kv_heads, head_size], int8
    value_int8: torch.Tensor,  # [num_tokens, num_kv_heads, head_size], int8
    k_scales: torch.Tensor,  # [num_tokens, num_kv_heads], float32
    v_scales: torch.Tensor,  # [num_tokens, num_kv_heads], float32
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size], int8
    value_cache: torch.Tensor,  # same shape
    kv_cache_scale: torch.Tensor,  # [2, num_blocks, num_kv_heads, block_size], float32
    slot_mapping: torch.Tensor,  # [num_tokens], int32 or int64
    block_size: int,
) -> None:
    """
    Fused write of pre-quantized int8 KV data AND their per-token per-head scales
    into paged cache buffers using torch native ops.

    Writes:
      key_cache[block_idx, :, offset, :] = key_int8[t, :, :]
      value_cache[block_idx, :, offset, :] = value_int8[t, :, :]
      kv_cache_scale[0, block_idx, :, offset] = k_scales[t, :]
      kv_cache_scale[1, block_idx, :, offset] = v_scales[t, :]

    Where: block_idx = slot_mapping[t] // block_size, offset = slot_mapping[t] % block_size

    NOTE: kunlun_ops.reshape_and_cache doesn't support writing pre-quantized int8
    directly (it requires bf16 input + k_max/v_max for internal quantization).
    This function uses torch native indexing as a workaround.

    Future optimization points:
      1. Custom kernel: fuse quant + scatter write + scale write in one kernel launch
      2. Extend reshape_and_cache with quant_mode=2 to accept pre-quantized int8 + scales
    """
    slots = slot_mapping.long()
    block_indices = slots // block_size  # [num_tokens]
    offsets = slots % block_size  # [num_tokens]

    num_blocks = key_cache.shape[0]

    valid = (block_indices >= 0) & (block_indices < num_blocks)
    if not torch.any(valid):
        return

    block_indices = block_indices[valid]
    offsets = offsets[valid]
    key_int8 = key_int8[valid]
    value_int8 = value_int8[valid]
    k_scales = k_scales[valid]
    v_scales = v_scales[valid]

    key_cache[block_indices, :, offsets, :] = key_int8
    value_cache[block_indices, :, offsets, :] = value_int8
    kv_cache_scale[0, block_indices, :, offsets] = k_scales
    kv_cache_scale[1, block_indices, :, offsets] = v_scales


def read_scale_from_cache(
    kv_cache_scale: torch.Tensor,  # [2, num_blocks, num_kv_heads, block_size]
    block_tables: torch.Tensor,  # [batch_size, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,  # [batch_size]
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read per-token per-head scales from scale buffer for decode attention.

    For each sequence, reads scales for all cached tokens based on block_tables.

    Returns:
        k_scales: [batch_size, max_seq_len, num_kv_heads], float32
        v_scales: same shape
    """
    batch_size = block_tables.shape[0]
    num_kv_heads = kv_cache_scale.shape[2]
    max_seq_len = seq_lens.max().item()

    # Allocate output: [batch_size, max_seq_len, num_kv_heads]
    k_scales = torch.zeros(
        batch_size, max_seq_len, num_kv_heads,
        dtype=kv_cache_scale.dtype, device=kv_cache_scale.device
    )
    v_scales = torch.zeros(
        batch_size, max_seq_len, num_kv_heads,
        dtype=kv_cache_scale.dtype, device=kv_cache_scale.device
    )

    # For each sequence, gather scales from blocks
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        num_blocks_used = (seq_len + block_size - 1) // block_size
        for blk_idx in range(num_blocks_used):
            physical_block = block_tables[b, blk_idx].item()
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, seq_len)
            tokens_in_block = end_pos - start_pos

            # kv_cache_scale[:, physical_block, :, :tokens_in_block]
            # shape: [num_kv_heads, tokens_in_block] -> transpose to [tokens_in_block, num_kv_heads]
            k_scales[b, start_pos:end_pos, :] = (
                kv_cache_scale[0, physical_block, :, :tokens_in_block].T
            )
            v_scales[b, start_pos:end_pos, :] = (
                kv_cache_scale[1, physical_block, :, :tokens_in_block].T
            )

    return k_scales, v_scales
