#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# KV Cache Per-Token Per-Head Dynamic Quantization
#

from typing import Tuple

import torch


def per_token_per_head_quant_from_bf16(
    x: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    out_int8: torch.Tensor,  # [num_tokens, num_kv_heads, head_size], int8 (preallocated)
    out_scales: torch.Tensor,  # [num_tokens, num_kv_heads], float32 (preallocated)
) -> None:
    """
    In-place per-token per-head symmetric int8 quantization.

    For each (token, head) vector along the head_size dim:
        scale = absmax(x[t, h, :])               # shape [num_tokens, num_kv_heads]
        x_int8 = round(x / (scale / 127.0))      # clamp to [-127, 127]

    Args:
        x: Input tensor [num_tokens, num_kv_heads, head_size], bf16/fp16
        out_int8: Pre-allocated int8 output [num_tokens, num_kv_heads, head_size]
        out_scales: Pre-allocated float32 output [num_tokens, num_kv_heads]
    """
    x_f32 = x.to(torch.float32)
    # absmax over head_size: [num_tokens, num_kv_heads]
    scales = x_f32.abs().amax(dim=-1)
    # avoid div-by-zero for all-zero rows
    safe = scales.clamp(min=1e-8).unsqueeze(-1) / 127.0
    out_int8.copy_(
        torch.round(x_f32 / safe).clamp_(-127, 127).to(torch.int8)
    )
    out_scales.copy_(scales)


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


# ----------------------------------------------------------------------------
# Plan A (CUDA-Graph friendly) draft kept here as reference. NOT enabled
# because it requires the KV-cache allocator to reserve the LAST physical
# block as a scratch slot — without that allocator change, redirecting
# "invalid" writes into the last block corrupts real data that legitimately
# lives there. Re-enable together with allocator changes that bump
# num_blocks by +1 and hide that last block from the scheduler.
# ----------------------------------------------------------------------------
# def reshape_and_cache_int8_with_scales(
#     key_int8, value_int8,
#     k_scales, v_scales,
#     key_cache, value_cache,
#     kv_cache_scale,
#     slot_mapping,
#     block_size: int,
# ) -> None:
#     num_blocks_total = key_cache.shape[0]
#     scratch_block_idx = num_blocks_total - 1   # reserved
#
#     slots = slot_mapping.long()
#     block_indices = slots // block_size
#     offsets = slots % block_size
#
#     # NOTE: condition must be ``>= num_blocks_total`` to keep the last
#     # *usable* block (== scratch_block_idx when allocator reserves +1) valid.
#     invalid = (slots < 0) | (block_indices >= num_blocks_total)
#     block_indices = torch.where(invalid,
#                                 torch.full_like(block_indices, scratch_block_idx),
#                                 block_indices)
#     offsets = torch.where(invalid,
#                           torch.zeros_like(offsets),
#                           offsets)
#
#     key_cache[block_indices, :, offsets, :] = key_int8
#     value_cache[block_indices, :, offsets, :] = value_int8
#     kv_cache_scale[0, block_indices, :, offsets] = k_scales
#     kv_cache_scale[1, block_indices, :, offsets] = v_scales


def reshape_and_cache_int8_with_scales(
    key: torch.Tensor,            # [num_tokens, num_kv_heads, head_size], bf16/fp16
    value: torch.Tensor,          # [num_tokens, num_kv_heads, head_size], bf16/fp16
    key_cache: torch.Tensor,      # [num_blocks, num_kv_heads, block_size, head_size+scale_pad], int8
    value_cache: torch.Tensor,    # same shape as key_cache
    k_scale_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads], float32 (strided view of kv_cache[0])
    v_scale_cache: torch.Tensor,  # same shape, strided view of kv_cache[1]
    slot_mapping: torch.Tensor,   # [num_tokens], int32 or int64
    block_size: int,
    head_size: int,
) -> None:
    """
    Fused per-token per-head quantization + scatter into paged KV cache.

    Aligned with vLLM Triton attention's
    ``triton_reshape_and_cache_flash_per_token_head_quant``:

    1. Quantize K/V per (token, head) using absmax -> int8 with float32 scale.
    2. Write int8 K/V into the *data* region of the cache:
           key_cache[block, head, slot, 0:head_size]   = key_int8[t, head, :]
           value_cache[block, head, slot, 0:head_size] = value_int8[t, head, :]
    3. Write float32 scale into the *inline padding* region of the cache:
           k_scale_cache[block, slot, head] = k_scale[t, head]
           v_scale_cache[block, slot, head] = v_scale[t, head]

    Where: block = slot_mapping[t] // block_size, slot = slot_mapping[t] % block_size.

    The scale views are produced by ``_ensure_scale_caches`` via ``as_strided``
    over the same storage as the KV cache, so writing through the view is
    equivalent to writing into the inline padding bytes of the KV cache. No
    separate scale buffer is allocated.

    NOTE: Until a fused CUDA / XPU kernel is available, this function runs in
    pure pytorch (quant + scatter + scale scatter). The scatter pattern matches
    what ``reshape_and_cache_int8_with_scales`` did before the refactor.
    """
    num_tokens = key.shape[0]
    if num_tokens == 0:
        return

    slots = slot_mapping.long()
    block_indices = slots // block_size
    offsets = slots % block_size

    num_blocks = key_cache.shape[0]

    valid = (block_indices >= 0) & (block_indices < num_blocks)
    if not torch.any(valid):
        return

    block_indices = block_indices[valid]
    offsets = offsets[valid]
    k_valid = key[valid]
    v_valid = value[valid]

    # Per-token per-head quant (compute scale + int8 from bf16/fp16).
    key_int8 = torch.empty(
        k_valid.shape[0], k_valid.shape[1], k_valid.shape[2],
        dtype=torch.int8, device=k_valid.device,
    )
    val_int8 = torch.empty(
        v_valid.shape[0], v_valid.shape[1], v_valid.shape[2],
        dtype=torch.int8, device=v_valid.device,
    )
    k_scales = torch.empty(
        k_valid.shape[0], k_valid.shape[1],
        dtype=torch.float32, device=k_valid.device,
    )
    v_scales = torch.empty(
        v_valid.shape[0], v_valid.shape[1],
        dtype=torch.float32, device=v_valid.device,
    )
    per_token_per_head_quant_from_bf16(k_valid, key_int8, k_scales)
    per_token_per_head_quant_from_bf16(v_valid, val_int8, v_scales)

    # 1. Write quantized K/V into the *data* region of the cache (head_size
    #    int8 elements at the start of each head row). The remaining
    #    `scale_pad` int8 elements per head row are reserved for the inline
    #    float32 scale and are filled in step 2.
    key_cache[block_indices, :, offsets, :head_size] = key_int8
    value_cache[block_indices, :, offsets, :head_size] = val_int8

    # 2. Write the float32 scale into the inline padding. The scale views
    #    are strided float32 tensors sharing the same storage as the KV
    #    cache (see _ensure_scale_caches), so this writes to the trailing
    #    `scale_pad` bytes of each head row, reinterpreted as float32.
    k_scale_cache[block_indices, offsets, :] = k_scales
    v_scale_cache[block_indices, offsets, :] = v_scales


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
