#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Patch ``Attention.get_kv_cache_spec`` so the KV-cache page size used for
# raw allocation matches the padded shape returned by the Kunlun attention
# backend's ``get_kv_cache_shape`` when the cache dtype carries inline
# per-(token, head) float32 scales (``int8_per_token_head`` /
# ``fp8_per_token_head``).
#
# Without this patch ``_allocate_kv_cache_tensors`` allocates
# ``num_blocks * page_size_bytes`` bytes using the unpadded ``head_size``
# while ``_reshape_kv_cache_tensors`` later calls
# ``.view((2, num_blocks, num_kv_heads, block_size, head_size + scale_pad))``,
# which fails with::
#
#     RuntimeError: shape '[2, N, H, B, D+pad]' is invalid for input ...
#

import logging

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_vllm_kunlun_per_token_head_patched"


def _padded_page_size_bytes(spec, scale_pad: int) -> int:
    import torch

    elem = torch.empty((), dtype=spec.dtype).element_size()
    head_size = spec.head_size
    head_size_v = getattr(spec, "head_size_v", None) or head_size
    # Mirror FullAttentionSpec.real_page_size_bytes but inflate the K and V
    # head rows by ``scale_pad`` cache-dtype elements each so they hold the
    # inline float32 per-(token, head) scale produced by the Kunlun backend.
    padded_kv = (head_size + scale_pad) + (head_size_v + scale_pad)
    return spec.block_size * spec.num_kv_heads * padded_kv * elem


def apply_patch() -> None:
    # Imports are deferred so that this module can be loaded by
    # ``vllm_kunlun.__init__.register()`` without risking a circular import
    # via the vllm/vllm_kunlun plugin loading sequence.
    from dataclasses import replace

    from vllm.attention.layer import Attention
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheSpec,
        SlidingWindowSpec,
    )

    from vllm_kunlun.v1.attention.backends.kunlun_attn import (
        _per_token_head_scale_pad,
        kv_cache_uses_per_token_head_scales,
    )

    if getattr(Attention, _PATCHED_FLAG, False):
        return

    original = Attention.get_kv_cache_spec

    def _patched_get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        spec = original(self, vllm_config)
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        if not kv_cache_uses_per_token_head_scales(cache_dtype_str):
            return spec
        if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            return spec
        if spec.page_size_padded is not None:
            return spec

        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
        scale_pad = _per_token_head_scale_pad(cache_dtype)
        padded = _padded_page_size_bytes(spec, scale_pad)
        real = spec.real_page_size_bytes
        if padded <= real:
            return spec

        new_spec = replace(spec, page_size_padded=padded)
        logger.info(
            "[vllm-kunlun] per-token-head KV: padded page size %d -> %d "
            "(scale_pad=%d, dtype=%s)",
            real,
            padded,
            scale_pad,
            cache_dtype_str,
        )
        return new_spec

    Attention.get_kv_cache_spec = _patched_get_kv_cache_spec
    setattr(Attention, _PATCHED_FLAG, True)
    logger.info(
        "[vllm-kunlun] patched Attention.get_kv_cache_spec for per-token-head KV"
    )
