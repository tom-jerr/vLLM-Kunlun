#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-kunlun project.
#
from dataclasses import dataclass
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import kunlun_ops
import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from vllm_kunlun.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm_kunlun.quantization.kv_cache.per_token_head import (
    reshape_and_cache_int8_with_scales,
    reconstruction_check,
)

import os
_KV_RECON_CHECK = os.environ.get("VLLM_KUNLUN_KV_RECON_CHECK", "0") == "1"
_KV_SHAPE_LOG = os.environ.get("VLLM_KUNLUN_KV_SHAPE_LOG", "0") == "1"

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import inspect

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.kv_cache_interface import AttentionSpec
import logging

logger = logging.getLogger(__name__)

# Cache dtypes that use inline per-(token, head) float32 scale stored in the
# trailing ``scale_pad`` elements of each KV-cache head row. See
# ``docs/per_token_plan.md`` for the storage layout.
_PER_TOKEN_HEAD_SCALE_DTYPES = frozenset({
    "int8_per_token_head",
    "fp8_per_token_head",
})


def kv_cache_uses_per_token_head_scales(cache_dtype_str: str) -> bool:
    """Return True iff ``cache_dtype_str`` uses inline per-token-head scales.

    The bool is also true for unsupported dtypes that aren't in
    ``STR_DTYPE_TO_TORCH_DTYPE`` (e.g. legacy names). Callers that need to
    compute ``scale_pad`` should resolve the torch dtype first; this helper
    only gates storage-shape decisions.
    """
    if cache_dtype_str in _PER_TOKEN_HEAD_SCALE_DTYPES:
        return True
    if cache_dtype_str not in STR_DTYPE_TO_TORCH_DTYPE:
        return False
    cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
    if cache_dtype == torch.int8 and "per_token_head" in cache_dtype_str:
        return True
    return False


def _per_token_head_scale_pad(cache_dtype: torch.dtype) -> int:
    """Number of int8/fp8 elements that pad each head row to hold one float32.

    Mirrors the doc's invariant:
        scale_pad = sizeof(float32) / sizeof(cache_dtype)
    """
    float32_bytes = torch.empty((), dtype=torch.float32).element_size()
    return float32_bytes // torch.empty((), dtype=cache_dtype).element_size()


class KunlunAttentionBackend(AttentionBackend):
    """KunlunAttentionBackend"""

    supported_kv_cache_dtypes: ClassVar[list] = [
        "auto",
        "float16",
        "bfloat16",
        "int8",
        "int8_per_token_head",
    ]

    # crucial to cuda graph
    accept_output_buffer = True

    @staticmethod
    def get_name() -> str:
        """get_name attention backend name
        attention backend in "FLASH_ATTN,
        FLASH_ATTN_DIFFKV, TRITON_ATTN,
        ROCM_ATTN, ROCM_AITER_MLA,
        ROCM_AITER_TRITON_MLA, ROCM_AITER_FA,
        ROCM_AITER_MLA_SPARSE, TORCH_SDPA,
        FLASHINFER, FLASHINFER_MLA, TRITON_MLA,
        CUTLASS_MLA, FLASHMLA, FLASHMLA_SPARSE,
        FLASH_ATTN_MLA, IPEX, NO_ATTENTION, FLEX_ATTENTION,
        TREE_ATTN, ROCM_AITER_UNIFIED_ATTN, CPU_ATTN,
        CUSTOM"
        """
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["KunlunAttentionImpl"]:
        """get_impl_cls"""
        return KunlunAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["KunlunMetadata"]:
        """get_metadata_cls"""
        return KunlunMetadata

    @staticmethod
    def get_builder_cls() -> Type["KunlunAttentionMetadataBuilder"]:
        """get_builder_cls"""
        return KunlunAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        """get_kv_cache_shape

        For ``int8_per_token_head`` (and the analogous ``fp8_per_token_head``),
        the last dim is padded by ``scale_pad = sizeof(float32) / sizeof(cache_dtype)``
        int8/fp8 elements, so the trailing bytes of each head row can be
        reinterpreted as a single float32 per-(token, head) scale. This matches
        the inline-scale layout used by vLLM's Triton attention backend (see
        ``docs/per_token_plan.md``). For all other dtypes the layout is unchanged.
        """
        if current_platform.is_kunlun():
            base = (2, num_blocks, num_kv_heads, block_size, head_size)
        else:
            base = (2, num_blocks, block_size * num_kv_heads * head_size)
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            logger.info(f"int8_per_token_head shape: {base[:-1] + (base[-1] + _per_token_head_scale_pad(cache_dtype),)}")
            return base[:-1] + (base[-1] + _per_token_head_scale_pad(cache_dtype),)
        return base

    @staticmethod
    def get_kv_cache_scale_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
    ) -> Tuple[int, ...]:
        """Shape of the per-(token, head) float32 scale view extracted from the
        inline padding of the KV cache.

        Aligned with the doc's ``k_scale_cache`` / ``v_scale_cache`` views:
            (num_blocks, block_size, num_kv_heads)
        Note: the scale is now an *inline* view of the KV-cache storage
        (returned by ``KunlunAttentionImpl._ensure_scale_caches``), not a
        separately-allocated buffer. This method only describes the view
        shape for callers that want to size pre-allocated buffers or for
        tests/diagnostics.
        """
        return (num_blocks, block_size, num_kv_heads)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        """swap_blocks"""
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """copy_blocks"""
        raise NotImplementedError


@dataclass
class KunlunMetadata(AttentionMetadata, PagedAttentionMetadata):
    """KunlunMetadata"""

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    num_actual_tokens: int
    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor

    multi_modal_placeholder_index_maps: Optional[torch.Tensor] = None
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # Prefix cache loc
    kv_lod_cpu: Optional[torch.Tensor] = None
    kv_lod_xpu: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of key/value length in the batch, especially for prefix cache
    max_kv_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    query_start_loc_host: Optional[torch.Tensor] = None
    # serve only for prefix cache
    kv_prefix_start_loc_host: Optional[torch.Tensor] = None
    kv_prefix_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["KunlunMetadata"] = None
    _cached_decode_metadata: Optional["KunlunMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    enable_kv_scales_calculation: Optional[bool] = False
    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: Optional[torch.Tensor] = None

    use_cascade: Optional[bool] = False

    seq_lens_tensor_cpu: Optional[torch.Tensor] = None

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    is_speculative: Optional[bool] = False
    max_model_len: int = 0

    def __post_init__(self):
        """__post_init__"""
        self.attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821
        self.encoder_attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821
        self.cross_attn_bias: Optional[List["AttentionBias"]] = None  # noqa: F821

    @property
    def is_all_encoder_attn_metadata_set(self):
        """is_all_encoder_attn_metadata_set"""
        return (
            (self.encoder_seq_lens is not None)
            and (self.encoder_seq_lens_tensor is not None)
            and (self.max_encoder_seq_len is not None)
        )

    @property
    def is_all_cross_attn_metadata_set(self):
        """is_all_cross_attn_metadata_set"""
        return (
            self.is_all_encoder_attn_metadata_set
            and (self.cross_slot_mapping is not None)
            and (self.cross_block_tables is not None)
        )

    @property
    def prefill_metadata(self) -> Optional["KunlunMetadata"]:
        """prefill_metadata"""
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        # Compute some attn_metadata fields which default to None
        query_start_loc = (
            None
            if self.query_start_loc is None
            else self.query_start_loc[-(self.num_prefills + 1) :]
            - self.query_start_loc[-(self.num_prefills + 1)]
        )
        # flash attention needs both lod information on host and device
        query_start_loc_host = (
            None
            if self.query_start_loc_host is None
            else self.query_start_loc_host[-(self.num_prefills + 1) :]
            - self.query_start_loc_host[-(self.num_prefills + 1)]
        )

        # TODO(chengruichang):how to support prefix cache
        kv_prefix_start_loc_host = None
        kv_prefix_start_loc = None
        slot_mapping = (
            None
            if self.slot_mapping is None
            else self.slot_mapping[-self.num_prefill_tokens :]
        )

        seq_lens_tensor = (
            None
            if self.seq_lens_tensor is None
            else self.seq_lens_tensor[-self.num_prefills :]
        )
        seq_lens = (
            None if self.seq_lens is None else self.seq_lens[-self.num_prefills :]
        )

        context_lens_tensor = (
            None
            if self.context_lens_tensor is None
            else self.context_lens_tensor[-self.num_prefills :]
        )

        block_tables = (
            None
            if self.block_tables is None
            else self.block_tables[-self.num_prefills :]
        )
        input_positions = (
            None
            if self.input_positions is None
            else self.input_positions[-self.num_prefills :]
        )

        if self.kv_lod_cpu is None:
            kv_lod_cpu = None
            kv_lod_xpu = None
        else:
            start = -(self.num_prefills + 1)
            base_cpu = self.kv_lod_cpu[start]
            kv_lod_cpu = self.kv_lod_cpu[start:] - base_cpu

            base_xpu = self.kv_lod_xpu[start]
            kv_lod_xpu = self.kv_lod_xpu[start:] - base_xpu

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = KunlunMetadata(
            num_actual_tokens=self.num_actual_tokens,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            seq_start_loc=None,
            kv_lod_cpu=kv_lod_cpu,
            kv_lod_xpu=kv_lod_xpu,
            max_query_len=self.max_query_len,
            max_kv_len=self.max_kv_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            input_positions=input_positions,
            kv_prefix_start_loc=kv_prefix_start_loc,
            kv_prefix_start_loc_host=kv_prefix_start_loc_host,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False,
            use_cascade=self.use_cascade,
            is_speculative=self.is_speculative,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["KunlunMetadata"]:
        """decode_metadata"""
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert (self.seq_lens_tensor is not None) or (
            self.encoder_seq_lens_tensor is not None
        )

        if self.num_prefills != 0:
            # Compute some attn_metadata fields which default to None
            slot_mapping = (
                None
                if self.slot_mapping is None
                else self.slot_mapping[: -self.num_prefill_tokens]
            )
            seq_lens_tensor = (
                None
                if self.seq_lens_tensor is None
                else self.seq_lens_tensor[: -self.num_prefills]
            )
            seq_lens_tensor_cpu = (
                None
                if self.seq_lens_tensor_cpu is None
                else self.seq_lens_tensor_cpu[: -self.num_prefills]
            )
            block_tables = (
                None
                if self.block_tables is None
                else self.block_tables[: -self.num_prefills]
            )
        else:
            # Compute some attn_metadata fields which default to None
            slot_mapping = None if self.slot_mapping is None else self.slot_mapping
            seq_lens_tensor = (
                None if self.seq_lens_tensor is None else self.seq_lens_tensor
            )
            seq_lens_tensor_cpu = (
                None if self.seq_lens_tensor_cpu is None else self.seq_lens_tensor_cpu
            )
            block_tables = None if self.block_tables is None else self.block_tables

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = KunlunMetadata(
            num_actual_tokens=self.num_actual_tokens,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps,
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            seq_lens_tensor_cpu=seq_lens_tensor_cpu,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False,
            use_cascade=self.use_cascade,
            is_speculative=self.is_speculative,
            max_model_len=self.max_model_len,
        )
        return self._cached_decode_metadata


M = TypeVar("M")


class KunlunAttentionMetadataBuilder:
    """KunlunAttentionMetadataBuilder"""

    # _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: ClassVar[Optional[int]] = 1
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """__init__"""
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()

        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.device = device

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set
            # the reorder_batch_threshold based on the number of speculative
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    1 + speculative_config.num_speculative_tokens,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    def reorder_batch(
        self, input_batch: "InputBatch", scheduler_output: "SchedulerOutput"
    ) -> bool:
        """reorder_batch"""
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: how if a prefilled sequence has only one token
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(
                    prefills[first_prefill], decodes[num_decodes - i]
                )
                first_prefill += 1
                modified_batch = True
            else:
                break
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens
        return modified_batch

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> KunlunMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens_tensor.fill_(1)
        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata
    ):
        """build"""
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        common_prefix_len = common_prefix_len
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc_host = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        query_start_loc = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].to(
            self.device, non_blocking=True
        )

        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu

        seq_start_loc = list(accumulate(seq_lens, initial=0))

        seq_start_loc_tensor = torch.empty(
            len(seq_start_loc), dtype=torch.int32, device=self.device
        )
        seq_start_loc_tensor.copy_(torch.as_tensor(seq_start_loc, dtype=torch.int32))

        kv_lod_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32, device="cpu")
        kv_lod_cpu[1:] = seq_lens_cpu.to(torch.int32).cumsum(dim=0)
        kv_lod_xpu = kv_lod_cpu.to(self.device)

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold or 1,
                require_uniform=True,
            )
        )

        num_scheduled_tokens = np.diff(
            common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        )
        tmp_decode_scheduled_tokens = num_scheduled_tokens[:num_decodes]

        if num_decode_tokens == 0:
            max_decode_seq_len = 0
        else:
            max_decode_seq_len = np.max(tmp_decode_scheduled_tokens)

        tmp_prefill_scheduled_tokens = num_scheduled_tokens[num_decodes:num_reqs]

        if num_prefill_tokens == 0:
            max_prefill_seq_len = 0
        else:
            max_prefill_seq_len = np.max(tmp_prefill_scheduled_tokens)

        use_cascade = False

        attn_metadata = KunlunMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens_tensor=seq_lens,
            seq_lens_tensor_cpu=seq_lens_cpu,
            kv_lod_xpu=kv_lod_xpu,
            kv_lod_cpu=kv_lod_cpu,
            max_query_len=max_prefill_seq_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            query_start_loc_host=query_start_loc_host,
            context_lens_tensor=None,
            block_tables=block_table_tensor,
            use_cuda_graph=False,
            use_cascade=use_cascade,
            is_speculative=self.reorder_batch_threshold > 1,
            max_model_len=self.vllm_config.model_config.max_model_len,
        )
        return attn_metadata

    def can_run_in_cudagraph(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> bool:
        """can_run_in_cudagraph"""
        # Full CUDA Graph always supported (FA2 support checked separately)
        return True

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        """use_cascade_attention"""
        return use_cascade_attention(*args, **kwargs)


class KunlunAttentionImpl(AttentionImpl[KunlunMetadata]):
    """KunlunAttentionImpl"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        kv_sharing_target_layer_name: Optional[str] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        sinks: Optional[torch.Tensor] = None,
        multi_modal_placeholder_index_maps: Optional[torch.Tensor] = None,
    ) -> None:
        """__init__"""
        if blocksparse_params is not None:
            raise ValueError("kunlunAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.use_irope = use_irope

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}."
            )

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.multi_modal_placeholder_index_maps = multi_modal_placeholder_index_maps

        # Per-token per-head scale views carved from the inline head padding
        # of the KV cache. Populated lazily on the first forward when
        # ``self.kv_cache_dtype == "int8_per_token_head"`` (see
        # ``_ensure_scale_caches``). Shape: ``(num_blocks, block_size,
        # num_kv_heads)``, dtype=float32, sharing storage with the kv_cache.
        self._k_scale_cache: Optional[torch.Tensor] = None
        self._v_scale_cache: Optional[torch.Tensor] = None
        # Logical head size (without the per-token-head scale padding) used by
        # the cache-write path to know where the data region ends inside the
        # padded last dim.
        self._orin_head_size: int = head_size

    # Per-token-head quant: strided float32 views over the trailing
    # ``scale_pad`` int8 elements of each KV-cache head row.
    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Create inline scale views for per-token-head quantization.

        The KV cache for ``int8_per_token_head`` has shape
        ``(2, num_blocks, num_kv_heads, block_size, head_size + scale_pad)``
        in int8. The trailing ``scale_pad`` int8 elements of each head row are
        reinterpreted as one float32 per-(token, head) scale. This method
        builds two float32 tensors of shape ``(num_blocks, block_size,
        num_kv_heads)`` that share storage with the KV cache, mirroring
        vLLM Triton attention's ``_ensure_scale_caches`` (see
        ``docs/per_token_plan.md``).

        Idempotent: subsequent calls with the same kv_cache are no-ops; calls
        with a different kv_cache (rare; e.g. after a remap) rebuild the
        views.
        """
        if self._k_scale_cache is not None and self._v_scale_cache is not None:
            # Already built for this layer; preserve cache so dequant reads
            # remain stable across forward calls.
            return
        # kv_cache: (2, num_blocks, num_kv_heads, block_size, head_size+scale_pad)
        num_blocks = kv_cache.shape[1]
        num_kv_heads = kv_cache.shape[2]
        block_size = kv_cache.shape[3]
        head_size_with_pad = kv_cache.shape[4]
        cache_dtype = kv_cache.dtype
        scale_pad = _per_token_head_scale_pad(cache_dtype)
        head_size = head_size_with_pad - scale_pad
        # Cache the logical head size so the write path can slice the data
        # region off the padded last dim.
        self._orin_head_size = head_size

        # Byte strides per (block, head, slot) in int8 element units.
        block_bytes = num_kv_heads * block_size * head_size_with_pad
        head_bytes = block_size * head_size_with_pad
        slot_bytes = head_size_with_pad
        # In float32 element units (4 bytes each): scale_pad=4 ensures
        # (head_size+scale_pad) is a multiple of 4 for all supported head
        # sizes, so these integer divisions are exact.
        block_f32 = block_bytes // 4
        head_f32 = head_bytes // 4
        slot_f32 = slot_bytes // 4
        scale_off_f32 = (head_size * torch.empty((), dtype=cache_dtype).element_size()) // 4

        # Build a float32 view over the raw storage of the kv_cache tensor.
        # `untyped_storage()` returns the underlying byte buffer; we wrap it
        # as a 0-element float32 tensor and then use `as_strided` to carve
        # out the scale region. This is identical to the approach used in
        # vLLM's Triton attention backend.
        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor(
            [], dtype=torch.float32, device=kv_cache.device,
        ).set_(raw)

        # K scales live in the kv_half=0 plane of the storage; their byte
        # offset within a head row is `head_size * elem_size`.
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, num_kv_heads),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        # V scales live in the kv_half=1 plane, one full K/V plane
        # (= `block_bytes` bytes) further into the storage.
        kv_half_bytes = num_blocks * num_kv_heads * block_size * head_size_with_pad
        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, num_kv_heads),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )

    # @nvtx.annotate("batch_prefill_merge", color="red")
    def _decode_via_prefill_merge(
        self,
        out: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens_cpu: torch.Tensor,
        k_perblock_perhead_scale: Optional[torch.Tensor] = None,
        v_perblock_perhead_scale: Optional[torch.Tensor] = None,
        k_static_perchannel_scale: Optional[torch.Tensor] = None,
        v_static_perchannel_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Unified decode path: per-block (block_size=1) prefill_attention +
        LSE merge.

        Replaces both speculative_attention and paged_attention with a
        block-by-block evaluation. Each cached token is treated as a 1-token
        "prefill block"; partial outputs are combined via
        attention_merge_stage. Supports:
          - bf16/fp16 cache (no scales)
          - int8 static (k/v_static_perchannel_scale broadcast to all blocks)
          - int8 dynamic per-token per-head (k/v_perblock_perhead_scale)

        Shapes (block_size = 1):
            out:                       [batch, qlen, head_num, head_dim]
            q:                         [batch, qlen, head_num, head_dim]
            k/v_cache:                 [num_blocks, kv_heads, 1, head_dim]
            block_tables:              [batch, max_blocks_per_seq]
            context_lens_cpu:          [batch] int32
            k/v_perblock_perhead_scale:[num_blocks, kv_heads] or None
            k/v_static_perchannel_scale:[kv_heads, head_dim] or None
        """
        batch_size, qlen, head_num, head_dim = q.shape
        device = q.device
        dtype = q.dtype

        q_lod_cpu = torch.tensor([0, qlen], dtype=torch.int32)
        q_lod_xpu = q_lod_cpu.to(device)
        kv_lod_cpu = torch.tensor([0, 1], dtype=torch.int32)
        kv_lod_xpu = kv_lod_cpu.to(device)

        
        for batch_idx in range(batch_size):
            ctx_len = int(context_lens_cpu[batch_idx].item())
            if ctx_len <= 0:
                continue
            q_tokens = q[batch_idx]  # [qlen, H, D]

            # During warm-up / dummy runs the contents of `block_tables` are
            # uninitialized memory (kernels that consume it on-device clamp
            # internally, but our python-side .item() does not). Clamp
            # defensively against the cache's first-dim length.
            num_total_blocks = k_cache.shape[0]
            max_block_idx = num_total_blocks - 1

            acc_out = None
            acc_lse = None
            for logical_idx in range(ctx_len):
                raw_block = int(
                    block_tables[batch_idx, logical_idx].item()
                )
                if raw_block < 0 or raw_block > max_block_idx:
                    physical_block = 0
                else:
                    physical_block = raw_block
                
                # with nvtx.annotate("kv_scale_calc", color="green"):
                if k_perblock_perhead_scale is not None:
                    k_scale = (
                        k_perblock_perhead_scale[physical_block]
                        .view(-1, 1)
                        .expand(-1, head_dim)
                        .contiguous()
                    )
                    v_scale = (
                        v_perblock_perhead_scale[physical_block]
                        .view(-1, 1)
                        .expand(-1, head_dim)
                        .contiguous()
                    )
                else:
                    k_scale = k_static_perchannel_scale
                    v_scale = v_static_perchannel_scale
                # with nvtx.annotate("init_var", color="blue"):
                bt = torch.tensor(
                    [[physical_block]], dtype=torch.int32, device=device
                )
                chunk_out = torch.zeros(
                    qlen, head_num, head_dim, dtype=dtype, device=device
                )
                # prefill_attention_lse with fp16 q requires fp16 lse via
                # use_xfa_boost=True; bf16/fp32 q can use fp32 lse directly.
                use_xfa_boost = (dtype == torch.float16)
                lse_dtype = torch.float16 if use_xfa_boost else torch.float32
                chunk_lse = torch.empty(
                    head_num, qlen, dtype=lse_dtype, device=device
                )
                # with nvtx.annotate("prefill_attention", color="yellow"):
                kunlun_ops.prefill_attention(
                    q=q_tokens,
                    k=k_cache,
                    v=v_cache,
                    out=chunk_out,
                    is_causal=False,
                    is_prefix_cache=True,
                    block_table=bt,
                    context_qlen_lod_cpu=q_lod_cpu,
                    context_qlen_lod_xpu=q_lod_xpu,
                    context_kvlen_lod_cpu=kv_lod_cpu,
                    context_kvlen_lod_xpu=kv_lod_xpu,
                    k_perchannel_scale=k_scale,
                    v_perchannel_scale=v_scale,
                    softmax_lse=chunk_lse,
                    unpadded_lse=True,
                    use_xfa_boost=use_xfa_boost,
                )
                
                # with nvtx.annotate("attention_merge_state", color="purple"):
                # attention_merge_stage expects float32 LSE.
                chunk_lse_t = chunk_lse.transpose(0, 1).contiguous().to(
                    torch.float32
                )

                if acc_out is None:
                    acc_out = chunk_out
                    acc_lse = chunk_lse_t
                    continue
                next_out = torch.empty_like(acc_out)
                next_lse = torch.empty_like(acc_lse)
                kunlun_ops.attention_merge_stage(
                    acc_out, acc_lse, chunk_out, chunk_lse_t,
                    next_out, next_lse,
                )
                acc_out = next_out
                acc_lse = next_lse

            out[batch_idx].copy_(acc_out)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Optional[KunlunMetadata],
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """forward"""
        query = query.view(-1, self.num_heads, self.head_size)
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            # Profiling run.
            return output.view(-1, self.num_heads * self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # int8 KV cache quantization modes:
        # - int8: per-tensor static quantization using _k_scale/_v_scale from
        #   checkpoint (default 1.0). No scale buffer allocated.
        # - int8_per_token_head: per-token per-head dynamic quantization with
        #   scales stored *inline* in the trailing ``scale_pad`` int8 elements
        #   of each KV-cache head row (see ``_ensure_scale_caches`` and
        #   ``docs/per_token_plan.md``). The last dim of the kv_cache tensor is
        #   ``head_size + scale_pad``.
        is_int8_cache = self.kv_cache_dtype in ("int8", "int8_per_token_head")
        use_per_token_quant = self.kv_cache_dtype == "int8_per_token_head"
        logger.info_once(f"use_per_token_quant: {use_per_token_quant}")
        if is_int8_cache and not use_per_token_quant:
            # Static mode: per-tensor static scale from _k_scale/_v_scale
            # (loaded from checkpoint or default 1.0)
            k_max_per_head = (
                layer._k_scale
            ).expand(self.num_kv_heads).contiguous().to(torch.float32)
            v_max_per_head = (
                layer._v_scale
            ).expand(self.num_kv_heads).contiguous().to(torch.float32)

            k_perchannel_scale = k_max_per_head.view(
                self.num_kv_heads, 1).expand(
                self.num_kv_heads, self.head_size).contiguous()
            v_perchannel_scale = v_max_per_head.view(
                self.num_kv_heads, 1).expand(
                self.num_kv_heads, self.head_size).contiguous()
        elif use_per_token_quant:
            # Per-token per-head dynamic quantization:
            # scales are stored per-token, no static per-head scale needed for write.
            # For decode kernels that still need perchannel_scale, we set None
            # and will do explicit dequant before calling the kernel.
            k_max_per_head = None
            v_max_per_head = None
            k_perchannel_scale = None
            v_perchannel_scale = None
        else:
            k_max_per_head = None
            v_max_per_head = None
            k_perchannel_scale = None
            v_perchannel_scale = None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize
        if attn_type != AttentionType.ENCODER and kv_cache.numel() > 0:
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            if (key is not None) and (value is not None):
                updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                value = value.contiguous()
                if use_per_token_quant:
                    # Per-token per-head dynamic quantization path. Aligned
                    # with vLLM Triton attention's
                    # triton_reshape_and_cache_flash_per_token_head_quant:
                    #
                    # 1. Lazily carve float32 scale views from the inline
                    #    head padding of the kv_cache (no separate scale
                    #    buffer is allocated anywhere).
                    # 2. Quantize K/V per (token, head), write int8 data into
                    #    the *data* region of the cache (head_size int8 elems)
                    #    and the float32 absmax scale into the *padding*
                    #    region of the cache (scale_pad bytes re-interpreted
                    #    as one float32). All three writes share the same
                    #    storage, so cache block copy/swap automatically
                    #    carries both data and scales.
                    self._ensure_scale_caches(kv_cache)

                    num_actual = attn_metadata.num_actual_tokens
                    key_slice = key[:num_actual]
                    val_slice = value[:num_actual]

                    # Optional: reconstruction check before attention.
                    # Enable with VLLM_KUNLUN_KV_RECON_CHECK=1. We run an
                    # *in-tensor* quant to keep the rest of the path free of
                    # extra allocations.
                    if _KV_RECON_CHECK:
                        from vllm_kunlun.quantization.kv_cache.per_token_head import (
                            per_token_per_head_quant,
                        )
                        key_int8_chk, k_scales_chk = per_token_per_head_quant(
                            key_slice,
                        )
                        val_int8_chk, v_scales_chk = per_token_per_head_quant(
                            val_slice,
                        )
                        reconstruction_check(
                            "K", key_slice, key_int8_chk, k_scales_chk,
                            log_fn=logger.info,
                        )
                        reconstruction_check(
                            "V", val_slice, val_int8_chk, v_scales_chk,
                            log_fn=logger.info,
                        )

                    block_size = key_cache.shape[2]
                    slot_map = updated_slot_mapping[:num_actual].to(torch.int32)
                    if _KV_SHAPE_LOG:
                        logger.info(
                            "[KV cache write shapes] key=%s value=%s "
                            "key_cache=%s value_cache=%s "
                            "k_scale_cache=%s v_scale_cache=%s slot_mapping=%s",
                            tuple(key_slice.shape),
                            tuple(val_slice.shape),
                            tuple(key_cache.shape),
                            tuple(value_cache.shape),
                            tuple(self._k_scale_cache.shape),
                            tuple(self._v_scale_cache.shape),
                            tuple(slot_map.shape),
                        )

                    # The single fused write: quant + scatter int8 into the
                    # data region of each head row + scatter float32 scale
                    # into the inline padding (via the strided view).
                    reshape_and_cache_int8_with_scales(
                        key_slice, val_slice,
                        key_cache, value_cache,
                        self._k_scale_cache, self._v_scale_cache,
                        slot_map, block_size, self._orin_head_size,
                    )
                elif is_int8_cache:
                    if not key_cache.is_contiguous():
                        # Hybrid Attention (e.g. Qwen3-Next): key_cache is
                        # non-contiguous due to as_strided_ interleaved k/v
                        # layout. reshape_and_cache kernel computes write offset
                        # as slot * head_size assuming contiguous block stride,
                        # but actual stride[0] = 2 * block_size * head_size.
                        # Fix: remap slot_mapping so kernel writes to the
                        # correct physical address. Zero extra memory needed.
                        block_size = key_cache.shape[2]
                        stride_factor = (
                            key_cache.stride()[0]
                            // (self.num_kv_heads * block_size * self.head_size)
                        )
                        _slots = updated_slot_mapping[
                            : attn_metadata.num_actual_tokens]
                        _block_ids = _slots.long() // block_size
                        _offsets = _slots.long() % block_size
                        _adjusted_slots = (
                            _block_ids * stride_factor * block_size + _offsets
                        ).to(torch.int32)

                        kunlun_ops.reshape_and_cache(
                            key[: attn_metadata.num_actual_tokens],
                            value[: attn_metadata.num_actual_tokens],
                            key_cache,
                            value_cache,
                            _adjusted_slots,
                            k_max=k_max_per_head,
                            v_max=v_max_per_head,
                            quant_mode=0,
                        )
                    else:
                        kunlun_ops.reshape_and_cache(
                            key[: attn_metadata.num_actual_tokens],
                            value[: attn_metadata.num_actual_tokens],
                            key_cache,
                            value_cache,
                            updated_slot_mapping.to(torch.int32),
                            k_max=k_max_per_head,
                            v_max=v_max_per_head,
                            quant_mode=0,
                        )
                else:
                    kunlun_ops.reshape_and_cache_flash(
                        key[: attn_metadata.num_actual_tokens],
                        value[: attn_metadata.num_actual_tokens],
                        key_cache,
                        value_cache,
                        updated_slot_mapping,
                        BLHD_LAYOUT=False,
                    )

            # For per-token int8 cache, decode dequantizes explicitly before
            # calling kernels because current kernels do not support per-token
            # per-head KV scales natively.

        assert attn_type == AttentionType.DECODER
        # Decoder self-attention supports chunked prefill.
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Only enforce this shape-constraint for decoder
        # self-attention

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            prefill_query = query[num_decode_tokens : attn_metadata.num_actual_tokens]
            prefill_key = key[num_decode_tokens : attn_metadata.num_actual_tokens]
            prefill_value = value[num_decode_tokens : attn_metadata.num_actual_tokens]

            # For hybrid Attention (Qwen3-Next.)
            if key_cache.is_contiguous():
                tmp_block_tables = prefill_meta.block_tables
            else:
                # For hybrid Attention (Qwen3-Next)
                tmp_block_tables = prefill_meta.block_tables * 2

            # Prefix cache (chunked prefill with cached KV tokens)
            if prefill_meta.query_start_loc_host[-1] != prefill_meta.kv_lod_cpu[-1]:
                if is_int8_cache:
                    # INT8 KV cache: kernel expects k/v to be paged int8 cache
                    # tensors with shape (block_num, kv_heads, block_size, head_dim).
                    # All KV (cached + new) has been written to cache by
                    # reshape_and_cache above; kernel reads via block_table.
                    kunlun_ops.prefill_attention(
                        q=prefill_query,
                        k=key_cache,
                        v=value_cache,
                        out=output[num_decode_tokens : attn_metadata.num_actual_tokens],
                        is_causal=True,
                        is_prefix_cache=True,
                        k_perchannel_scale=k_perchannel_scale,
                        v_perchannel_scale=v_perchannel_scale,
                        block_table=tmp_block_tables,
                        context_qlen_lod_cpu=prefill_meta.query_start_loc_host,
                        context_qlen_lod_xpu=prefill_meta.query_start_loc,
                        context_kvlen_lod_cpu=prefill_meta.kv_lod_cpu,
                        context_kvlen_lod_xpu=prefill_meta.kv_lod_xpu,
                        alibi_slopes=self.alibi_slopes,
                        softmax_lse=None,
                        swa_left=(
                            self.sliding_window if self.sliding_window is not None else -1
                        ),
                        swa_right=0 if self.sliding_window is not None else -1,
                        sink=(
                            self.sinks.to(torch.float32) if self.sinks is not None else None
                        ),
                    )
                else:
                    # BF16: pass current chunk's k/v directly; kernel joins
                    # with cached KV via block_table internally.
                    kunlun_ops.prefill_attention(
                        q=prefill_query,
                        k=key_cache,
                        v=value_cache,
                        out=output[num_decode_tokens : attn_metadata.num_actual_tokens],
                        is_causal=True,
                        is_prefix_cache=True,
                        k_perchannel_scale=k_perchannel_scale,
                        v_perchannel_scale=v_perchannel_scale,
                        block_table=tmp_block_tables,
                        context_qlen_lod_cpu=prefill_meta.query_start_loc_host,
                        context_qlen_lod_xpu=prefill_meta.query_start_loc,
                        context_kvlen_lod_cpu=prefill_meta.kv_lod_cpu,
                        context_kvlen_lod_xpu=prefill_meta.kv_lod_xpu,
                        alibi_slopes=self.alibi_slopes,
                        softmax_lse=None,
                        swa_left=(
                            self.sliding_window if self.sliding_window is not None else -1
                        ),
                        swa_right=0 if self.sliding_window is not None else -1,
                        sink=(
                            self.sinks.to(torch.float32) if self.sinks is not None else None
                        ),
                    )
            else:
                
                kunlun_ops.prefill_attention(
                    q=prefill_query,
                    k=prefill_key,
                    v=prefill_value,
                    out=output[num_decode_tokens : attn_metadata.num_actual_tokens],
                    is_causal=True,
                    k_perchannel_scale=k_perchannel_scale,
                    v_perchannel_scale=v_perchannel_scale,
                    context_qlen_lod_cpu=prefill_meta.query_start_loc_host,
                    context_qlen_lod_xpu=prefill_meta.query_start_loc,
                    alibi_slopes=self.alibi_slopes,
                    softmax_lse=None,
                    swa_left=(
                        self.sliding_window if self.sliding_window is not None else -1
                    ),
                    swa_right=0 if self.sliding_window is not None else -1,
                    sink=(
                        self.sinks.to(torch.float32) if self.sinks is not None else None
                    ),
                )

        if decode_meta := attn_metadata.decode_metadata:
            assert (
                attn_type != AttentionType.ENCODER_ONLY
            ), "Encoder-only models should not have decode metadata."
            decode_query = query[:num_decode_tokens]

            # For hybrid Attention (Qwen3-Next
            if key_cache.is_contiguous():
                tmp_block_tables = decode_meta.block_tables
            else:
                tmp_block_tables = (
                    decode_meta.block_tables * 2
                )  # only test in Qwen3-Next
            if is_int8_cache:
                # Always reshape cache to block_size=1 layout for the unified
                # prefill_attention + merge_state decode path.
                #
                # Two int8 KV cache layouts coexist:
                #   * int8_per_token_head  (use_per_token_quant=True):
                #       key_cache shape: (num_blocks, num_kv_heads, block_size,
                #                         head_size + scale_pad)
                #     The trailing ``scale_pad`` int8 elements per head row
                #     are an *inline* float32 per-(token, head) scale, viewed
                #     via ``self._k_scale_cache`` / ``self._v_scale_cache``
                #     (shape ``(num_blocks, block_size, num_kv_heads)``).
                #     We must slice the data region (``:self.head_size``)
                #     before reshaping to a contiguous block_size=1 layout.
                #   * int8 (use_per_token_quant=False):
                #       key_cache shape: (num_blocks, num_kv_heads, block_size,
                #                         head_size)
                #     No padding; static per-channel scales come from
                #     ``k_perchannel_scale`` / ``v_perchannel_scale``.
                original_block_size = key_cache.shape[2]
                num_physical_blocks = key_cache.shape[0]
                if use_per_token_quant:
                    # Strip trailing scale_pad before reshape; ``contiguous()``
                    # forces a copy of just the data region.
                    _key_cache = key_cache[..., : self.head_size].permute(
                        0, 2, 1, 3
                    ).reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                        1,
                        self.head_size,
                    ).contiguous()
                    _value_cache = value_cache[..., : self.head_size].permute(
                        0, 2, 1, 3
                    ).reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                        1,
                        self.head_size,
                    ).contiguous()

                    # Inline-scale views: (num_blocks, block_size, num_kv_heads)
                    # already in (block, slot, head) order, so just flatten the
                    # leading two dims.
                    assert (
                        self._k_scale_cache is not None
                        and self._v_scale_cache is not None
                    ), (
                        "Per-token-head scale views must be built by "
                        "_ensure_scale_caches before decode."
                    )
                    _k_perblock_perhead = self._k_scale_cache.reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                    ).contiguous()
                    _v_perblock_perhead = self._v_scale_cache.reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                    ).contiguous()
                    _k_static = None
                    _v_static = None
                else:
                    # Plain int8 cache: last dim is already self.head_size.
                    _key_cache = key_cache.permute(0, 2, 1, 3).reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                        1,
                        self.head_size,
                    ).contiguous()
                    _value_cache = value_cache.permute(0, 2, 1, 3).reshape(
                        num_physical_blocks * original_block_size,
                        self.num_kv_heads,
                        1,
                        self.head_size,
                    ).contiguous()
                    _k_perblock_perhead = None
                    _v_perblock_perhead = None
                    _k_static = k_perchannel_scale
                    _v_static = v_perchannel_scale

                token_offsets = torch.arange(
                    original_block_size,
                    dtype=tmp_block_tables.dtype,
                    device=tmp_block_tables.device,
                )
                tmp_block_tables = (
                    tmp_block_tables.unsqueeze(-1) * original_block_size
                    + token_offsets.view(1, 1, original_block_size)
                ).reshape(tmp_block_tables.shape[0], -1).contiguous()

                # Determine batch_size / qlen for both regular decode (qlen=1) and
                # speculative decode (qlen>1).
                if attn_metadata.is_speculative:
                    batch_size = attn_metadata.num_decodes
                    query_seq_len = decode_query.shape[0]
                    assert query_seq_len % batch_size == 0
                    qlen = query_seq_len // batch_size
                else:
                    batch_size = decode_meta.block_tables.shape[0]
                    qlen = 1

                if _KV_SHAPE_LOG:
                    logger.info(
                        "[decode_via_prefill_merge shapes] decode_query=%s "
                        "output=%s key_cache=%s value_cache=%s "
                        "k_perblock=%s v_perblock=%s k_static=%s v_static=%s "
                        "context_lens_cpu=%s block_tables=%s "
                        "tmp_block_tables=%s num_decode_tokens=%s "
                        "batch_size=%s qlen=%s use_per_token_quant=%s",
                        tuple(decode_query.shape),
                        tuple(output[:num_decode_tokens].shape),
                        tuple(_key_cache.shape),
                        tuple(_value_cache.shape),
                        None if _k_perblock_perhead is None
                        else tuple(_k_perblock_perhead.shape),
                        None if _v_perblock_perhead is None
                        else tuple(_v_perblock_perhead.shape),
                        None if _k_static is None else tuple(_k_static.shape),
                        None if _v_static is None else tuple(_v_static.shape),
                        tuple(decode_meta.seq_lens_tensor_cpu.shape),
                        tuple(decode_meta.block_tables.shape),
                        tuple(tmp_block_tables.shape),
                        num_decode_tokens,
                        batch_size,
                        qlen,
                        use_per_token_quant,
                    )

                logger.info_once(
                    "USE UNIFIED DECODE: prefill_attention + merge_state "
                    "(block_size=1)"
                )
                self._decode_via_prefill_merge(
                    out=output[:num_decode_tokens].view(
                        batch_size, qlen, self.num_heads, self.head_size
                    ),
                    q=decode_query.view(
                        batch_size, qlen, self.num_heads, self.head_size
                    ),
                    k_cache=_key_cache,
                    v_cache=_value_cache,
                    block_tables=tmp_block_tables,
                    context_lens_cpu=decode_meta.seq_lens_tensor_cpu,
                    k_perblock_perhead_scale=_k_perblock_perhead,
                    v_perblock_perhead_scale=_v_perblock_perhead,
                    k_static_perchannel_scale=_k_static,
                    v_static_perchannel_scale=_v_static,
                )
            else:
                # Note: bf16 kv cache calc
                # Determine batch_size and qlen based on whether it's speculative
                logger.info(f"bf16 kv compute")
                if not attn_metadata.is_speculative:
                    batch_size = decode_meta.block_tables.shape[0]
                    qlen = 1
                    # Reshape q for non-speculative case
                    q = decode_query.unsqueeze(
                        0
                    )  # [1, batch_size*qlen, head_num, head_dim]
                    out = output[:num_decode_tokens]
                else:
                    batch_size = attn_metadata.num_decodes
                    query_seq_len, head_num, head_dim = decode_query.shape
                    assert query_seq_len % batch_size == 0
                    qlen = query_seq_len // batch_size
                    q = decode_query.view(batch_size, qlen, head_num, head_dim)
                    out = output[:num_decode_tokens]
                    assert out.is_contiguous()
                    out = out.view(batch_size, qlen, head_num, self.head_size)

                kunlun_ops.speculative_attention(
                    out=out,
                    q=q,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    context_lens_cpu=decode_meta.seq_lens_tensor_cpu,
                    context_lens_xpu=decode_meta.seq_lens_tensor,
                    batch_num=batch_size,
                    qlen=qlen,
                    max_context_len=decode_meta.max_model_len,
                    head_num=self.num_heads,
                    head_dim=self.head_size,
                    scale=self.scale,
                    kv_head_num=self.num_kv_heads,
                    block_size=key_cache.shape[2],
                    max_num_blocks_per_seq=decode_meta.block_tables.shape[1],
                    block_tables=tmp_block_tables,
                )


        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    num_sms: int,
    dcp_world_size: int,
    use_local_attention: bool = False,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    return False

    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time
