#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Li Wei, Tang Shiwen
# Email: liwei157@baidu.com, tangshiwen@baidu.com
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

from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsKVCacheMethod,
    CompressedTensorsLinearMethod,
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)

from vllm_kunlun.quantization.utils import _remove_quantization_method

from .compressed_tensors_moe import KunlunCompressedTensorsMoEMethod

# reove the original compressed-tensors quantization methods
_remove_quantization_method("compressed-tensors")


# register the kunlun compressed-tensors quantization methods
@register_quantization_config("compressed-tensors")
class KunlunCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            input_tfms, output_tfms = get_linear_transform_schemes(
                layer, prefix, self.transform_config, self.packed_modules_mapping
            )

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)

            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, quant_scheme, input_tfms, output_tfms
                )

            else:
                return quant_method

        if isinstance(layer, Attention):
            return KunlunCompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return KunlunCompressedTensorsMoEMethod.get_moe_method(self, layer, prefix)
        return None


class KunlunCompressedTensorsKVCacheMethod(CompressedTensorsKVCacheMethod):
    """Prepare the static per-head KV scale layouts after weight loading."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        if layer.kv_cache_dtype != "int8_perchannel":
            return

        k_step = layer._k_scale
        v_step = layer._v_scale
        num_kv_heads = layer.impl.num_kv_heads
        head_size = layer.impl.head_size
        valid_scale_sizes = (1, num_kv_heads)
        if (
            k_step.numel() not in valid_scale_sizes
            or v_step.numel() not in valid_scale_sizes
        ):
            raise ValueError(
                "INT8 KV cache expects scalar or per-head k_scale/v_scale, "
                f"got k={tuple(k_step.shape)}, v={tuple(v_step.shape)}, "
                f"num_kv_heads={num_kv_heads}"
            )

        # Checkpoint scale is step=absmax/127. reshape_and_cache consumes one
        # absmax per KV head, while attention consumes it expanded over head_dim.
        k_cache_max = (
            k_step.float().flatten().mul(127).expand(num_kv_heads).contiguous()
        )
        v_cache_max = (
            v_step.float().flatten().mul(127).expand(num_kv_heads).contiguous()
        )
        k_attention_max = k_cache_max[:, None].expand(-1, head_size).contiguous()
        v_attention_max = v_cache_max[:, None].expand(-1, head_size).contiguous()
        layer._kunlun_kv_scale_layouts = (
            k_cache_max,
            v_cache_max,
            k_attention_max,
            v_attention_max,
        )
