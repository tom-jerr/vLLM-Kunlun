"""
Per-channel dynamic INT8 KV-cache quantization accuracy tests.

Two scenarios are exercised, mirroring the runtime KV-quant scheme:

* Prefill: the full prompt's K/V is available in one shot. Each layer computes
  ``k_max = key.abs().amax(dim=0)`` (shape ``[num_kv_heads, head_dim]``) and
  ``v_max`` once, then passes them into ``reshape_and_cache`` (quant_mode=1)
  and ``paged_attention`` as per-channel scales (kernel divides by 127 internally).

* Decode: each step appends one new K/V token. The layer keeps the fixed
  prefill ``k_scale`` / ``v_scale`` (actually per-channel **max**) and does not
  update them during decode. All prefill and decode tokens are quantized and
  dequantized with this same per-layer scale:
      reshape_and_cache(new_k, ..., prefill_k_scale, ..., quant_mode=1)
      paged_attention(... k_perchannel_scale=prefill_k_scale ...)

The bf16 reference path runs the same K/V through ``reshape_and_cache``
(quant_mode=0) and ``paged_attention`` without scales.

Run:
    python -u test_per_channel_dynamic_quant.py
"""

import time
import torch

import torch_xmlir  # noqa: F401  (registers the xpu backend behind cuda:* names)
torch_xmlir.xpu.set_device(0)

import sys
if "kunlun_ops" not in sys.modules:
    try: 
        import xtorch_ops as _torch_ops  # noqa: F401

        sys.modules["kunlun_ops"] = _torch_ops
    except ImportError:
        # torch_ops not available in this environment; leave kunlun_ops
        # unresolved so the original ImportError surfaces at the real call site.
        pass
import kunlun_ops


# ----------------------- config -----------------------
device = "cuda:0"
dtype = torch.bfloat16

batch_size = 1
num_q_heads = 8
num_kv_heads = 8
head_size = 128
block_size = 16
num_blocks = 512
ctx_lens = [64, 512, 2048, 8192]  # benchmarked at multiple lengths
qlen_spec = 1                     # speculative decode: 1 token / batch
num_iters = 200                   # perf iterations
warmup_iters = 20

torch.manual_seed(0)


# ----------------------- helpers -----------------------
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def bench(fn, iters: int, warmup: int) -> float:
    """Return mean ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _flat(t: torch.Tensor) -> torch.Tensor:
    """Kernel takes per-channel max as a flat float32 tensor of length
    num_kv_heads * head_size."""
    return t.to(torch.float32).reshape(-1).contiguous()


# ----------------------- inputs -----------------------
def build_case(ctx_len: int):
    assert ctx_len <= num_blocks * block_size

    q = torch.randn(batch_size, qlen_spec, num_q_heads, head_size,
                    dtype=dtype, device=device)
    key = torch.randn(ctx_len, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(ctx_len, num_kv_heads, head_size, dtype=dtype, device=device)

    slot_mapping = torch.arange(ctx_len, dtype=torch.int32, device=device)
    num_blocks_used = (ctx_len + block_size - 1) // block_size
    block_tables = torch.arange(num_blocks_used, dtype=torch.int32,
                                device=device).unsqueeze(0)
    context_lens_cpu = torch.tensor([ctx_len], dtype=torch.int32)
    context_lens_xpu = context_lens_cpu.to(device)

    # ----- BF16 reference cache -----
    key_cache_bf16 = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                                 dtype=dtype, device=device)
    value_cache_bf16 = torch.zeros_like(key_cache_bf16)
    kunlun_ops.reshape_and_cache(
        key, value,
        key_cache_bf16, value_cache_bf16,
        slot_mapping,
        None, None, 0, False,
    )
    out_bf16 = torch.zeros(batch_size, qlen_spec, num_q_heads, head_size,
                           dtype=dtype, device=device)

    # ----- INT8 per-channel cache -----
    k_max = key.float().abs().amax(dim=0).reshape(-1).contiguous().clamp_min_(1e-8)
    v_max = value.float().abs().amax(dim=0).reshape(-1).contiguous().clamp_min_(1e-8)

    key_cache_i8 = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                               dtype=torch.int8, device=device)
    value_cache_i8 = torch.zeros_like(key_cache_i8)
    kunlun_ops.reshape_and_cache(
        key, value,
        key_cache_i8, value_cache_i8,
        slot_mapping,
        k_max, v_max, 1, False,
    )
    # Kunlun convention: paged_attention takes per-channel **max** (kernel /127 internally).
    k_perchannel_scale = k_max.to(torch.float32).contiguous()
    v_perchannel_scale = v_max.to(torch.float32).contiguous()
    out_i8 = torch.zeros(batch_size, qlen_spec, num_q_heads, head_size,
                         dtype=dtype, device=device)

    def run_bf16():
        kunlun_ops.paged_attention(
            out_bf16, q,
            key_cache_bf16, value_cache_bf16,
            block_tables, context_lens_cpu, context_lens_xpu,
            False, True, -1, -1, -1, 0.0, 0,
            None, None, None, None, None, None,
            False, True,
        )

    def run_i8():
        kunlun_ops.paged_attention(
            out_i8, q,
            key_cache_i8, value_cache_i8,
            block_tables, context_lens_cpu, context_lens_xpu,
            False, True, -1, -1, -1, 0.0, 0,
            None, None, k_perchannel_scale, v_perchannel_scale, None, None,
            False, True,
        )

    return run_bf16, run_i8, out_bf16, out_i8


# =====================================================================
# Accuracy: per-channel dynamic scale (prefill + decode)
# =====================================================================
#
# Models the runtime scheme that will live in kunlun_attn.py:
#   - layer.k_scale / layer.v_scale: per-channel **max**, shape
#     [num_kv_heads, head_dim], dtype float32. Initialized from prefill.
#   - prefill: k_max = key.abs().amax(dim=0); update layer state to k_max.
#   - decode : do not update layer.k_scale / layer.v_scale.
#   - In both, the prefill layer.k_scale/v_scale are passed as
#     k_perchannel_scale / v_perchannel_scale to reshape_and_cache (mode=1)
#     and paged_attention, so all tokens use the same quant/dequant scale.
# =====================================================================
def _init_caches():
    kc_bf16 = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                          dtype=dtype, device=device)
    vc_bf16 = torch.zeros_like(kc_bf16)
    kc_i8 = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                        dtype=torch.int8, device=device)
    vc_i8 = torch.zeros_like(kc_i8)
    return kc_bf16, vc_bf16, kc_i8, vc_i8


def _block_tables_for(ctx_len: int) -> torch.Tensor:
    nb = (ctx_len + block_size - 1) // block_size
    return torch.arange(nb, dtype=torch.int32, device=device).unsqueeze(0)


def run_attn(out, q, kc, vc, ctx_len, k_scale_flat=None, v_scale_flat=None):
    """Single batch, qlen_spec query, full-context paged_attention."""
    block_tables = _block_tables_for(ctx_len)
    ctx_lens_cpu = torch.tensor([ctx_len], dtype=torch.int32)
    ctx_lens_xpu = ctx_lens_cpu.to(device)
    kunlun_ops.paged_attention(
        out, q,
        kc, vc,
        block_tables, ctx_lens_cpu, ctx_lens_xpu,
        False, True, -1, -1, -1, 0.0, 0,
        None, None, k_scale_flat, v_scale_flat, None, None,
        False, True,
    )


def test_prefill(prefill_len: int):
    """Prefill scenario: compute per-channel max from the whole prefill."""
    key = torch.randn(prefill_len, num_kv_heads, head_size,
                      dtype=dtype, device=device)
    value = torch.randn(prefill_len, num_kv_heads, head_size,
                        dtype=dtype, device=device)
    q = torch.randn(batch_size, qlen_spec, num_q_heads, head_size,
                    dtype=dtype, device=device)
    slot_mapping = torch.arange(prefill_len, dtype=torch.int32, device=device)

    kc_bf16, vc_bf16, kc_i8, vc_i8 = _init_caches()

    # bf16 reference
    kunlun_ops.reshape_and_cache(
        key, value, kc_bf16, vc_bf16, slot_mapping,
        None, None, 0, False,
    )

    # Layer state: k_scale / v_scale shape [num_kv_heads, head_dim].
    k_scale = key.float().abs().amax(dim=0).contiguous().clamp_min_(1e-8)
    v_scale = value.float().abs().amax(dim=0).contiguous().clamp_min_(1e-8)
    assert k_scale.shape == (num_kv_heads, head_size)

    k_scale_flat = _flat(k_scale)
    v_scale_flat = _flat(v_scale)

    kunlun_ops.reshape_and_cache(
        key, value, kc_i8, vc_i8, slot_mapping,
        k_scale_flat, v_scale_flat, 1, False,
    )

    out_bf16 = torch.zeros(batch_size, qlen_spec, num_q_heads, head_size,
                           dtype=dtype, device=device)
    out_i8 = torch.zeros_like(out_bf16)
    run_attn(out_bf16, q, kc_bf16, vc_bf16, prefill_len)
    run_attn(out_i8, q, kc_i8, vc_i8, prefill_len, k_scale_flat, v_scale_flat)
    torch.cuda.synchronize()
    return cos_sim(out_i8, out_bf16), max_abs_diff(out_i8, out_bf16), \
        k_scale, v_scale, kc_bf16, vc_bf16, kc_i8, vc_i8, q


def test_decode(prefill_len: int, decode_steps: int):
    """Decode scenario: keep using the fixed prefill per-channel max.

    Re-uses the prefill state, then appends one token per step. Decode does
    not update the layer's k_scale / v_scale; the same prefill scales are
    passed to both reshape_and_cache and paged_attention for all new tokens.
    """
    # Bootstrap from a prefill run (also seeds caches and layer scales).
    _, _, k_scale, v_scale, kc_bf16, vc_bf16, kc_i8, vc_i8, _ = \
        test_prefill(prefill_len)
    assert k_scale.shape == (num_kv_heads, head_size)
    k_scale_flat = _flat(k_scale)
    v_scale_flat = _flat(v_scale)

    cur_len = prefill_len
    final_cs = final_md = None

    for step in range(decode_steps):
        new_k = torch.randn(qlen_spec, num_kv_heads, head_size,
                            dtype=dtype, device=device)
        new_v = torch.randn(qlen_spec, num_kv_heads, head_size,
                            dtype=dtype, device=device)
        q = torch.randn(batch_size, qlen_spec, num_q_heads, head_size,
                        dtype=dtype, device=device)

        slot_dec = torch.tensor([cur_len + i for i in range(qlen_spec)],
                                dtype=torch.int32, device=device)

        # bf16 reference: write new tokens.
        kunlun_ops.reshape_and_cache(
            new_k, new_v, kc_bf16, vc_bf16, slot_dec,
            None, None, 0, False,
        )
        # int8 path: write with the fixed prefill per-channel max.
        kunlun_ops.reshape_and_cache(
            new_k, new_v, kc_i8, vc_i8, slot_dec,
            k_scale_flat, v_scale_flat, 1, False,
        )

        cur_len += qlen_spec
        out_bf16 = torch.zeros(batch_size, qlen_spec, num_q_heads, head_size,
                               dtype=dtype, device=device)
        out_i8 = torch.zeros_like(out_bf16)
        run_attn(out_bf16, q, kc_bf16, vc_bf16, cur_len)
        run_attn(out_i8, q, kc_i8, vc_i8, cur_len, k_scale_flat, v_scale_flat)
        torch.cuda.synchronize()

        final_cs = cos_sim(out_i8, out_bf16)
        final_md = max_abs_diff(out_i8, out_bf16)

    return final_cs, final_md

if __name__ == '__main__':

    # ===================== bench: speculative decode =====================
    # print(f"{'ctx_len':>8} | {'cos_sim':>10} | {'max_diff':>10} | "
    #     f"{'bf16 (ms)':>10} | {'int8 (ms)':>10} | {'speedup':>8}")
    # print("-" * 72)
    # for ctx_len in ctx_lens:
    #     run_bf16, run_i8, out_bf16, out_i8 = build_case(ctx_len)
    #     run_bf16()
    #     run_i8()
    #     torch.cuda.synchronize()

    #     cs = cos_sim(out_i8, out_bf16)
    #     md = max_abs_diff(out_i8, out_bf16)
    #     ms_bf16 = bench(run_bf16, num_iters, warmup_iters)
    #     ms_i8 = bench(run_i8, num_iters, warmup_iters)
    #     print(f"{ctx_len:>8} | {cs:>10.6f} | {md:>10.4f} | "
    #         f"{ms_bf16:>10.4f} | {ms_i8:>10.4f} | {ms_bf16/ms_i8:>7.3f}x")


    # ----------------------- run accuracy suite -----------------------
    print()
    print("==== prefill scenario (per-channel dynamic scale) ====")
    print(f"{'prefill':>8} | {'cos_sim':>10} | {'max_diff':>10}")
    print("-" * 36)
    for n in [64, 512, 2048]:
        cs, md, *_ = test_prefill(n)
        print(f"{n:>8} | {cs:>10.6f} | {md:>10.4f}")

    print()
    print("==== decode scenario (fixed prefill per-channel scale) ====")
    print(f"{'prefill':>8} | {'steps':>6} | {'cos_sim':>10} | {'max_diff':>10}")
    print("-" * 44)
    # ifeval & live code bench
    for n, steps in [(1024,1024), (2048,1024)]:
        cs, md = test_decode(n, steps)
        print(f"{n:>8} | {steps:>6} | {cs:>10.6f} | {md:>10.4f}")
