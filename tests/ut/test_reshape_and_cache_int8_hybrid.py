# Minimal reproduction for the int8 dynamic-per-channel KV-cache garbage
# bug in attention + Mamba/GDN hybrid models (e.g. Qwen3.5).
#
# Background
# ----------
# Hybrid (attention + Mamba/GDN) models force the attention KV cache into a
# *block-first* (interleaved) physical layout via
# ``_update_hybrid_attention_mamba_layout`` (gpu_model_runner). After that
# re-striding, ``key_cache`` is NON-contiguous: ``stride[0] == 2 * hidden_size``
# instead of ``hidden_size``. This is required so the KVCacheManager block_id /
# page semantics line up with the real physical K/V addresses.
#
# The ONLY kernel that quantizes int8 per-channel KV cache on this build is the
# legacy ``kunlun_ops.reshape_and_cache`` (quant_mode=1, int32 slot_mapping).
# That kernel FLAT-INDEXES the cache and assumes it is contiguous, so under the
# interleaved block-first layout it writes the quantized K/V to the WRONG
# physical slots. Decode then reads garbage -> the model emits repeated "!!!!".
#
# NOTE (verified on this build): ``reshape_and_cache_flash`` does NOT accept an
# int8 key_cache for quant_mode=1 -- it raises
#   RuntimeError: reshape_and_cache_flash not support key/key_cache/slot_mapping dtype
# So dispatching to the flash kernel is NOT a viable fix for int8. The fix must
# keep the int8 attention cache CONTIGUOUS (skip the hybrid interleave when
# cache_dtype == "int8_dynamic_per_channel"), so the legacy kernel writes to the
# correct slots. This test encodes exactly that contract.
#
# Run (XPU/CUDA device required; uses xtorch_ops, API identical to kunlun_ops):
#     python tests/ut/test_reshape_and_cache_int8_hybrid.py
#     SMOKE=1 python tests/ut/test_reshape_and_cache_int8_hybrid.py   # single case
#
import os

import torch

# xtorch_ops exposes the same reshape_and_cache API as kunlun_ops; runs on cuda:0.
import xtorch_ops as ops


# --------------------------------------------------------------------------- #
# Reference per-channel int8 quant (mirrors quant_mode=1):
#   scale = amax / 127 ;  q = clamp(round(x / scale), -127, 127)
# amax (a.k.a. k_max / v_max) is passed flat, shape [num_kv_heads * head_size].
# --------------------------------------------------------------------------- #
def quant_per_channel(x, amax):
    # x: [num_tokens, num_kv_heads, head_size]; amax: [num_kv_heads, head_size]
    scale = (amax / 127.0).clamp_min(1e-8)
    return torch.round(x.cpu() / scale.cpu()).clamp_(-127, 127).to(torch.int8)


def _read_logical(cache_int8, slot_mapping, block_size):
    """Read per-token int8 vectors from a logical
    [num_blocks, num_kv_heads, block_size, head_size] cache."""
    out = []
    for slot in slot_mapping.tolist():
        blk, off = slot // block_size, slot % block_size
        out.append(cache_int8[blk, :, off, :].cpu())
    return torch.stack(out, dim=0)  # [num_tokens, num_kv_heads, head_size]


# --------------------------------------------------------------------------- #
def run_case(num_blocks, block_size, num_kv_heads, head_size, num_tokens,
             device, dtype=torch.float16):
    torch.manual_seed(0)
    key = torch.randn(num_tokens, num_kv_heads, head_size,
                      dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size,
                        dtype=dtype, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)

    amax_k = key.float().abs().amax(dim=0).clamp_min(1e-8)  # [H, D]
    amax_v = value.float().abs().amax(dim=0).clamp_min(1e-8)
    k_max_flat = amax_k.reshape(-1).contiguous()            # [H*D]
    v_max_flat = amax_v.reshape(-1).contiguous()

    # Reference quantized vectors (what each slot must contain).
    ref_q = quant_per_channel(key, amax_k)  # [num_tokens, H, D] int8

    results = {}

    # ----- Path 1: CONTIGUOUS BHLD cache (pure-attention models, Qwen3) ---- #
    kv_contig = torch.zeros(2, num_blocks, num_kv_heads, block_size, head_size,
                            dtype=torch.int8, device=device)
    kc, vc = kv_contig[0], kv_contig[1]
    assert kc.is_contiguous(), "expected contiguous cache for path 1"
    ret = ops.reshape_and_cache(
        key, value, kc, vc, slot_mapping,
        k_max=k_max_flat, v_max=v_max_flat, quant_mode=1)
    assert ret == 0, f"contiguous write returned {ret}"
    got = _read_logical(kc, slot_mapping, block_size)
    results["contiguous"] = (got.int() - ref_q.int()).abs().max().item()

    # ----- Path 2: HYBRID block-first / interleaved (non-contiguous) ------- #
    # Reproduce _update_hybrid_attention_mamba_layout: a single (2, num_blocks,
    # ...) buffer re-strided to (num_blocks, 2, ...) so K and V interleave per
    # block and key_cache.stride[0] == 2 * hidden_size.
    hidden = num_kv_heads * block_size * head_size
    kv_hybrid = torch.zeros(2, num_blocks, num_kv_heads, block_size, head_size,
                            dtype=torch.int8, device=device)
    kv_hybrid.as_strided_(
        size=kv_hybrid.shape,
        stride=(hidden, 2 * hidden, *kv_hybrid.stride()[2:]),
    )
    kc_h, vc_h = kv_hybrid[0], kv_hybrid[1]
    assert not kc_h.is_contiguous(), "hybrid key_cache must be non-contiguous"

    ret = ops.reshape_and_cache(
        key, value, kc_h, vc_h, slot_mapping,
        k_max=k_max_flat, v_max=v_max_flat, quant_mode=1)
    # ret is silently 0 even though the data is wrong -> garbage, not an error.
    got_hybrid = _read_logical(kc_h, slot_mapping, block_size)
    results["hybrid_ret"] = ret
    results["hybrid"] = (got_hybrid.int() - ref_q.int()).abs().max().item()

    # ----- Path 3: THE FIX -- physical contiguous view + slot block-doubling #
    # No gpu_model_runner / vllm change. The interleaved storage maps
    #   logical K[blk] -> physical block 2*blk
    #   logical V[blk] -> physical block 2*blk + 1
    # reshape_and_cache flat-indexes a contiguous cache, so expose a contiguous
    # physical view of the underlying storage (K view at offset 0, V view at
    # offset `hidden` = physical block 1) and double the block index in
    # slot_mapping. The single remapped slot then lands K at 2*blk and V at
    # 2*blk+1.
    kv_hybrid.zero_()  # zero through the strided alias clears the storage
    nb_phys = kc_h.shape[0]
    kview = torch.as_strided(
        kc_h, (2 * nb_phys, num_kv_heads, block_size, head_size),
        (hidden, block_size * head_size, head_size, 1), 0)
    vview = torch.as_strided(
        vc_h, (2 * nb_phys - 1, num_kv_heads, block_size, head_size),
        (hidden, block_size * head_size, head_size, 1), hidden)
    blk = slot_mapping // block_size
    off = slot_mapping % block_size
    slot_doubled = ((2 * blk) * block_size + off).to(torch.int32)
    ret_fix = ops.reshape_and_cache(
        key, value, kview, vview, slot_doubled,
        k_max=k_max_flat, v_max=v_max_flat, quant_mode=1)
    # Read back via the LOGICAL kc_h / vc_h (what the attention kernel sees).
    got_fix_k = _read_logical(kc_h, slot_mapping, block_size)
    got_fix_v = _read_logical(vc_h, slot_mapping, block_size)
    ref_qv = quant_per_channel(value, amax_v)
    results["fix_ret"] = ret_fix
    results["fix_k"] = (got_fix_k.int() - ref_q.int()).abs().max().item()
    results["fix_v"] = (got_fix_v.int() - ref_qv.int()).abs().max().item()

    return results


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no XPU/CUDA device available")
        return 0
    device = torch.device("cuda:0")
    smoke = os.getenv("SMOKE", "0") == "1"

    cases = [
        # (num_blocks, block_size, num_kv_heads, head_size, num_tokens)
        (8, 16, 4, 64, 20),
    ] if smoke else [
        (8, 16, 4, 64, 20),
        (16, 32, 8, 128, 50),
        (4, 64, 2, 128, 100),
    ]

    failures = 0
    for i, (nb, bs, h, d, nt) in enumerate(cases):
        print(f"{'=' * 64}")
        print(f"Case {i + 1}: num_blocks={nb} block_size={bs} kv_heads={h} "
              f"head_size={d} num_tokens={nt}")
        try:
            res = run_case(nb, bs, h, d, nt, device)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {e}")
            failures += 1
            continue

        print(f"  contiguous int8 max_diff = {res['contiguous']}  "
              f"(EXPECT <=1 -- correct, rounding only)")
        print(f"  hybrid     int8 max_diff = {res['hybrid']}  "
              f"(ret={res['hybrid_ret']}; EXPECT large -- this is the bug)")
        print(f"  FIX (phys view + slot*2): K_diff={res['fix_k']} "
              f"V_diff={res['fix_v']} (ret={res['fix_ret']}; EXPECT <=1 -- fixed)")

        ok = True
        # Path 1 (contiguous) must match the reference up to a single int8 step
        # (the kernel's rounding mode may differ from torch.round by 1 ULP).
        if res["contiguous"] > 1:
            print("  [FAIL] contiguous int8 write should match reference "
                  "(max_diff <= 1)")
            ok = False
        # Path 2 (hybrid block-first) must mis-write -> demonstrates the bug.
        if res["hybrid"] <= 2:
            print("  [FAIL] expected the flat-indexing kernel to corrupt the "
                  "block-first cache, but it matched -- bug not reproduced")
            ok = False
        # Path 3 (the fix) must restore correctness on the block-first layout.
        if res["fix_k"] > 1 or res["fix_v"] > 1:
            print("  [FAIL] fix (physical view + slot doubling) should write "
                  "correct K/V on the block-first layout (max_diff <= 1)")
            ok = False

        print(f"  => {'PASS (bug reproduced, contiguous path correct)' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    print(f"{'=' * 64}")
    print(f"{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
