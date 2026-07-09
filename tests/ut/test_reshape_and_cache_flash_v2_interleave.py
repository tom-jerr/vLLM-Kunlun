# Probe: does reshape_and_cache_flash_v2 support writing int8 quantized KV
# into a HYBRID block-first (interleaved / non-contiguous) cache purely via
# strides (i.e. WITHOUT the physical-view + slot-doubling workaround we need
# for the legacy flat-index reshape_and_cache)?
#
# reshape_and_cache_flash_v2 (xtorch_ops/_xflashinfer_cache.py L1194-1234) is
# documented as "stride-based KV cache write with optional quant". If it truly
# honours strides for an int8 cache, then for the Qwen3.5 hybrid layout we
# could just hand it the interleaved key_cache/value_cache directly and only
# adjust the READ path (block_table*2) -- exactly like the fp16
# reshape_and_cache_flash path -- instead of remapping slots on the write side.
#
# This test answers 4 things:
#   A. Does v2 accept an int8 key_cache at all (quant_mode=3 per-channel)?
#   B. What is the quant formula -- round(x/scale) or round(x*scale)? (scale is
#      4-D [1,1,head_num,head_size] for per-channel.)
#   C. On a CONTIGUOUS cache, does the quantized round-trip match a reference?
#   D. On a NON-CONTIGUOUS block-first (interleaved) cache, does v2 write to the
#      correct physical slots (stride-aware) -- the whole point.
#
# Run (XPU/CUDA device; uses xtorch_ops, API identical to kunlun_ops):
#     python tests/ut/test_reshape_and_cache_flash_v2_interleave.py
#
import os

import torch

import xtorch_ops as ops


def _read_logical(cache_int8, slot_mapping, block_size):
    """Read per-token int8 vectors from a logical
    [num_blocks, block_size, num_heads, head_size] (BLHD) cache."""
    out = []
    for slot in slot_mapping.tolist():
        blk, off = slot // block_size, slot % block_size
        out.append(cache_int8[blk, off, :, :].cpu())  # [H, D]
    return torch.stack(out, dim=0)  # [num_tokens, H, D]


def _ref_quant(x, amax, mode):
    """Reference per-channel int8 quant. amax/scale: [H, D].
    mode='div'  -> q = round(x / (amax/127))
    mode='mul'  -> q = round(x * scale)  where scale is passed AS the value fed
                   to the kernel (we try scale = 127/amax for this variant).
    """
    x = x.float().cpu()
    if mode == "div":
        sc = (amax / 127.0).clamp_min(1e-8).cpu()
        return torch.round(x / sc).clamp_(-127, 127).to(torch.int8)
    else:  # mul
        sc = (127.0 / amax.clamp_min(1e-8)).cpu()
        return torch.round(x * sc).clamp_(-127, 127).to(torch.int8)


def build_scale_4d(amax, mode):
    """Kernel wants per-channel scale as [1, 1, head_num, head_size]."""
    if mode == "div":
        val = (amax / 127.0).clamp_min(1e-8)      # scale = amax/127
    else:
        val = (127.0 / amax.clamp_min(1e-8))      # scale = 127/amax
    return val.reshape(1, 1, amax.shape[0], amax.shape[1]).contiguous()


def run(num_blocks, block_size, num_heads, head_size, num_tokens, device):
    torch.manual_seed(0)
    key = torch.randn(num_tokens, num_heads, head_size,
                      dtype=torch.float16, device=device)
    value = torch.randn(num_tokens, num_heads, head_size,
                        dtype=torch.float16, device=device)
    slot = torch.arange(num_tokens, dtype=torch.int32, device=device)
    amax_k = key.float().abs().amax(dim=0).clamp_min(1e-8)  # [H, D]
    amax_v = value.float().abs().amax(dim=0).clamp_min(1e-8)

    QUANT_MODE = 3  # per-channel, per v2 docstring
    results = {}

    # ------------------------------------------------------------------ #
    # A + B + C: contiguous BLHD cache, try both quant formulas.
    # v2 doc: key_cache BHLD [num_block, head_num, block_size, head_size]
    #   -- NOTE the doc's "Returns" says BHLD but lists block_size before
    #   head_size in text; we test the [nb, bs, H, D] (BLHD) read layout that
    #   the fp16 reshape_and_cache_flash(BLHD_LAYOUT=...) uses, matching how
    #   kunlun_attn stores it. Adjust if the kernel rejects.
    # ------------------------------------------------------------------ #
    for mode in ("div", "mul"):
        kc = torch.zeros(num_blocks, block_size, num_heads, head_size,
                         dtype=torch.int8, device=device)
        vc = torch.zeros_like(kc)
        sk = build_scale_4d(amax_k, mode).to(device)
        sv = build_scale_4d(amax_v, mode).to(device)
        try:
            ret = ops.reshape_and_cache_flash_v2(
                key, value, kc, vc, slot,
                k_scale=sk, v_scale=sv, quant_mode=QUANT_MODE)
            got = _read_logical(kc, slot, block_size)
            ref = _ref_quant(key, amax_k, mode)
            diff = (got.int() - ref.int()).abs().max().item()
            results[f"contig_{mode}"] = (f"ret={ret}", f"max_diff={diff}")
        except Exception as e:  # noqa: BLE001
            results[f"contig_{mode}"] = ("RAISED", repr(e))

    # ------------------------------------------------------------------ #
    # D: hybrid block-first interleaved (non-contiguous) cache.
    # Reproduce _update_hybrid_attention_mamba_layout for a BLHD cache:
    # joint (2, nb, bs, H, D) re-strided so K/V interleave per block.
    # If v2 is stride-aware, feeding the strided kc/vc directly should write
    # correctly with the ORIGINAL slot (no doubling) -- that is the test.
    # ------------------------------------------------------------------ #
    hidden = block_size * num_heads * head_size
    kv = torch.zeros(2, num_blocks, block_size, num_heads, head_size,
                     dtype=torch.int8, device=device)
    kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
    kc_h, vc_h = kv[0], kv[1]
    assert not kc_h.is_contiguous(), "hybrid key_cache must be non-contiguous"

    # pick the formula that worked on the contiguous path if any; else try div.
    mode = "div"
    for m in ("div", "mul"):
        r = results.get(f"contig_{m}")
        if r and isinstance(r[1], str) and r[1].startswith("max_diff") \
                and int(r[1].split("=")[1]) <= 1:
            mode = m
            break
    sk = build_scale_4d(amax_k, mode).to(device)
    sv = build_scale_4d(amax_v, mode).to(device)
    try:
        ret = ops.reshape_and_cache_flash_v2(
            key, value, kc_h, vc_h, slot,
            k_scale=sk, v_scale=sv, quant_mode=QUANT_MODE)
        got_k = _read_logical(kc_h, slot, block_size)
        got_v = _read_logical(vc_h, slot, block_size)
        ref_k = _ref_quant(key, amax_k, mode)
        ref_v = _ref_quant(value, amax_v, mode)
        dk = (got_k.int() - ref_k.int()).abs().max().item()
        dv = (got_v.int() - ref_v.int()).abs().max().item()
        results["hybrid_stride"] = (f"ret={ret} mode={mode}",
                                    f"K_diff={dk} V_diff={dv}")
    except Exception as e:  # noqa: BLE001
        results["hybrid_stride"] = ("RAISED", repr(e))

    return results


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no XPU/CUDA device available")
        return 0
    device = torch.device("cuda:0")
    cases = [
        (8, 16, 4, 64, 20),
        (16, 32, 8, 128, 50),
    ]
    for i, (nb, bs, h, d, nt) in enumerate(cases):
        print(f"{'=' * 66}")
        print(f"Case {i + 1}: nb={nb} bs={bs} heads={h} head_size={d} tokens={nt}")
        res = run(nb, bs, h, d, nt, device)
        for k, v in res.items():
            print(f"  {k:16s}: {v[0]:24s} {v[1]}")
    print(f"{'=' * 66}")
    print("Interpretation:")
    print("  - contig_* max_diff<=1  => v2 supports int8 per-channel quant, "
          "and that formula (div/mul) is correct.")
    print("  - hybrid_stride K/V_diff<=1 => v2 IS stride-aware for int8 "
          "-> we can drop the slot-doubling workaround and only fix the READ "
          "path (block_table*2), like fp16.")
    print("  - hybrid_stride large/RAISED => v2 is NOT stride-aware for int8 "
          "-> keep the physical-view + slot-doubling fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
