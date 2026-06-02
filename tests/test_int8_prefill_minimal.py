"""
Minimal diagnostic: constant K/V, single token/head, small head_dim.
Tests prefill_attention, paged_attention, speculative_attention with int8 cache.
If int8 still wrong here, it's layout/scale/branch, not softmax/mask.
"""
import torch
import kunlun_ops

device = "cuda:0"
dtype = torch.bfloat16


def check(name, out, baseline):
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1).unsqueeze(0),
        baseline.float().reshape(-1).unsqueeze(0),
    ).item()
    max_diff = (out.float() - baseline.float()).abs().max().item()
    status = "OK" if cos > 0.99 else "FAIL"
    print(f"  [{status}] {name}: cos={cos:.6f}, max_diff={max_diff:.4f}")
    return status == "OK"


for num_tokens, num_heads, head_dim, block_size in [
    (1, 1, 8, 4),
    (1, 1, 16, 16),
    (2, 1, 8, 4),
    (4, 2, 128, 128),
]:
    print(f"\n{'='*60}")
    print(f"num_tokens={num_tokens}, num_heads={num_heads}, head_dim={head_dim}, block_size={block_size}")

    num_q_heads = num_heads  # no GQA for simplicity

    q = torch.ones(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.full((num_tokens, num_heads, head_dim), 1.0, dtype=dtype, device=device)
    v = torch.full((num_tokens, num_heads, head_dim), 2.0, dtype=dtype, device=device)

    qlen_lod = torch.tensor([0, num_tokens], dtype=torch.int32)
    qlen_lod_xpu = qlen_lod.to(device)

    # Baseline: bf16 direct
    out_baseline = torch.zeros(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    kunlun_ops.prefill_attention(
        q=q, k=k, v=v, out=out_baseline, is_causal=True,
        context_qlen_lod_cpu=qlen_lod, context_qlen_lod_xpu=qlen_lod_xpu,
    )
    print(f"  baseline out[0,:4]: {out_baseline[0, 0, :4].tolist()}")

    # Int8 cache setup
    k_max = torch.tensor([1.0] * num_heads, dtype=torch.float32, device=device)
    v_max = torch.tensor([2.0] * num_heads, dtype=torch.float32, device=device)

    num_blocks = (num_tokens + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, dtype=torch.int8, device=device)
    value_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, dtype=torch.int8, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)

    kunlun_ops.reshape_and_cache(k, v, key_cache, value_cache, slot_mapping, k_max=k_max, v_max=v_max, quant_mode=0)

    print(f"  key_cache[0,0,0,:4]: {key_cache[0, 0, 0, :4].tolist()} (expect ~127)")
    print(f"  value_cache[0,0,0,:4]: {value_cache[0, 0, 0, :4].tolist()} (expect ~127)")

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    # Per-channel scale: [num_heads, head_dim], contiguous
    k_scale_pc = (k_max).view(num_heads, 1).expand(num_heads, head_dim).contiguous()
    v_scale_pc = (v_max).view(num_heads, 1).expand(num_heads, head_dim).contiguous()

    # --- Test 1: prefill_attention with int8 cache ---
    kvlen_lod = torch.tensor([0, num_tokens], dtype=torch.int32)
    kvlen_lod_xpu = kvlen_lod.to(device)

    out_prefill = torch.zeros(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    kunlun_ops.prefill_attention(
        q=q, k=key_cache, v=value_cache, out=out_prefill,
        is_causal=True, is_prefix_cache=True,
        block_table=block_table,
        context_qlen_lod_cpu=qlen_lod, context_qlen_lod_xpu=qlen_lod_xpu,
        context_kvlen_lod_cpu=kvlen_lod, context_kvlen_lod_xpu=kvlen_lod_xpu,
        k_perchannel_scale=k_scale_pc, v_perchannel_scale=v_scale_pc,
    )
    print(f"  prefill int8 out[0,:4]: {out_prefill[0, 0, :4].tolist()} (expect ~2.0)")
    check("prefill_attention int8 per-channel", out_prefill, out_baseline)

    # --- Test 2: paged_attention with int8 cache ---
    # paged_attention is decode-style: 1 query token per sequence, needs context_lens
    context_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
    context_lens_xpu = context_lens_cpu.to(device)
    # query for decode: single token
    q_decode = torch.ones(1, num_q_heads, head_dim, dtype=dtype, device=device)

    out_paged = torch.zeros(1, num_q_heads, head_dim, dtype=dtype, device=device)
    kunlun_ops.paged_attention(
        x=q_decode,
        k_cache=key_cache,
        v_cache=value_cache,
        block_tables=block_table,
        context_lens_cpu=context_lens_cpu,
        context_lens_xpu=context_lens_xpu,
        is_context=False,
        is_causal=True,
        out=out_paged,
        vo_head_dim=head_dim,
        k_perchannel_scale=k_scale_pc,
        v_perchannel_scale=v_scale_pc,
    )
    # Baseline for decode: attend over all num_tokens cached KV, expect ~2.0
    print(f"  paged_attn out[0,:4]: {out_paged[0, 0, :4].tolist()} (expect ~2.0)")
    # For constant V=2.0, any correct attention gives out=2.0
    expected_paged = torch.full((1, num_q_heads, head_dim), 2.0, dtype=dtype, device=device)
    check("paged_attention int8 per-channel", out_paged, expected_paged)

    # --- Test 3: speculative_attention with int8 cache ---
    qlen_spec = 1
    out_spec = torch.zeros(1, qlen_spec, num_q_heads, head_dim, dtype=dtype, device=device)
    q_spec = torch.ones(1, qlen_spec, num_q_heads, head_dim, dtype=dtype, device=device)

    try:
        kunlun_ops.speculative_attention(
            out=out_spec,
            q=q_spec,
            k_cache=key_cache,
            v_cache=value_cache,
            context_lens_cpu=context_lens_cpu,
            context_lens_xpu=context_lens_xpu,
            batch_num=1,
            qlen=qlen_spec,
            max_context_len=131072,
            head_num=num_q_heads,
            head_dim=head_dim,
            scale=0.0,
            kv_head_num=num_heads,
            block_size=block_size,
            max_num_blocks_per_seq=block_table.shape[1],
            block_tables=block_table,
            k_perchannel_scale=k_scale_pc,
            v_perchannel_scale=v_scale_pc,
        )
        print(f"  spec_attn out[0,:4]: {out_spec[0, 0, 0, :4].tolist()} (expect ~2.0)")
        expected_spec = torch.full((1, qlen_spec, num_q_heads, head_dim), 2.0, dtype=dtype, device=device)
        check("speculative_attention int8 per-channel", out_spec, expected_spec)
    except ValueError as e:
        print(f"  [SKIP] speculative_attention: {e} (unsupported config)")
