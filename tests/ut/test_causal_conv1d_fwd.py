# Portions of this code are derived from:
#
# FlashInfer: https://github.com/flashinfer-ai/flashinfer
# Copyright (c) 2024 by FlashInfer team.
"""Unit test for kunlun_ops.causal_conv1d_fwd (new _fusion.py API, prefill).

Verifies the correct calling convention for the new in-place prefill API:
    causal_conv1d_fwd(x, weight, bias=None, conv_states=None,
                      query_start_loc=None, cache_indices=None,
                      has_initial_state=None, silu_activation=True,
                      is_ncw=False, query_start_loc_cpu=None,
                      pad_slot_id=-1) -> int
"""

import torch
import torch.nn.functional as F
import kunlun_ops

PAD_SLOT_ID = -1


def causal_conv1d_fwd_torch(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
):
    """CPU reference (NCW). x: (dim, cu_seqlen); conv_states: (N, dim, state_len)."""
    dim, _ = x.shape
    _, width = weight.shape
    K = width
    state_len = K - 1
    device = "cpu"
    dtype = x.dtype

    x = x.cpu()
    weight = weight.cpu()
    bias = bias.cpu() if bias is not None else None
    conv_states = conv_states.cpu() if conv_states is not None else None
    query_start_loc = query_start_loc.cpu()
    cache_indices = cache_indices.cpu() if cache_indices is not None else None
    has_initial_state = (has_initial_state.cpu()
                         if has_initial_state is not None else None)

    bias_f32 = bias.to(torch.float32) if bias is not None else None

    out = torch.zeros_like(x)
    batch = query_start_loc.numel() - 1

    for i in range(batch):
        if cache_indices is not None:
            cache_line = cache_indices[i].item()
            if cache_line == pad_slot_id:
                continue
        else:
            cache_line = i

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        L = end - start
        if L <= 0:
            continue

        x_seq = x[:, start:end]

        if state_len > 0:
            if (conv_states is not None
                    and has_initial_state is not None
                    and bool(has_initial_state[i].item())):
                prior = conv_states[cache_line, :, :state_len]
            else:
                prior = torch.zeros((dim, state_len), device=device, dtype=dtype)
            inputs = torch.cat([prior, x_seq], dim=1)
        else:
            inputs = x_seq

        if bias_f32 is not None:
            acc = bias_f32.unsqueeze(1).repeat(1, L)
        else:
            acc = torch.zeros((dim, L), device=device, dtype=torch.float32)

        for m in range(K):
            w = weight[:, m].unsqueeze(1)
            val = inputs[:, m:m + L]
            prod = val * w
            acc = acc + prod.to(torch.float32)

        if isinstance(activation, bool) and activation:
            activation = "silu"
        if activation in ("silu", "swish"):
            y_seq = F.silu(acc)
        else:
            y_seq = acc

        out[:, start:end] = y_seq.to(dtype)

        if conv_states is not None:
            if cache_line == pad_slot_id:
                pass
            elif state_len > 0:
                tgt = conv_states[cache_line, :, :state_len]
                if state_len <= L:
                    new_state = x_seq[:, L - state_len: L]
                else:
                    if (has_initial_state is not None) and bool(
                            has_initial_state[i].item()):
                        left = tgt[:, L:]
                    else:
                        left = torch.zeros(
                            (dim, state_len - L), device=device, dtype=dtype
                        )
                    new_state = torch.cat([left, x_seq], dim=1)
                tgt.copy_(new_state)

    return out


def test_causal_conv1d_fwd(batch,
                           seq_lens,
                           dim,
                           width,
                           dtype,
                           device,
                           num_cache_lines,
                           cache_indices,
                           has_initial_state,
                           silu_activation=True,
                           use_bias=False):
    assert len(seq_lens) == batch

    query_start_loc_list = [0]
    for sl in seq_lens:
        query_start_loc_list.append(query_start_loc_list[-1] + sl)
    total_tokens = query_start_loc_list[-1]

    query_start_loc = torch.tensor(
        query_start_loc_list, dtype=torch.int32, device=device)

    # NCW layout: x is (dim, cu_seqlen) for reference,
    # (cu_seqlen, dim) for kunlun_ops (is_ncw=False)
    x_cpu = torch.randn(dim, total_tokens, dtype=dtype)
    x_ref = x_cpu.clone().to(device)
    x_xpu = x_cpu.permute(1, 0).contiguous().clone().to(device)

    weight = torch.randn(dim, width, dtype=dtype, device=device)

    bias = None
    if use_bias:
        bias = torch.randn(dim, dtype=torch.float32, device=device)

    state_width = width - 1
    conv_states_ref = torch.randn(
        num_cache_lines, dim, state_width, dtype=dtype, device=device)
    conv_states_xpu = conv_states_ref.permute(0, 2, 1).contiguous().clone()

    # Reference (NCW)
    out_ref = causal_conv1d_fwd_torch(
        x_ref, weight, bias, conv_states_ref,
        query_start_loc, cache_indices, has_initial_state, silu_activation)

    # kunlun_ops (NWC) — mutates x_xpu in place.
    ret = kunlun_ops.causal_conv1d_fwd(
        x=x_xpu, weight=weight, bias=bias, conv_states=conv_states_xpu,
        query_start_loc=query_start_loc, cache_indices=cache_indices,
        has_initial_state=has_initial_state, silu_activation=silu_activation,
        is_ncw=False, query_start_loc_cpu=query_start_loc_list, pad_slot_id=-1)
    assert ret == 0, f"causal_conv1d_fwd failed with ret={ret}"

    # Compare: out_ref is (dim, cu_seqlen), x_xpu is (cu_seqlen, dim)
    out_xpu_ncw = x_xpu.permute(1, 0)

    if dtype == torch.bfloat16:
        rtol = 5e-2
        atol = 5e-2
    elif dtype == torch.float16:
        rtol = 5e-2
        atol = 5e-2
    else:
        rtol = 3e-3
        atol = 3e-3

    max_diff = (out_ref.float() - out_xpu_ncw.float().cpu()).abs().max().item()
    try:
        assert torch.allclose(out_ref.float(),
                              out_xpu_ncw.float().cpu(), atol=atol, rtol=rtol)
        print("PASS")
    except AssertionError:
        print("FAILED diff_out_max=%f" % max_diff)
        raise


if __name__ == "__main__":
    print("test_causal_conv1d_fwd.py : ")
    device = torch.device('cuda:0')
    test_cases = [
        # (batch, seq_lens, dim, width, dtype, num_cache_lines,
        #  cache_indices, has_initial_state, silu, bias)
        (1, [5], 8192, 4, torch.bfloat16, 494,
         torch.tensor([1], dtype=torch.int32, device=device),
         torch.tensor([False], device=device), True, False),
        (1, [5], 8192, 4, torch.float16, 494,
         torch.tensor([1], dtype=torch.int32, device=device),
         torch.tensor([False], device=device), True, False),
        (1, [5], 8192, 4, torch.float32, 494,
         torch.tensor([1], dtype=torch.int32, device=device),
         torch.tensor([False], device=device), True, False),
        (4, [1, 4, 6, 10], 8192, 4, torch.bfloat16, 494,
         torch.tensor([2, 6, 10, 14], dtype=torch.int32, device=device),
         torch.tensor([True, False, False, False], device=device), True, False),
        (4, [1, 4, 6, 10], 8192, 4, torch.float16, 494,
         torch.tensor([2, 6, 10, 14], dtype=torch.int32, device=device),
         torch.tensor([True, False, False, False], device=device), True, False),
        (4, [1, 4, 6, 10], 8192, 4, torch.float, 494,
         torch.tensor([2, 6, 10, 14], dtype=torch.int32, device=device),
         torch.tensor([True, False, False, False], device=device), True, False),
    ]

    passed = 0
    failed = 0
    for i, (batch, seqs, dim, w, dt, mc, ci, his, silu, bias) in enumerate(
            test_cases):
        print(f"{'=' * 60}")
        print(f"Test {i+1}: batch={batch}, seqs={seqs}, dim={dim}, "
              f"w={w}, dtype={dt}, max_reqs={mc}, cache_indices={ci}, "
              f"has_initial_state={his},silu={silu}, bias={bias}")
        print(f"{'=' * 60}")
        try:
            test_causal_conv1d_fwd(batch, seqs, dim, w, dt, device, mc, ci,
                                   his, silu, bias)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")
    if failed > 0:
        exit(1)
