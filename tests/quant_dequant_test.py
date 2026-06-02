import torch
import kunlun_ops

# ============ 方案A: quant2d + dequant2d_per_token ============
print("=" * 60)
print("方案A: 动态 per-token 量化/反量化精度测试")
print("=" * 60)

num_tokens = 32
num_heads = 8
head_size = 128
device = "cuda:0"
n = num_heads * head_size  # 1024

# 构造随机 bf16 key/value
torch.manual_seed(42)
key_bf16 = torch.randn(num_tokens, n, dtype=torch.bfloat16, device=device)
value_bf16 = torch.randn(num_tokens, n, dtype=torch.bfloat16, device=device)

# 量化 key
key_int8 = torch.empty(num_tokens, n, dtype=torch.int8, device=device)
key_scale = torch.empty(num_tokens, dtype=torch.float32, device=device)
kunlun_ops.quant2d(key_bf16, key_int8, key_scale)

# 量化 value
val_int8 = torch.empty(num_tokens, n, dtype=torch.int8, device=device)
val_scale = torch.empty(num_tokens, dtype=torch.float32, device=device)
kunlun_ops.quant2d(value_bf16, val_int8, val_scale)

# 反量化 key
key_dequant = torch.empty(num_tokens, n, dtype=torch.bfloat16, device=device)
kunlun_ops.dequant2d_per_token(key_int8, key_scale, key_dequant, is_absmax=True)

# 反量化 value
val_dequant = torch.empty(num_tokens, n, dtype=torch.bfloat16, device=device)
kunlun_ops.dequant2d_per_token(val_int8, val_scale, val_dequant, is_absmax=True)

# 精度评估
key_cos = torch.nn.functional.cosine_similarity(
    key_bf16.float().reshape(-1).unsqueeze(0),
    key_dequant.float().reshape(-1).unsqueeze(0)
).item()
key_diff = (key_bf16.float() - key_dequant.float()).abs()

val_cos = torch.nn.functional.cosine_similarity(
    value_bf16.float().reshape(-1).unsqueeze(0),
    val_dequant.float().reshape(-1).unsqueeze(0)
).item()
val_diff = (value_bf16.float() - val_dequant.float()).abs()

print(f"Key - cosine_sim: {key_cos:.6f}, max_diff: {key_diff.max().item():.6f}, mean_diff: {key_diff.mean().item():.6f}")
print(f"Val - cosine_sim: {val_cos:.6f}, max_diff: {val_diff.max().item():.6f}, mean_diff: {val_diff.mean().item():.6f}")
print(f"Key scale range: [{key_scale.min().item():.4f}, {key_scale.max().item():.4f}]")
print(f"Val scale range: [{val_scale.min().item():.4f}, {val_scale.max().item():.4f}]")
print()

# 测试不同分布的数据（模拟实际 attention key/value 的分布）
print("--- 测试正态分布 std=0.1 (更接近实际 KV) ---")
key_small = torch.randn(num_tokens, n, dtype=torch.bfloat16, device=device) * 0.1
key_int8_s = torch.empty_like(key_int8)
key_scale_s = torch.empty_like(key_scale)
kunlun_ops.quant2d(key_small, key_int8_s, key_scale_s)
key_dequant_s = torch.empty_like(key_dequant)
kunlun_ops.dequant2d_per_token(key_int8_s, key_scale_s, key_dequant_s, is_absmax=True)
cos_s = torch.nn.functional.cosine_similarity(
    key_small.float().reshape(-1).unsqueeze(0),
    key_dequant_s.float().reshape(-1).unsqueeze(0)
).item()
diff_s = (key_small.float() - key_dequant_s.float()).abs()
print(f"cosine_sim: {cos_s:.6f}, max_diff: {diff_s.max().item():.6f}, mean_diff: {diff_s.mean().item():.6f}")
print()
print("方案A 测试通过!" if key_cos > 0.99 and val_cos > 0.99 else "方案A 精度不足!")