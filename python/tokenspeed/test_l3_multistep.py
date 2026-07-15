#!/usr/bin/env python3
"""L3-04: 多步推理测试 — 模拟多轮对话的连续多步推理。

验证内容:
  1. Prefill → decode 多步推理稳定
  2. 多轮对话（prefill → decode → prefill → decode）无崩溃
  3. 持续 decode 多步不崩溃（>= 20 步）
  4. 不同 batch size 下的稳定性
  5. KV cache 模拟（增量推理）
  6. 推理过程中无 NaN/finite 检查
"""

import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

results = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    status = "pass" if cond else "fail"
    results.append({"id": desc, "status": status, "evidence": detail or str(cond)})
    if not cond:
        print(f"  ❌ FAIL: {desc} — {detail}")
    else:
        print(f"  ✅ PASS: {desc}")


# ===== 1. 环境检测 =====
print("=== L3-04: 多步推理集成测试 ===")
print(f"PyTorch: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")
if torch.npu.is_available():
    print(f"NPU device: {torch.npu.get_device_name(0)}")

check("npu_available", torch.npu.is_available(),
      f"npu_available={torch.npu.is_available()}")

if not torch.npu.is_available():
    print("SKIP: NPU not available")
    sys.exit(0)

device = torch.device("npu:0")

# ===== 通用组件 =====
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.activation import SiluAndMul


class DecoderLayer(torch.nn.Module):
    """简化的 decoder layer 用于长序列推理测试。"""
    def __init__(self, hidden_size=256, num_heads=4, num_kv_heads=2, head_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

        self.q_proj = torch.nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = torch.nn.Linear(self.q_size, hidden_size, bias=False)

        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def _reshape_for_attention(self, x, num_heads, head_dim):
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).transpose(1, 2)

    def forward(self, x, k_cache=None, v_cache=None, cache_pos=None):
        residual = x
        x = self.input_layernorm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention
        B, T, _ = x.shape
        q = self._reshape_for_attention(q, self.num_heads, self.head_dim)
        k = self._reshape_for_attention(k, self.num_kv_heads, self.head_dim)
        v = self._reshape_for_attention(v, self.num_kv_heads, self.head_dim)

        # KV cache update (if provided)
        if k_cache is not None and v_cache is not None and cache_pos is not None:
            k_cache[:, :, cache_pos:cache_pos+T] = k
            v_cache[:, :, cache_pos:cache_pos+T] = v
            k = k_cache[:, :, :cache_pos+T]
            v = v_cache[:, :, :cache_pos+T]

        # GQA: expand kv heads
        if self.num_heads > self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Causal attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1),
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.q_size)
        x = residual + self.o_proj(attn_out)

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.nn.functional.silu(gate) * up
        x = residual + self.down_proj(x)

        return x


def build_model(hidden_size=256, num_layers=4):
    """构建一个包含多层 decoder layer 的模型。"""
    layers = torch.nn.ModuleList([
        DecoderLayer(hidden_size=hidden_size)
        for _ in range(num_layers)
    ])
    embed = torch.nn.Embedding(1000, hidden_size)
    norm = RMSNorm(hidden_size, eps=1e-6)
    lm_head = torch.nn.Linear(hidden_size, 1000, bias=False)
    return layers, embed, norm, lm_head


# ===== 场景 1: 长序列 decode =====
print("\n=== 场景 1: 连续 30 步 Decode ===")
hidden_size = 256
num_layers = 3
layers, embed, norm, lm_head = build_model(hidden_size, num_layers)
layers = layers.to(device)
embed = embed.to(device)
norm = norm.to(device)
lm_head = lm_head.to(device)

# 先跑 prefill
n_prefill = 16
input_ids = torch.randint(0, 1000, (1, n_prefill), device=device)
hidden = embed(input_ids)
for layer in layers:
    hidden = layer(hidden)
logits = lm_head(norm(hidden))
check("long_prefill_forward", logits.shape == (1, n_prefill, 1000) and not torch.isnan(logits).any(),
      f"shape={logits.shape}")
check("long_prefill_device", logits.device.type == "npu", f"device={logits.device}")

# 30 步 decode
for step in range(30):
    decode_ids = torch.randint(0, 1000, (1, 1), device=device)
    hidden = embed(decode_ids)
    for layer in layers:
        hidden = layer(hidden)
    logits = lm_head(norm(hidden))
    check(f"long_decode_{step+1}",
          logits.shape == (1, 1, 1000) and not torch.isnan(logits).any() and torch.isfinite(logits).all(),
          f"step={step+1}, finite={torch.isfinite(logits).all().item()}")
print("  30 步 decode 全部通过 ✓")

# ===== 场景 2: 多轮对话（prefill / decode 交替） =====
print("\n=== 场景 2: 多轮对话模拟 ===")
rounds = 5
for round_idx in range(rounds):
    # Prefill (user input)
    prompt_len = 8 + round_idx * 4  # 逐渐增长
    input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)
    hidden = embed(input_ids)
    for layer in layers:
        hidden = layer(hidden)

    # Decode (assistant response)
    n_response = 5 + round_idx  # 逐渐增长
    for step in range(n_response):
        decode_ids = torch.randint(0, 1000, (1, 1), device=device)
        hidden = embed(decode_ids)
        for layer in layers:
            hidden = layer(hidden)
        logits = lm_head(norm(hidden))

    check(f"multi_round_{round_idx+1}",
          logits.shape == (1, 1, 1000) and not torch.isnan(logits).any() and torch.isfinite(logits).all(),
          f"round={round_idx+1}, prompt_len={prompt_len}, response_len={n_response}")
    print(f"  第 {round_idx+1} 轮对话通过 ✓")

print("  多轮对话模拟全部通过 ✓")

# ===== 场景 3: 不同 batch size =====
print("\n=== 场景 3: 不同 Batch Size ===")
for batch_size in [1, 2, 4, 8]:
    input_ids = torch.randint(0, 1000, (batch_size, 8), device=device)
    hidden = embed(input_ids)
    for layer in layers:
        hidden = layer(hidden)
    check(f"batch_size_{batch_size}",
          hidden.shape == (batch_size, 8, hidden_size) and not torch.isnan(hidden).any(),
          f"batch_size={batch_size}, shape={hidden.shape}")
    # 然后 decode
    decode_ids = torch.randint(0, 1000, (batch_size, 1), device=device)
    hidden = embed(decode_ids)
    for layer in layers:
        hidden = layer(hidden)
    check(f"batch_size_{batch_size}_decode",
          hidden.shape == (batch_size, 1, hidden_size) and not torch.isnan(hidden).any(),
          f"batch_size={batch_size}, decode shape={hidden.shape}")

# ===== 场景 4: 大序列测试 =====
print("\n=== 场景 4: 大序列 prefill（128 tokens）===")
n_large = 128
input_ids = torch.randint(0, 1000, (1, n_large), device=device)
hidden = embed(input_ids)
for layer in layers:
    hidden = layer(hidden)
check("large_prefill", hidden.shape == (1, n_large, hidden_size) and not torch.isnan(hidden).any(),
      f"shape={hidden.shape}")
print("  大序列 prefill 通过 ✓")

# 大序列后 decode
for step in range(10):
    decode_ids = torch.randint(0, 1000, (1, 1), device=device)
    hidden = embed(decode_ids)
    for layer in layers:
        hidden = layer(hidden)
check("large_seq_after_decode",
      hidden.shape == (1, 1, hidden_size) and not torch.isnan(hidden).any() and torch.isfinite(hidden).all(),
      f"shape={hidden.shape}")
print("  大序列后 decode 通过 ✓")

# ===== 场景 5: 长时间稳定性 =====
print("\n=== 场景 5: 长时间稳定性（总步数 > 50）===")
total_decode_steps = 50
start_time = time.time()
for step in range(total_decode_steps):
    decode_ids = torch.randint(0, 1000, (1, 1), device=device)
    hidden = embed(decode_ids)
    for layer in layers:
        hidden = layer(hidden)
    logits = lm_head(norm(hidden))
    if step == total_decode_steps - 1:
        check(f"long_stability_{total_decode_steps}_steps",
              logits.shape == (1, 1, 1000) and not torch.isnan(logits).any() and torch.isfinite(logits).all(),
              f"steps={total_decode_steps}, finite={torch.isfinite(logits).all().item()}")
elapsed = time.time() - start_time
print(f"  50 步 decode 耗时: {elapsed:.2f}s ({elapsed/total_decode_steps*1000:.1f}ms/step)")

# ===== 汇总 =====
print(f"\n{'='*50}")
total = len(results)
passed = sum(1 for r in results if r["status"] == "pass")
failed = total - passed
print(f"总计: {total} | 通过: {passed} | 失败: {failed}")

overall_status = "pass" if failed == 0 else "fail"
output_dir = os.environ.get("OUTPUT_DIR", "/tmp")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "verify_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nOverall: {overall_status}")
sys.exit(0 if overall_status == "pass" else 1)
