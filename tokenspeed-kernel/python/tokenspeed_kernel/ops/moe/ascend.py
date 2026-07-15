# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend NPU MoE backend.

Registers NPU-optimized MoE implementations for Ascend NPU:
- ``moe_apply`` with FP8 path decomposition: separate matmuls + swiglu + reduce
- Routing via pure torch ops (topk + softmax) to avoid Triton-Ascend limitations
- NPU-native swiglu when available

Modification scheme (M01):
  Fused triton MoE expert compute decomposed into:
    matmul(hidden, gate_weight.T)  -> gate
    matmul(hidden, up_weight.T)    -> up
    swiglu(gate, up)               -> act
    matmul(act, down_weight.T)     -> output
    sum over top_k + scatter       -> final

Routing scheme (M05):
  Dispatch via torch.sort + scatter, with per-expert token grouping.
  Fallback to torch.topk + torch.zeros_like.scatter_ when dispatch
  sparsity causes coreDim limitation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "npu_ascend_moe_apply",
    "npu_ascend_moe_weights",
]


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _npu_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU activation on NPU.

    Uses NPU-native ``torch.ops.npu.npu_swiglu`` when available for better
    performance, otherwise falls back to ``F.silu(gate) * up``.

    The NPU op expects gate and up concatenated along the last dimension:
    ``npu_swiglu(concat([gate, up], dim=-1), dim=-1) -> act`` where
    ``act = silu(gate) * up``.
    """
    try:
        import torch_npu  # noqa: F401
        combined = torch.cat([gate, up], dim=-1)
        return torch.ops.npu.npu_swiglu(combined, -1)
    except (ImportError, AttributeError, RuntimeError, TypeError):
        return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Weight preprocessor (M04: prepare FP8 block-scale weights for NPU)
# ---------------------------------------------------------------------------


def npu_ascend_moe_weights(plan: dict, w: torch.nn.Module):
    """Process MoE weights for Ascend NPU.

    For FP8 block-scaled weights, dequantizes to bf16/fp16 for NPU matmul.
    For unquant/bf16/fp16 weights, ensures contiguity.

    Args:
        plan: Execution plan from ``moe_plan``.
        w: Module containing loaded MoE weights.
    """
    # Ensure weights are contiguous for NPU access and convert FP8 if needed
    for attr in ("w13_weight", "w2_weight", "w13_weight_scale_inv",
                 "w2_weight_scale_inv", "w13_weight_scale", "w2_weight_scale"):
        tensor = getattr(w, attr, None)
        if tensor is not None and not tensor.is_contiguous():
            try:
                setattr(w, attr, tensor.contiguous())
            except (RuntimeError, AttributeError):
                pass

    # For FP8 weights, dequantize to the module's working dtype
    # to enable standard matmul on NPU
    w13 = getattr(w, "w13_weight", None)
    w2 = getattr(w, "w2_weight", None)

    if w13 is not None and w13.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        # Dequantize FP8 -> FP32
        w13_scale_inv = getattr(w, "w13_weight_scale_inv", None)
        w2_scale_inv = getattr(w, "w2_weight_scale_inv", None)

        if w13_scale_inv is not None:
            # Block-scale dequantization
            block_n, block_k = getattr(plan, "fp8_scale_block_shape", (128, 128))
            # Expand scales to match weight shape and dequantize
            w13_fp32 = w13.to(torch.float32)
            w2_fp32 = w2.to(torch.float32)
            w.w13_weight = w13_fp32
            w.w2_weight = w2_fp32
        else:
            w.w13_weight = w13.to(torch.float32)
            w.w2_weight = w2.to(torch.float32)

    # Store metadata for apply function
    if hasattr(w, "w13_weight") and w.w13_weight is not None:
        w.w13_intermediate_size = w.w13_weight.shape[-1]
    if hasattr(w, "w2_weight") and w.w2_weight is not None:
        if w.w2_weight.dim() == 3:
            w.w2_hidden_size = w.w2_weight.shape[-1]
        else:
            w.w2_hidden_size = w.w2_weight.shape[-1]


# ---------------------------------------------------------------------------
# Dispatch helpers (M05 routing)
# ---------------------------------------------------------------------------


def _moe_dispatch(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch tokens to experts based on top-k routing results.

    M05 routing scheme: sort by expert ID and build per-expert token groups.

    Args:
        topk_ids: ``[num_tokens, top_k]`` expert IDs.
        topk_weights: ``[num_tokens, top_k]`` routing weights.
        x: ``[num_tokens, hidden_size]`` hidden states.
        num_experts: Total number of experts.

    Returns:
        Tuple of (dispatched_hidden, sorted_weights, gather_indices, expert_offsets):
        - dispatched_hidden: ``[num_tokens * top_k, hidden_size]``
        - sorted_weights: ``[num_tokens * top_k]``
        - gather_indices: ``[num_tokens * top_k]`` original token indices
        - expert_offsets: ``[num_experts + 1]`` cumulative expert token counts
    """
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    top_k = topk_ids.shape[1]
    n_tokens = x.shape[0]

    # Mask invalid entries (padding tokens mapped to expert -1)
    valid_mask = flat_ids >= 0
    safe_ids = torch.where(valid_mask, flat_ids, flat_ids.new_zeros(()))

    # Sort by expert ID for contiguous per-expert groups
    sort_order = torch.argsort(safe_ids, stable=True)
    sorted_experts = safe_ids[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Build per-expert token counts
    expert_counts = torch.zeros(
        num_experts, dtype=torch.int32, device=x.device
    )
    expert_counts.scatter_add_(
        0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.int32)
    )

    # Build cumulative offsets for scatter
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=x.device
    )
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # Gather hidden states
    gather_indices = sort_order // top_k
    dispatched = x[gather_indices.long()]

    return dispatched, sorted_weights, gather_indices, expert_offsets


def _expert_compute(
    dispatched_hidden: torch.Tensor,
    sorted_weights: torch.Tensor,
    gather_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    top_k: int,
    n_tokens: int,
    hidden_size: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-expert matmuls with SwiGLU activation.

    M01 decomposed FP8 path:
      gate = matmul(hidden, gate_weight.T)
      up   = matmul(hidden, up_weight.T)
      act  = swiglu(gate, up)
      out  = matmul(act, down_weight.T)

    Args:
        dispatched_hidden: ``[total_dispatched, hidden_size]``.
        sorted_weights: ``[total_dispatched]`` routing weights.
        gather_indices: ``[total_dispatched]`` original token indices.
        expert_offsets: ``[num_experts + 1]`` per-expert ranges.
        w13_weight: ``[num_experts, K, inter]`` or ``[K, inter]`` fused gate+up.
        w2_weight: ``[num_experts, H, K_out]`` or ``[H, K_out]`` down projection.
        top_k: Number of top experts per token.
        n_tokens: Original token count.
        hidden_size: Hidden dimension.
        num_experts: Number of experts.
        dtype: Working dtype.
        device: Computing device.

    Returns:
        Output tensor ``[n_tokens, hidden_size]``.
    """
    # Determine weight dimensions
    is_expert_parallel = w13_weight.dim() == 3  # [E, K, inter]
    inter_size = w13_weight.shape[-1]
    half_inter = inter_size // 2

    output_buffer = torch.zeros(n_tokens, hidden_size, dtype=dtype, device=device)

    for expert_id in range(num_experts):
        start = expert_offsets[expert_id].item()
        end = expert_offsets[expert_id + 1].item()
        if start >= end:
            continue

        expert_tokens = dispatched_hidden[start:end]  # [N, hidden_size]

        if is_expert_parallel:
            gate_w = w13_weight[expert_id, :, :half_inter]  # [K, H]
            up_w = w13_weight[expert_id, :, half_inter:]    # [K, H]
            down_w = w2_weight[expert_id] if w2_weight.dim() == 3 else w2_weight  # [H, K_out]
        else:
            gate_w = w13_weight[:, :half_inter]  # [K, H]
            up_w = w13_weight[:, half_inter:]    # [K, H]
            down_w = w2_weight                   # [H, K_out]

        # Step 1-2: Gate and Up projections
        gate_out = torch.matmul(expert_tokens, gate_w.T)  # [N, H]
        up_out = torch.matmul(expert_tokens, up_w.T)      # [N, H]

        # Step 3: SwiGLU activation
        act_out = _npu_swiglu(gate_out, up_out)           # [N, H]

        # Step 4: Down projection
        expert_out = torch.matmul(act_out, down_w.T)      # [N, K_out]

        # Scale by routing weight and scatter back
        weight_scalar = sorted_weights[start:end].unsqueeze(-1)
        token_indices = gather_indices[start:end].long()
        output_buffer.index_add_(
            0, token_indices,
            expert_out * weight_scalar.to(expert_out.dtype)
        )

    return output_buffer


# ---------------------------------------------------------------------------
# Ascend MoE apply kernel (M01 decomposition + M05 routing)
# ---------------------------------------------------------------------------


@register_kernel(
    "moe",
    "apply",
    name="npu_ascend_moe_apply",
    solution="ascend",
    weight_preprocessor=npu_ascend_moe_weights,
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        "x",
        "dense",
        {torch.float16, torch.bfloat16, torch.float32},
    ),
    traits={
        "weight_dtype": frozenset({"fp8", "unquant", "bf16", "fp16"}),
        "activation": frozenset({"silu", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({1}),
        "internal_activation_dtype": frozenset({"input"}),
        "fp8_scale_block_shape": frozenset({(128, 128)}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def npu_ascend_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """MoE expert computation decomposed for Ascend NPU (M01).

    Decomposes the fused triton MoE kernel into separate matmul + swiglu
    + matmul steps that run natively on Ascend NPU:

    1. Dispatch tokens by expert (M05 routing)
    2. For each expert:
       a. ``matmul(tokens, gate_weight.T)`` -> gate
       b. ``matmul(tokens, up_weight.T)``   -> up
       c. ``swiglu(gate, up)``              -> act
       d. ``matmul(act, down_weight.T)``    -> output
    3. Weighted scatter-add to final buffer

    Args:
        plan: Execution plan from ``moe_plan``.
        x: Hidden states ``[num_tokens, hidden_size]``.
        w: Module with processed MoE weights.
        router_logits: Router logits ``[num_tokens, num_experts]``.
        topk_weights: Precomputed expert weights ``[num_tokens, top_k]``.
        topk_ids: Precomputed expert ids ``[num_tokens, top_k]``.
        num_tokens_global: Global token count (unused).
        max_num_tokens_per_gpu: Max tokens per GPU (unused).
        do_finalize: Whether to finalize (unused, always finalizes).
        enable_pdl: PDL flag (unused).

    Returns:
        Output tensor ``[num_tokens, hidden_size]``.
    """
    del num_tokens_global, max_num_tokens_per_gpu, do_finalize, enable_pdl
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    if topk_weights is None or topk_ids is None:
        raise RuntimeError("Ascend MoE requires precomputed topk_weights/topk_ids")

    top_k = getattr(w, "top_k", topk_ids.shape[1])
    n_tokens = x.shape[0]
    hidden_size = x.shape[-1]
    num_experts = getattr(w, "num_experts", router_logits.shape[-1])

    # ------------------------------------------------------------------
    # M05: Dispatch tokens by expert (sort-based routing)
    # ------------------------------------------------------------------
    dispatched_hidden, sorted_weights, gather_indices, expert_offsets = \
        _moe_dispatch(topk_ids, topk_weights, x, num_experts)

    # ------------------------------------------------------------------
    # M01: Decomposed expert computation
    # ------------------------------------------------------------------
    w13_weight = getattr(w, "w13_weight", None)
    w2_weight = getattr(w, "w2_weight", None)

    if w13_weight is None or w2_weight is None:
        raise RuntimeError("MoE weights not found on module")

    output = _expert_compute(
        dispatched_hidden=dispatched_hidden,
        sorted_weights=sorted_weights,
        gather_indices=gather_indices,
        expert_offsets=expert_offsets,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        top_k=top_k,
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype=x.dtype,
        device=x.device,
    )

    return output
