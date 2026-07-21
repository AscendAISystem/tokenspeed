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

import logging

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

logger = logging.getLogger(__name__)

__all__ = [
    "npu_ascend_moe_apply",
    "npu_ascend_moe_apply_native",
    "npu_ascend_moe_weights",
]


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _npu_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float | None = None,
    beta: float = 0.0,
    limit: float | None = None,
) -> torch.Tensor:
    """Compute SwiGLU activation on NPU.

    Supports both standard SwiGLU (silu(gate) * up) and GPT-OSS style
    non-standard activation: (up + beta) * (gate * sigmoid(alpha * gate))
    with optional clamping.

    Standard path: uses NPU-native ``torch.ops.npu.npu_swiglu`` when
    available for better performance, otherwise falls back to
    ``F.silu(gate) * up``.

    GPT-OSS path (when beta != 0.0): replicates the HF reference:
    ``gate = gate.clamp(max=limit)``,
    ``up = up.clamp(min=-limit, max=limit)``,
    ``gate_act = gate * sigmoid(alpha * gate)``,
    ``output = (up + beta) * gate_act``.

    Args:
        gate: Gate tensor.
        up: Up tensor.
        alpha: GPT-OSS activation alpha (e.g. 1.702 for GPT-OSS-20B).
        beta: GPT-OSS up factor (e.g. 1.0 for GPT-OSS-20B). When 0.0,
              standard SwiGLU is used (default).
        limit: Clamp limit (e.g. 7.0 for GPT-OSS-20B).

    Returns:
        Activation output tensor.
    """
    if beta != 0.0:
        # GPT-OSS style: (up + beta) * (gate * sigmoid(alpha * gate))
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        gate_act = gate * torch.sigmoid(gate * alpha) if alpha is not None else gate * torch.sigmoid(gate)
        return (up + beta) * gate_act

    # Standard SwiGLU
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
    w13_weight_bias: torch.Tensor | None = None,
    w2_weight_bias: torch.Tensor | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float = 0.0,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Compute per-expert matmuls with SwiGLU activation using npu_grouped_matmul.

    M01 decomposed FP8 path, optimized with ``npu_grouped_matmul``:
      grouped_matmul([expert_tokens_i], [gate_w_i.T])   -> gate_i
      grouped_matmul([expert_tokens_i], [up_w_i.T])     -> up_i
      swiglu(gate_i, up_i)                               -> act_i
      grouped_matmul([act_i], [down_w_i.T])              -> out_i

    Scatter-add with routing weights stays per-expert (``index_add_`` cannot
    be grouped). Optional per-expert bias is added to gate, up, and down outputs.

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
        w13_weight_bias: Optional ``[num_experts, K]`` bias for fused gate+up.
        w2_weight_bias: Optional ``[num_experts, H]`` bias for down projection.
        swiglu_alpha: GPT-OSS activation alpha (e.g. 1.702). Passed to
            ``_npu_swiglu`` when ``swiglu_beta != 0``.
        swiglu_beta: GPT-OSS up factor (e.g. 1.0). When non-zero, uses
            GPT-OSS style activation instead of standard SwiGLU.
        swiglu_limit: Clamp limit for GPT-OSS activation (e.g. 7.0).

    Returns:
        Output tensor ``[n_tokens, hidden_size]``.
    """
    is_expert_parallel = w13_weight.dim() == 3  # [E, 2*intermediate, hidden] for w13
    total_inter = w13_weight.shape[1] if is_expert_parallel else w13_weight.shape[0]
    half_inter = total_inter // 2

    has_bias = w13_weight_bias is not None
    if has_bias:
        has_bias_ep = w13_weight_bias.dim() == 2  # [E, K]

    # Collect per-expert tokens and weights (pre-transposed for grouped_matmul)
    expert_token_list = []
    gate_w_list = []   # each: [hidden_size, intermediate] pre-transposed
    up_w_list = []     # each: [hidden_size, intermediate] pre-transposed
    down_w_list = []   # each: [intermediate, hidden] pre-transposed
    gate_bias_list = []  # each: [intermediate] if has_bias
    up_bias_list = []    # each: [intermediate] if has_bias
    down_bias_list = []  # each: [hidden] if has_bias
    expert_ranges = []
    valid_experts = []

    for expert_id in range(num_experts):
        start = expert_offsets[expert_id].item()
        end = expert_offsets[expert_id + 1].item()
        if start >= end:
            continue

        expert_tokens = dispatched_hidden[start:end]
        expert_token_list.append(expert_tokens)

        if is_expert_parallel:
            # w13_weight[expert_id]: [2*intermediate, hidden]
            gate_w = w13_weight[expert_id, :half_inter, :].T   # [hidden, intermediate]
            up_w = w13_weight[expert_id, half_inter:, :].T     # [hidden, intermediate]
            down_w = (w2_weight[expert_id] if w2_weight.dim() == 3 else w2_weight).T  # [intermediate, hidden]
            if has_bias and has_bias_ep:
                gate_bias_list.append(w13_weight_bias[expert_id, :half_inter])
                up_bias_list.append(w13_weight_bias[expert_id, half_inter:])
                down_bias_list.append(
                    w2_weight_bias[expert_id] if w2_weight_bias is not None and w2_weight_bias.dim() == 2
                    else None
                )
            elif has_bias:
                gate_bias_list.append(w13_weight_bias[:half_inter])
                up_bias_list.append(w13_weight_bias[half_inter:])
                down_bias_list.append(w2_weight_bias if w2_weight_bias is not None else None)
        else:
            gate_w = w13_weight[:half_inter, :].T   # [hidden, intermediate]
            up_w = w13_weight[half_inter:, :].T     # [hidden, intermediate]
            down_w = w2_weight.T                    # [intermediate, hidden]
            if has_bias:
                gate_bias_list.append(w13_weight_bias[:half_inter] if w13_weight_bias is not None else None)
                up_bias_list.append(w13_weight_bias[half_inter:] if w13_weight_bias is not None else None)
                down_bias_list.append(w2_weight_bias if w2_weight_bias is not None else None)

        gate_w_list.append(gate_w.contiguous())
        up_w_list.append(up_w.contiguous())
        down_w_list.append(down_w.contiguous())
        expert_ranges.append((start, end))
        valid_experts.append(expert_id)

    if not expert_token_list:
        return torch.zeros(n_tokens, hidden_size, dtype=dtype, device=device)

    # Grouped matmul: all experts' gate and up projections in parallel
    gate_out_list = torch.ops.npu.npu_grouped_matmul.List(
        expert_token_list, gate_w_list, split_item=0, group_type=0
    )
    up_out_list = torch.ops.npu.npu_grouped_matmul.List(
        expert_token_list, up_w_list, split_item=0, group_type=0
    )

    # Add bias to gate and up outputs if available
    if has_bias:
        gate_out_list = [
            g + b.unsqueeze(0) if b is not None else g
            for g, b in zip(gate_out_list, gate_bias_list)
        ]
        up_out_list = [
            u + b.unsqueeze(0) if b is not None else u
            for u, b in zip(up_out_list, up_bias_list)
        ]

    # SwiGLU activation (elementwise, per-expert)
    # Pass through GPT-OSS activation params if applicable (swiglu_beta != 0)
    act_out_list = [
        _npu_swiglu(g, u, alpha=swiglu_alpha, beta=swiglu_beta, limit=swiglu_limit)
        for g, u in zip(gate_out_list, up_out_list)
    ]

    # Grouped matmul: all experts' down projections
    expert_out_list = torch.ops.npu.npu_grouped_matmul.List(
        act_out_list, down_w_list, split_item=0, group_type=0
    )

    # Add down bias if available
    if has_bias and any(b is not None for b in down_bias_list):
        expert_out_list = [
            o + b.unsqueeze(0) if b is not None else o
            for o, b in zip(expert_out_list, down_bias_list)
        ]

    # Weighted scatter-add to output buffer
    output_buffer = torch.zeros(n_tokens, hidden_size, dtype=dtype, device=device)
    for idx, expert_id in enumerate(valid_experts):
        start, end = expert_ranges[idx]
        weight_scalar = sorted_weights[start:end].unsqueeze(-1)
        token_indices = gather_indices[start:end].long()
        output_buffer.index_add_(
            0, token_indices,
            expert_out_list[idx] * weight_scalar.to(expert_out_list[idx].dtype)
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
        "supports_bias": frozenset({True, False}),
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
    num_local_experts = int(getattr(w, "num_local_experts", num_experts))
    # Local EP: offset+mask to convert global expert IDs to local
    ep_size = int(getattr(w, "ep_size", 1))
    logger.info("[MoE] ascend 路径 (torch API fallback, ep_size=%d)", ep_size)
    if ep_size > 1:
        expert_offset = int(getattr(w, "ep_rank", 0)) * num_local_experts
        local_ids = topk_ids - expert_offset
        local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
        topk_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
        topk_ids = torch.where(local_mask, local_ids, topk_ids.new_full((), -1))
        num_experts = num_local_experts

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

    # Read optional per-expert bias tensors
    w13_weight_bias = getattr(w, "w13_weight_bias", None)
    w2_weight_bias = getattr(w, "w2_weight_bias", None)

    # Read GPT-OSS activation params from module (set in MoELayer.__init__)
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None:
        swiglu_alpha = swiglu_arg.alpha
        swiglu_limit = swiglu_arg.limit
    else:
        swiglu_alpha = None
        swiglu_limit = None
    swiglu_beta = getattr(w, "swiglu_beta", 0.0) or 0.0

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
        w13_weight_bias=w13_weight_bias,
        w2_weight_bias=w2_weight_bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )

    return output


# ---------------------------------------------------------------------------
# Ascend MoE apply native kernel (M01 decomposition + NPU native dispatch)
# ---------------------------------------------------------------------------


@register_kernel(
    "moe",
    "apply",
    name="npu_ascend_moe_apply_native",
    solution="ascend_native",
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
        "supports_bias": frozenset({True, False}),
    },
    priority=Priority.PERFORMANT,
)
def npu_ascend_moe_apply_native(
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
    """MoE expert computation using NPU native ops (ascend_native).

    Uses ``torch_npu.npu_moe_init_routing_v2`` for NPU-native dispatch,
    ``npu_grouped_matmul`` with ``group_list`` for compute (no per-expert
    token list building), and ``index_add_`` for weighted scatter-add combine.

    The ascend_native solution replaces the sort-based routing (M05) with
    a dedicated NPU routing primitive while keeping the same M01 decomposed
    matmul + swiglu + matmul expert compute structure.

    Priority is set to PERFORMANT (vs PORTABLE for the ascend fallback),
    so this kernel is preferred on NPU when both are registered.

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

    import torch_npu  # noqa: F401

    top_k = getattr(w, "top_k", topk_ids.shape[1])
    n_tokens = x.shape[0]
    hidden_size = x.shape[-1]
    num_experts = getattr(w, "num_experts", router_logits.shape[-1])
    num_local_experts = int(getattr(w, "num_local_experts", num_experts))
    # Local EP: offset+mask to convert global expert IDs to local
    ep_size = int(getattr(w, "ep_size", 1))
    logger.info("[MoE] ascend_native 路径 (ep_size=%d)", ep_size)
    if ep_size > 1:
        expert_offset = int(getattr(w, "ep_rank", 0)) * num_local_experts
        local_ids = topk_ids - expert_offset
        local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
        topk_weights = torch.where(local_mask, topk_weights, torch.zeros_like(topk_weights))
        topk_ids = torch.where(local_mask, local_ids, topk_ids.new_full((), -1))
        num_experts = num_local_experts

    w13_weight = getattr(w, "w13_weight", None)
    w2_weight = getattr(w, "w2_weight", None)
    if w13_weight is None or w2_weight is None:
        raise RuntimeError("MoE weights not found on module")

    # Read optional per-expert bias tensors
    w13_weight_bias = getattr(w, "w13_weight_bias", None)
    w2_weight_bias = getattr(w, "w2_weight_bias", None)

    # Read GPT-OSS activation params
    swiglu_arg = getattr(w, "swiglu_arg", None)
    if swiglu_arg is not None:
        swiglu_alpha = swiglu_arg.alpha
        swiglu_limit = swiglu_arg.limit
    else:
        swiglu_alpha = None
        swiglu_limit = None
    swiglu_beta = getattr(w, "swiglu_beta", 0.0) or 0.0

    is_expert_parallel = w13_weight.dim() == 3
    total_inter = w13_weight.shape[1] if is_expert_parallel else w13_weight.shape[0]
    half_inter = total_inter // 2

    # ------------------------------------------------------------------
    # Step 1: Dispatch using NPU-native npu_moe_init_routing_v2
    #
    # NOTE: npu_moe_init_routing_v2 does NOT handle -1 expert IDs.
    # EP masking sets non-local experts to -1, which causes the op
    # to return zero expert counts. So we replace -1 with 0 first
    # (same as _moe_dispatch in the ascend fallback path). The zero
    # weights for masked entries ensure no contribution to the output.
    # ------------------------------------------------------------------
    flat_ids_safe = torch.where(
        topk_ids.reshape(-1) >= 0,
        topk_ids.reshape(-1),
        topk_ids.new_zeros(()).reshape(-1),
    )
    topk_ids_safe = flat_ids_safe.reshape(topk_ids.shape).contiguous().int()
    expanded_x, expanded_row_idx, expert_count, _ = torch_npu.npu_moe_init_routing_v2(
        x, topk_ids_safe,
        active_num=-1, expert_capacity=-1, expert_num=num_experts,
        drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        quant_mode=-1, active_expert_range=[0, num_experts], row_idx_type=0,
    )
    valid_count = expert_count.sum().item()
    if valid_count == 0:
        return torch.zeros(n_tokens, hidden_size, dtype=x.dtype, device=x.device)

    expanded_x = expanded_x[:valid_count]
    expanded_row_idx = expanded_row_idx[:valid_count]

    # ------------------------------------------------------------------
    # Step 2: Build per-expert weight lists and group_list
    #         (no per-expert token list — use group_list instead)
    # ------------------------------------------------------------------
    valid_mask = expert_count > 0
    valid_ids = valid_mask.nonzero().squeeze(-1)
    valid_counts = expert_count[valid_ids]
    # group_list: cumsum of per-expert token counts (List[int] format)
    group_list = valid_counts.cumsum(dim=0).int().tolist()

    gate_w_list = []
    up_w_list = []
    down_w_list = []
    gate_bias_list = [] if w13_weight_bias is not None else None
    up_bias_list = [] if w13_weight_bias is not None else None
    down_bias_list = [] if w2_weight_bias is not None else None
    has_bias_ep = (w13_weight_bias is not None and w13_weight_bias.dim() == 2)

    for eid in valid_ids.tolist():
        if is_expert_parallel:
            gate_w = w13_weight[eid, :half_inter, :].T.contiguous()
            up_w = w13_weight[eid, half_inter:, :].T.contiguous()
            down_w = (w2_weight[eid] if w2_weight.dim() == 3 else w2_weight).T.contiguous()
            if w13_weight_bias is not None and has_bias_ep:
                gate_bias_list.append(w13_weight_bias[eid, :half_inter])
                up_bias_list.append(w13_weight_bias[eid, half_inter:])
                down_bias_list.append(
                    w2_weight_bias[eid] if w2_weight_bias is not None and w2_weight_bias.dim() == 2
                    else None
                )
            elif w13_weight_bias is not None:
                gate_bias_list.append(w13_weight_bias[:half_inter])
                up_bias_list.append(w13_weight_bias[half_inter:])
                down_bias_list.append(w2_weight_bias if w2_weight_bias is not None else None)
        else:
            gate_w = w13_weight[:half_inter, :].T.contiguous()
            up_w = w13_weight[half_inter:, :].T.contiguous()
            down_w = w2_weight.T.contiguous()
            if w13_weight_bias is not None:
                gate_bias_list.append(w13_weight_bias[:half_inter] if w13_weight_bias is not None else None)
                up_bias_list.append(w13_weight_bias[half_inter:] if w13_weight_bias is not None else None)
                down_bias_list.append(w2_weight_bias if w2_weight_bias is not None else None)

        gate_w_list.append(gate_w)
        up_w_list.append(up_w)
        down_w_list.append(down_w)

    # ------------------------------------------------------------------
    # Step 3: Grouped matmul — gate and up projections
    #         x is a single tensor, group_list splits by expert rows
    # ------------------------------------------------------------------
    x_list = [expanded_x]
    gate_out_list = torch.ops.npu.npu_grouped_matmul.List(
        x_list, gate_w_list,
        group_list=group_list, split_item=0, group_type=0, group_list_type=0,
    )
    up_out_list = torch.ops.npu.npu_grouped_matmul.List(
        x_list, up_w_list,
        group_list=group_list, split_item=0, group_type=0, group_list_type=0,
    )

    # Add bias to gate and up if available
    if w13_weight_bias is not None:
        gate_out_list = [
            g + b.unsqueeze(0) if b is not None else g
            for g, b in zip(gate_out_list, gate_bias_list)
        ]
        up_out_list = [
            u + b.unsqueeze(0) if b is not None else u
            for u, b in zip(up_out_list, up_bias_list)
        ]

    # ------------------------------------------------------------------
    # Step 4: SwiGLU activation (per-expert, elementwise)
    # ------------------------------------------------------------------
    act_out_list = [
        _npu_swiglu(g, u, alpha=swiglu_alpha, beta=swiglu_beta, limit=swiglu_limit)
        for g, u in zip(gate_out_list, up_out_list)
    ]

    # ------------------------------------------------------------------
    # Step 5: Grouped matmul — down projection
    # ------------------------------------------------------------------
    cat_act = torch.cat(act_out_list, dim=0)
    expert_out_list = torch.ops.npu.npu_grouped_matmul.List(
        [cat_act], down_w_list,
        group_list=group_list, split_item=0, group_type=0, group_list_type=0,
    )

    # Add down bias if available
    if w2_weight_bias is not None and any(b is not None for b in down_bias_list):
        expert_out_list = [
            o + b.unsqueeze(0) if b is not None else o
            for o, b in zip(expert_out_list, down_bias_list)
        ]

    # ------------------------------------------------------------------
    # Step 6: Weighted scatter-add to output buffer
    #         Use sort_order (same safe IDs as Step 1, so the ordering
    #         matches npu_moe_init_routing_v2) for weight permutation
    #         and gather indices.
    # ------------------------------------------------------------------
    flat_weights = topk_weights.reshape(-1)
    sort_order = torch.argsort(flat_ids_safe.int(), stable=True)
    permuted_weights = flat_weights[sort_order][:valid_count]
    gather_indices = sort_order[:valid_count] // top_k

    output_buffer = torch.zeros(n_tokens, hidden_size, dtype=x.dtype, device=x.device)
    cum = 0
    for idx, _ in enumerate(valid_ids.tolist()):
        count = valid_counts[idx].item()
        weight_scalar = permuted_weights[cum:cum + count].unsqueeze(-1)
        token_indices = gather_indices[cum:cum + count].long()
        output_buffer.index_add_(
            0, token_indices,
            expert_out_list[idx] * weight_scalar.to(expert_out_list[idx].dtype),
        )
        cum += count

    return output_buffer
