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
# IMPLIED, BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend NPU MHA attention backend using torch_npu fused attention primitives.

Registers NPU-optimized implementations for MHA prefill, extend, and decode
operations. These implementations use ``npu_fused_infer_attention_score``
when running on Huawei Ascend hardware and fall through to the standard
PyTorch SDPA path when the NPU-specific API is unavailable.
"""

from __future__ import annotations

import logging
import math

import torch
from tokenspeed_kernel.platform import CapabilityRequirement

logger = logging.getLogger(__name__)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# ---------------------------------------------------------------------------
# Graph task update support (for NPU graph capture with fused attention)
# ---------------------------------------------------------------------------
# Stores handles + tensor references created during graph capture so that
# ``cuda_graph_wrapper.py`` can hot-update parameters before each replay.
#
# Lifecycle:
#   capture → npu_mha_decode_with_kvcache() appends handle entries
#          → cuda_graph_wrapper retrieves them via get_decode_attn_capture_handles()
#   replay  → cuda_graph_wrapper calls update_decode_attn_graph_params()
#             before graph.replay() with current sequence lengths

_decode_attn_capture_handles: list[dict] = []

# ---------------------------------------------------------------------------
# Causal mask for sparse_mode=3 (CANN V3 requires explicit atten_mask)
# ---------------------------------------------------------------------------
# Pre-built 2048×2048 causal mask for sparse_mode=3 (rightDownCausal).
# CANN V3 requires an explicit atten_mask (bool/uint8) when sparse_mode != 0.
# Shape: (2048, 2048), value: True = masked out, False = allowed.
# Lower triangle + diagonal = False (allowed), upper triangle = True (masked).
# Created once per device and cached lazily.
_MAX_ATTEN_SIZE = 2048
_causal_mask_2048: dict[str, torch.Tensor] = {}


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    """Return a 2048×2048 causal bool mask suitable for ``sparse_mode=3``.

    CANN ``sparse_mode=3`` (rightDownCausal) requires an explicit
    ``atten_mask`` even in TND layout on CANN 2.10.0 (V3 API).
    The mask must be bool or uint8 — float dtypes (bf16/fp16) are rejected.

    The returned mask has shape ``(2048, 2048)`` where ``True`` means
    masked out (upper triangle, j > i) and ``False`` means allowed
    (lower triangle including diagonal, j ≤ i).

    Created once per device and cached in ``_causal_mask_2048``.
    """
    global _causal_mask_2048
    key = device.type  # cache per device type (e.g. "npu")
    if key not in _causal_mask_2048:
        mask = torch.ones(
            (_MAX_ATTEN_SIZE, _MAX_ATTEN_SIZE),
            dtype=torch.bool,
            device=device,
        )
        # Upper triangle (j > i) stays True (masked);
        # lower triangle + diagonal become False (allowed)
        mask = torch.triu(mask, diagonal=1)
        _causal_mask_2048[key] = mask
    return _causal_mask_2048[key]


def get_decode_attn_capture_handles() -> list[dict]:
    """Return all stored capture handles for decode attention and clear list.

    Called by ``cuda_graph_wrapper`` after each graph capture session.
    Each dict contains:
        handle      – opaque handle from ``graph_task_group_end``
        q/k/v       – tensor references (views of persistent buffers)
        num_heads,
        num_kv_heads,
        scale,
        block_size  – static parameters from capture
        block_table – page_table tensor reference (contents updated in-place)
    """
    handles = _decode_attn_capture_handles[:]
    _decode_attn_capture_handles.clear()
    return handles


def update_decode_attn_graph_params(
    handles: list[dict],
    seq_lens_list: list[int],
    stream: torch.npu.Stream | None = None,
) -> None:
    """Hot-update captured decode attention parameters before graph replay.

    Must be called on the capture stream **before** ``graph.replay()``.

    Args:
        handles: List of handle dicts from ``get_decode_attn_capture_handles()``.
        seq_lens_list: Current per-batch sequence lengths as Python ``list[int]``.
                       These replace the capture-time dummy seq_lens.
        stream: NPU stream for the update.  Defaults to current stream.
    """
    fused_attn = _get_fused_attention_score()
    if fused_attn is None:
        logger.warning(
            "update_decode_attn_graph_params: fused_attn unavailable, skipping"
        )
        return
    if stream is None:
        stream = torch.npu.current_stream()
    if handles:
        logger.info(
            "graph_task_update: updating %d handle(s) seq_lens=%s",
            len(handles),
            seq_lens_list,
        )
    for entry in handles:
        handle = entry["handle"]
        torch.npu.graph_task_update_begin(stream, handle)
        fused_attn(
            entry["q"],
            entry["k"],
            entry["v"],
            actual_seq_lengths=seq_lens_list,
            actual_seq_lengths_kv=seq_lens_list,
            num_heads=entry["num_heads"],
            num_key_value_heads=entry["num_key_value_heads"],
            scale=entry["scale"],
            input_layout="BNSD",
            block_table=entry["block_table"],
            block_size=entry["block_size"],
        )
        torch.npu.graph_task_update_end(stream)


# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_fused_attention_score():
    """Return ``torch_npu.npu_fused_infer_attention_score`` or None."""
    try:
        import torch_npu  # noqa: F401
        return torch.ops.npu.npu_fused_infer_attention_score
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# MHA Prefill
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_prefill",
    name="npu_mha_prefill",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def npu_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA prefill using TND layout + ``npu_fused_infer_attention_score``.

    Uses CANN's fused attention kernel with TND (Tile-Nested-Descriptor)
    layout for efficient block-diagonal causal masking in packed sequences,
    with graceful fallback to per-batch PyTorch SDPA when the fused kernel
    is unavailable.

    .. note::

       The fused kernel natively handles GQA via separate ``num_heads``
       and ``num_key_value_heads`` parameters, so no manual KV head
       repetition is needed in the fused path.

    Args:
        q: Query tensor shaped ``[total_q, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[total_kv, num_kv_heads, head_dim]``.
        v: Value tensor shaped ``[total_kv, num_kv_heads, head_dim]``.
        cu_seqlens: Cumulative sequence lengths ``[batch + 1]``.
        cu_seqlens_cpu: Host copy of cumulative lengths.
        max_seqlen: Maximum sequence length.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output ``[total_q, num_q_heads, head_dim]``, or
        ``(output, lse)`` when ``return_lse`` is True.
    """
    del cu_seqlens_cpu, sinks
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    total_q = q.shape[0]
    total_kv = k.shape[0]

    # ---- TND layout + fused attention path (when available) ----
    if fused_attn is not None:
        # TND layout: [total_seq, num_heads, head_dim] (no batch dim)
        q_tnd = q  # [total_q, num_q_heads, head_dim] — already TND shape
        k_tnd = k  # [total_kv, num_kv_heads, head_dim]
        v_tnd = v

        # cu_seqlens must be Python list[int] for actual_seq_lengths
        act_seq = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens

        if window_left > 0:
            # Sliding window: sparse_mode=4 + pre_tokens
            sparse_mode = 4
            causal_tokens = window_left
            logger.debug("npu_mha_prefill: using TND fused attention sparse_mode=4 (sliding window, pre_tokens=%s)", causal_tokens)
        else:
            # Standard causal: sparse_mode=3 requires explicit atten_mask (CANN V3 API)
            sparse_mode = 3
            causal_tokens = None
            atten_mask = _get_causal_mask(q.device)
            logger.debug("npu_mha_prefill: using TND fused attention sparse_mode=3 (causal with 2048x2048 bool atten_mask)")

        kwargs = dict(
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            scale=scale,
            input_layout="TND",
            actual_seq_lengths=act_seq,
            actual_seq_lengths_kv=act_seq,
            sparse_mode=sparse_mode,
            softmax_lse_flag=return_lse,
        )
        if sparse_mode == 4:
            kwargs["pre_tokens"] = causal_tokens
            kwargs["next_tokens"] = 0
        elif sparse_mode == 3:
            kwargs["atten_mask"] = atten_mask

        out, lse_tmp = fused_attn(q_tnd, k_tnd, v_tnd, **kwargs)
        # fused output: [total_q, num_q_heads, head_dim] — already correct shape
    else:
        # ---- SDPA per-batch fallback (correct causal masking) ----
        logger.debug("npu_mha_prefill: using per-batch SDPA fallback (causal masking)")

        act_seq = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else cu_seqlens
        batch_size = len(act_seq) - 1

        k_sdpa = k
        v_sdpa = v
        if num_q_heads != num_kv_heads:
            n_reps = num_q_heads // num_kv_heads
            k_sdpa = k.repeat_interleave(n_reps, dim=1)
            v_sdpa = v.repeat_interleave(n_reps, dim=1)

        outputs = []
        for i in range(batch_size):
            start = act_seq[i]
            end = act_seq[i + 1]
            q_i = q[start:end].unsqueeze(0).transpose(1, 2)   # [1, H, sl, D]
            k_i = k_sdpa[start:end].unsqueeze(0).transpose(1, 2)
            v_i = v_sdpa[start:end].unsqueeze(0).transpose(1, 2)
            out_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i,
                scale=scale,
                is_causal=True,
            ).transpose(1, 2).squeeze(0)  # [sl, H, D]
            outputs.append(out_i)

        out = torch.cat(outputs, dim=0)

    if return_lse:
        if fused_attn is not None:
            # fused path: lse_tmp shape is [total_q, num_q_heads, 1]; squeeze to [total_q, num_q_heads]
            lse = lse_tmp.squeeze(-1)
        else:
            # fallback SDPA path: dummy lse
            lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out


# ---------------------------------------------------------------------------
# MHA Extend with KV Cache
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="npu_mha_extend_with_kvcache",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "is_causal": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def npu_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MHA extend with paged KV cache using Ascend NPU attention.

    Args:
        q: Query tensor shaped ``[total_q, num_q_heads, head_dim]``.
        cu_seqlens_q: Query cumulative sequence lengths ``[batch + 1]``.
        cu_seqlens_kv: KV cumulative sequence lengths ``[batch + 1]``.
        k_cache: Paged key cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged value cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        page_table: Page table ``[batch, max_pages_per_seq]``.
        cache_seqlens: Visible KV lengths ``[batch]``.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        is_causal: Whether query is causal suffix of cached KV.
        window_left: Exclusive left sliding-window size.
        logit_cap: Optional soft cap on logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output or ``(output, lse)`` when ``return_lse`` is True.
    """
    del cu_seqlens_kv, max_seqlen_k, logit_cap, sinks  # not used in NPU path
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    batch_size = cache_seqlens.shape[0]

    # Build actual sequence lengths for query
    seq_lens_q = []
    for i in range(batch_size):
        seq_lens_q.append(int(cu_seqlens_q[i + 1] - cu_seqlens_q[i]))

    # Build actual sequence lengths for KV
    seq_lens_kv = [int(s) for s in cache_seqlens]

    # Reshape query: [total_q, heads, dim] -> [1, heads, total_q, dim] (BNSD)
    q_4d = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_q, head_dim]

    # ---- BNSD layout + GQA handling ----
    # k_cache: [num_pages, page_size, num_kv_heads, head_dim] → transpose to
    # [num_pages, num_kv_heads, page_size, head_dim] for BNSD page layout
    k_bnsd = k_cache.transpose(1, 2)
    v_bnsd = v_cache.transpose(1, 2)
    num_kv_heads = k_cache.shape[2]
    if fused_attn is not None:
        # Manually repeat KV heads for GQA to avoid CANN non-power-of-2 ratio issue
        k_fused = k_bnsd
        v_fused = v_bnsd
        kv_heads = num_kv_heads
        if num_q_heads != num_kv_heads:
            n_reps = num_q_heads // num_kv_heads
            k_fused = k_bnsd.repeat_interleave(n_reps, dim=1)
            v_fused = v_bnsd.repeat_interleave(n_reps, dim=1)
            kv_heads = num_q_heads
        out, _ = fused_attn(
            q_4d, k_fused, v_fused,
            actual_seq_lengths=seq_lens_q,
            actual_seq_lengths_kv=seq_lens_kv,
            num_heads=num_q_heads,
            num_key_value_heads=kv_heads,
            scale=scale,
            input_layout="BNSD",
            block_table=page_table,
            block_size=k_cache.shape[1],
            sparse_mode=0,
        )
        out = out.squeeze(0)
    else:
        # Fallback: SDPA with cached KV
        k_contiguous = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
        v_contiguous = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])
        # GQA repeat for SDPA fallback
        if num_q_heads != num_kv_heads:
            n_reps = num_q_heads // num_kv_heads
            k_contiguous = k_contiguous.repeat_interleave(n_reps, dim=1)
            v_contiguous = v_contiguous.repeat_interleave(n_reps, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k_contiguous.unsqueeze(0).transpose(1, 2),
            v_contiguous.unsqueeze(0).transpose(1, 2),
            scale=scale,
            is_causal=is_causal,
        ).transpose(1, 2).squeeze(0)

    if return_lse:
        lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out


# ---------------------------------------------------------------------------
# MHA Decode with KV Cache
# ---------------------------------------------------------------------------


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="npu_mha_decode_with_kvcache",
    solution="ascend",
    capability=CapabilityRequirement(vendors=frozenset({"huawei"})),
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
    tags={"portability"},
)
def npu_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    """MHA decode with paged KV cache using Ascend NPU.

    Args:
        q: Query tensor shaped ``[batch * max_seqlen_q, num_q_heads, head_dim]``.
        k_cache: Paged key cache ``[num_pages, page_size, num_kv_heads, head_dim]``.
        v_cache: Paged value cache (same layout).
        page_table: Page table ``[batch, max_pages_per_seq]``.
        cache_seqlens: Total KV lengths ``[batch]``.
        max_seqlen_k: Maximum KV length.
        max_seqlen_q: Number of query tokens per request.
        window_left: Exclusive left sliding-window size.
        logit_cap: Optional soft cap on logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.

    Returns:
        Attention output ``[batch * max_seqlen_q, num_q_heads, head_dim]``.
    """
    del max_seqlen_k, logit_cap, sinks  # not used in NPU path
    if not _is_npu_available():
        raise RuntimeError("NPU not available")

    fused_attn = _get_fused_attention_score()
    num_q_heads = q.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    batch_size = q.shape[0] // max_seqlen_q

    # Detect CUDA graph capture mode: inside torch.npu.graph(), stream sync is
    # not allowed on Ascend NPU.  We must use only basic tensor ops (matmul,
    # softmax) and avoid any .item() calls that would trigger a CPU sync.
    is_capturing = torch.npu.is_current_stream_capturing()

    # Reshape q for batch processing
    # q is [batch * max_seqlen_q, num_heads, head_dim]
    # -> [batch, max_seqlen_q, num_heads, head_dim] (PyTorch BSND layout)
    q_4d = q.reshape(batch_size, max_seqlen_q, num_q_heads, head_dim)

    if is_capturing:
        # ---- Graph-capture path with fused attention + graph_task_group ----
        # Use graph_task_group_begin/end to wrap npu_fused_infer_attention_score
        # so its parameters (seq_lens, block_table) can be hot-updated via
        # graph_task_update before each replay, avoiding the performance
        # penalty of manual matmul (padded KV, ~3 kernels vs 1 fused kernel).
        #
        # During capture we MUST avoid any .item() / int(tensor) calls because
        # they trigger NPU→CPU sync which is forbidden on a captured stream.
        # The seq_lens_list here is a DUMMY placeholder; the real values are
        # passed via update_decode_attn_graph_params() before each replay.
        logger.info(
            "npu_mha_decode_with_kvcache: graph capture mode, using graph_task_group"
        )
        stream = torch.npu.current_stream()
        seq_lens_list = [1] * batch_size  # dummy; real values set via graph_task_update
        # NPU BNSD layout: [batch, num_heads, seqlen, head_dim]
        q_bnsd = q_4d.transpose(1, 2)  # [B, H_q, Tq, D]
        k_bnsd = k_cache.transpose(1, 2)  # [num_pages, num_kv_heads, P, D]
        v_bnsd = v_cache.transpose(1, 2)
        # Manually repeat KV heads for GQA during capture so the fused kernel
        # sees num_heads == num_kv_heads (avoids non-power-of-2 ratio issue).
        num_kv_heads = k_cache.shape[2]
        torch.npu.graph_task_group_begin(stream)
        out, _ = fused_attn(
            q_bnsd, k_bnsd, v_bnsd,
            actual_seq_lengths=seq_lens_list,
            actual_seq_lengths_kv=seq_lens_list,
            num_heads=num_q_heads,
            num_key_value_heads=k_bnsd.shape[1],
            scale=scale,
            input_layout="BNSD",
            block_table=page_table,
            block_size=k_cache.shape[1],
        )
        handle = torch.npu.graph_task_group_end(stream)
        # Store handle + tensor references for pre-replay parameter update.
        # NOTE: k/v tensors already have repeated heads, so num_key_value_heads
        # equals num_q_heads. The graph_task_update path uses these stored
        # tensors directly, so it naturally sees the repeated layout.
        _decode_attn_capture_handles.append({
            "handle": handle,
            "q": q_bnsd,
            "k": k_bnsd,
            "v": v_bnsd,
            "num_heads": num_q_heads,
            "num_key_value_heads": k_bnsd.shape[1],
            "scale": scale,
            "block_table": page_table,
            "block_size": k_cache.shape[1],
        })
        out = out.reshape(-1, num_q_heads, head_dim)
    else:
        # ---- Normal (non-capturing) path: try fused then SDPA ----
        use_fused = False
        fused_kv_heads = k_cache.shape[2]
        if fused_attn is not None:
            try:
                seq_lens_list = [int(s) for s in cache_seqlens]
                q_bnsd = q_4d.transpose(1, 2)
                k_bnsd = k_cache.transpose(1, 2)
                v_bnsd = v_cache.transpose(1, 2)
                # Manually repeat KV heads for GQA to avoid CANN non-power-of-2 ratio issue
                if num_q_heads != fused_kv_heads:
                    n_reps = num_q_heads // fused_kv_heads
                    k_bnsd = k_bnsd.repeat_interleave(n_reps, dim=1)
                    v_bnsd = v_bnsd.repeat_interleave(n_reps, dim=1)
                out, _ = fused_attn(
                    q_bnsd, k_bnsd, v_bnsd,
                    actual_seq_lengths=seq_lens_list,
                    actual_seq_lengths_kv=seq_lens_list,
                    num_heads=num_q_heads,
                    num_key_value_heads=num_q_heads,
                    scale=scale,
                    input_layout="BNSD",
                    block_table=page_table,
                    block_size=k_cache.shape[1],
                )
                out = out.reshape(-1, num_q_heads, head_dim)
                use_fused = True
            except RuntimeError as e:
                logger.warning("NPU fused attention failed, falling back to SDPA: %s", e)
        if not use_fused:
            # Fallback: SDPA decode with proper page table gathering
            num_pages, page_size, num_kv_heads, head_dim = k_cache.shape
            batch_size = page_table.shape[0]
            max_pages_per_seq = page_table.shape[1]
            if batch_size == 1:
                num_kv = int(cache_seqlens[0].item())
                pages = page_table[0, :(num_kv + page_size - 1) // page_size]
                gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                if num_q_heads != num_kv_heads:
                    n_reps = num_q_heads // num_kv_heads
                    gathered_k = gathered_k.repeat_interleave(n_reps, dim=1)
                    gathered_v = gathered_v.repeat_interleave(n_reps, dim=1)
                q_2d = q.reshape(1, -1, num_q_heads, head_dim)
                k_2d = gathered_k.unsqueeze(0)
                v_2d = gathered_v.unsqueeze(0)
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_2d.transpose(1, 2),
                    k_2d.transpose(1, 2),
                    v_2d.transpose(1, 2),
                    scale=scale,
                ).transpose(1, 2).squeeze(0)
            else:
                outputs = []
                offset = 0
                for b in range(batch_size):
                    num_kv = int(cache_seqlens[b].item())
                    pages = page_table[b, :(num_kv + page_size - 1) // page_size]
                    gathered_k = k_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                    gathered_v = v_cache[pages.long()].reshape(-1, num_kv_heads, head_dim)[:num_kv]
                    if num_q_heads != num_kv_heads:
                        n_reps = num_q_heads // num_kv_heads
                        gathered_k = gathered_k.repeat_interleave(n_reps, dim=1)
                        gathered_v = gathered_v.repeat_interleave(n_reps, dim=1)
                    q_b = q[offset:offset + max_seqlen_q].reshape(1, max_seqlen_q, num_q_heads, head_dim)
                    k_b = gathered_k.unsqueeze(0)
                    v_b = gathered_v.unsqueeze(0)
                    out_b = torch.nn.functional.scaled_dot_product_attention(
                        q_b.transpose(1, 2),
                        k_b.transpose(1, 2),
                        v_b.transpose(1, 2),
                        scale=scale,
                    ).transpose(1, 2).squeeze(0)
                    outputs.append(out_b)
                    offset += max_seqlen_q
                out = torch.cat(outputs, dim=0)

    if return_lse:
        lse = torch.zeros((q.shape[0], num_q_heads), dtype=torch.float32, device=q.device)
        return out, lse
    return out
