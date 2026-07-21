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

"""Ascend NPU sampling backend.

Pure-torch sampling backend for Huawei Ascend NPU devices.  Uses Gumbel-max
trick for stochastic sampling and ``torch_npu.npu_top_k_top_p_sample`` when
available.  Penalties (repetition / frequency / presence) are implemented in
pure PyTorch (no flashinfer dependency).

CUDA path unchanged — all NPU code guarded by ``_is_npu_available()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax

from tokenspeed.runtime.sampling.backends.base import (
    CUDA_GRAPH_VARIANT_DEFAULT,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# NPU availability helpers
# ---------------------------------------------------------------------------


def _is_npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _gather_scalars(
    index: torch.Tensor,
    *,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    seed: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
    n: int = 1,
) -> tuple:
    """Pure-torch gather-and-expand for per-request sampling scalars.

    Replaces ``tokenspeed_kernel.ops.sampling.triton.gather_and_expand_scalars``
    which requires a CUDA-capable device (Triton).  This version uses
    ``index_select`` + ``repeat_interleave`` and works on any device.

    Returns ``(temperatures, top_ks, top_ps, seeds_or_None, offsets_or_None)``.
    """
    bs = index.size(0)
    total = bs * n

    results = []
    for pool_tensor in (temperature, top_k, top_p, seed, offsets):
        if pool_tensor is None:
            results.append(None)
        else:
            gathered = pool_tensor.index_select(0, index)  # [bs]
            if n > 1:
                gathered = gathered.repeat_interleave(n, dim=0)  # [bs * n]
            results.append(gathered)

    return tuple(results)


def _get_npu_top_k_top_p_sample():
    """Return ``torch_npu.npu_top_k_top_p_sample`` callable or ``None``."""
    try:
        import torch_npu  # noqa: F401

        return torch.ops.npu.npu_top_k_top_p_sample
    except (ImportError, AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Penalty + bias application (pure torch, ported from flashinfer_full.py)
# ---------------------------------------------------------------------------


@nvtx_range("sampling:npu_penalties", color="yellow")
def _apply_penalties_and_bias(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    counts: torch.Tensor,
    rep_pen_pool: torch.Tensor,
    freq_pen_pool: torch.Tensor,
    pres_pen_pool: torch.Tensor,
    logit_bias: torch.Tensor | None = None,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Apply repetition/frequency/presence penalties and logit bias.

    Args:
        logits: ``[bs * N, V]`` float32 on NPU.
        sampling_info: Batch sampling info with ``req_pool_indices``.
        counts: ``[pool_rows, V]`` int32 token counts.
        rep_pen_pool: ``[pool_rows]`` bf16 repetition penalty scalars.
        freq_pen_pool: ``[pool_rows]`` bf16 frequency penalty scalars.
        pres_pen_pool: ``[pool_rows]`` bf16 presence penalty scalars.
        logit_bias: Optional ``[pool_rows, V]`` bf16 logit bias.
        num_tokens_per_req: >1 for spec-decode verify path.

    Returns:
        Logits with penalties applied.
    """
    pool_idx = sampling_info.req_pool_indices

    if num_tokens_per_req > 1:
        pool_idx = torch.repeat_interleave(pool_idx, num_tokens_per_req, dim=0)

    counts_slice = counts.index_select(0, pool_idx)  # [bs*N, V]
    active = counts_slice > 0
    counts_f = counts_slice.to(logits.dtype)
    active_f = active.to(logits.dtype)

    # Gather per-request penalty scalars [bs*N] -> [bs*N, 1]
    rep = rep_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)
    freq = freq_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)
    presence = pres_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)

    # 1. Repetition penalty (multiplicative)
    scales = torch.where(active, rep.expand_as(logits), torch.ones_like(logits))
    logits = torch.where(logits > 0, logits / scales, logits * scales)

    # 2. Frequency + presence (additive)
    logits = logits - freq * counts_f - presence * active_f

    # 3. Per-token logit_bias (additive)
    if logit_bias is not None:
        logits = logits + logit_bias.index_select(0, pool_idx)

    return logits


@nvtx_range("sampling:npu_accum_counts", color="yellow")
def _accumulate_counts(
    counts: torch.Tensor,
    pool_idx: torch.Tensor,
    tokens: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Graph-safe in-place scatter: ``counts[pool_idx, tokens] += weights``."""
    counts.index_put_(
        (pool_idx, tokens.long()),
        weights.to(torch.int32),
        accumulate=True,
    )


# ---------------------------------------------------------------------------
# Verify helpers (copy of greedy's pure-torch chain-greedy verify)
# ---------------------------------------------------------------------------


def _verify_chain_greedy_torch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    batch_size: int,
    num_draft_tokens: int,
) -> None:
    """Pure-torch chain-greedy verify.  See ``greedy.py`` for docs."""
    bs = batch_size
    n = num_draft_tokens

    match = candidates[:, 1:] == target_predict[:, :-1].to(candidates.dtype)
    leading = torch.cumprod(match.to(torch.int32), dim=1)
    num_accepted = leading.sum(dim=1).to(torch.int32)

    predicts.copy_(target_predict.reshape(-1).to(torch.int32))

    device = candidates.device
    pos = torch.arange(n, device=device).unsqueeze(0)
    batch_off = torch.arange(bs, device=device).unsqueeze(1) * n
    flat_idx = (batch_off + pos).to(torch.int32)
    valid = pos <= num_accepted.unsqueeze(1)
    accept_index.copy_(
        torch.where(valid, flat_idx, torch.full_like(accept_index, -1))
    )
    accept_token_num.copy_(num_accepted)


# ---------------------------------------------------------------------------
# AscendSamplingBackend
# ---------------------------------------------------------------------------


class AscendSamplingBackend(SamplingBackend):
    """NPU sampling backend using Gumbel-max trick + ``npu_top_k_top_p_sample``.

    Supports temperature, top-k, top-p, repetition/frequency/presence
    penalties, and per-token logit_bias.  Uses pure PyTorch for all
    operations; no flashinfer dependency.

    Single-step ``sample()`` flow:
      1. Grammar bitmask (same as greedy)
      2. Apply penalties (pure torch)
      3. Temperature scaling
      4. If all greedy: plain argmax
      5. Else: ``npu_top_k_top_p_sample`` (fused) or Gumbel-max fallback
      6. TP-sync broadcast

    Multi-step ``verify()`` flow:
      1. Grammar bitmask
      2. Apply penalties + temperature
      3. Generate target predictions via argmax
      4. Chain-greedy verify (same as greedy.py, pure torch)
    """

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = False

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)

        if config.max_req_pool_size <= 0 or config.vocab_size <= 0:
            raise ValueError(
                "AscendSamplingBackend requires max_req_pool_size > 0 and "
                f"vocab_size > 0; got max_req_pool_size={config.max_req_pool_size}, "
                f"vocab_size={config.vocab_size}"
            )

        pool_rows = config.max_req_pool_size + 1

        # Token count buffer for penalties
        self._counts = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.int32,
            device=config.device,
        )

        # Per-slot logit bias (bf16)
        self._logit_bias = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.bfloat16,
            device=config.device,
        )

        # Per-slot scalars
        self._temperature_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._top_k_pool = torch.ones(
            (pool_rows,), dtype=torch.int32, device=config.device
        )
        self._top_p_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._seed_pool = torch.zeros(
            (pool_rows,), dtype=torch.int64, device=config.device
        )
        self._freq_pen_pool = torch.zeros(
            (pool_rows,), dtype=torch.bfloat16, device=config.device
        )
        self._pres_pen_pool = torch.zeros(
            (pool_rows,), dtype=torch.bfloat16, device=config.device
        )
        self._rep_pen_pool = torch.full(
            (pool_rows,), 1.0, dtype=torch.bfloat16, device=config.device
        )

        # Buffers for sample/verify outputs
        max_pad_bs = config.max_bs
        max_n = config.max_draft_tokens_per_req

        self._ones_buf = torch.ones(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )
        self._sample_token_buf = torch.empty(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_buf = torch.zeros(
            (max_pad_bs * max_n,), dtype=torch.int32, device=config.device
        )
        self._accept_index_buf = torch.zeros(
            (max_pad_bs * max_n,), dtype=torch.int32, device=config.device
        )
        self._accept_length_buf = torch.zeros(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))
        self._freq_pen_pool[pool_idx].fill_(float(sp.frequency_penalty))
        self._pres_pen_pool[pool_idx].fill_(float(sp.presence_penalty))
        self._rep_pen_pool[pool_idx].fill_(float(sp.repetition_penalty))

        # Zero the slot's count row
        self._counts[pool_idx].fill_(0)

        # Zero + scatter logit_bias
        self._logit_bias[pool_idx].fill_(0.0)
        bias_map = getattr(sp, "logit_bias", None) if sp is not None else None
        if bias_map:
            vocab = self._logit_bias.shape[1]
            raw_ids = [int(tid) for tid in bias_map.keys()]
            assert all(0 <= tid < vocab for tid in raw_ids), (
                f"logit_bias contains out-of-vocab token id(s); "
                f"vocab_size={vocab}, offending={[t for t in raw_ids if not 0 <= t < vocab]}"
            )
            token_ids = torch.tensor(
                raw_ids,
                device=self._logit_bias.device,
                dtype=torch.long,
            )
            bias_values = torch.tensor(
                list(bias_map.values()),
                device=self._logit_bias.device,
                dtype=torch.bfloat16,
            )
            self._logit_bias[pool_idx, token_ids] = bias_values

    def reset_capture_state(self) -> None:
        self._counts[0].fill_(0)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    @nvtx_range("sampling:ascend_sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits = logits_output.next_token_logits.float()

        # Grammar bitmask
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        # Apply penalties
        logits = _apply_penalties_and_bias(
            logits,
            sampling_info,
            counts=self._counts,
            rep_pen_pool=self._rep_pen_pool,
            freq_pen_pool=self._freq_pen_pool,
            pres_pen_pool=self._pres_pen_pool,
            logit_bias=self._logit_bias,
            num_tokens_per_req=1,
        )

        if sampling_info.is_all_greedy:
            tokens = sampling_argmax(logits, out=self._sample_token_buf[: logits.shape[0]])
        else:
            temperatures, top_ks, top_ps, seeds, offsets = _gather_scalars(
                sampling_info.req_pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                seed=self._seed_pool,
                offsets=sampling_info.valid_cache_lengths,
            )

            # Temperature scaling
            logits = logits / temperatures[:, None]

            # Use fused NPU top_k_top_p_sample when available
            tokens = self._sample_npu_top_k_top_p(logits, top_ks, top_ps, seeds, offsets)

        # TP-rank sync
        self.maybe_broadcast(tokens)

        # Accumulate counts for penalties
        _accumulate_counts(
            self._counts,
            sampling_info.req_pool_indices,
            tokens,
            torch.ones_like(tokens, dtype=torch.int32),
        )

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, tokens
            )

        bs = logits.shape[0]
        return tokens, self._ones_buf[:bs]

    def _sample_npu_top_k_top_p(
        self,
        logits: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply top-k/top-p sampling via fused NPU op or Gumbel-max fallback.

        Args:
            logits: ``[bs, V]`` temperature-scaled logits.
            top_ks: ``[bs]`` int32 top-k values.
            top_ps: ``[bs]`` float32 top-p values.
            seeds: ``[bs]`` int64 per-request seeds.
            offsets: ``[bs]`` int64 per-request offsets (valid cache lengths).

        Returns:
            Sampled token ids ``[bs]`` int32.
        """
        impl = _get_npu_top_k_top_p_sample()
        if impl is not None:
            # torch_npu fused op: top_p dtype must match logits dtype
            # (float16/bfloat16 for half, float32 for float32).
            try:
                _top_ps = top_ps.to(logits.dtype) if top_ps.dtype != logits.dtype else top_ps
                sampled = impl(
                    logits, top_ks, _top_ps, post_sample="multiNomial", generator=None
                )
                # API returns (sampled_ids, filtered_logits) tuple
                if isinstance(sampled, (list, tuple)):
                    sampled = sampled[0]
                return sampled.to(torch.int32)
            except (RuntimeError, TypeError) as e:
                # Fallback if fused op fails
                pass

        # Gumbel-max fallback
        return self._gumbel_max_sample(logits, top_ks, top_ps, seeds, offsets)

    def _gumbel_max_sample(
        self,
        logits: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Gumbel-max top-k + top-p sampling (pure torch).

        Args:
            logits: ``[bs, V]`` float32.
            top_ks: ``[bs]`` int32.
            top_ps: ``[bs]`` float32.
            seeds: ``[bs]`` int64 (unused in pure-torch path).
            offsets: ``[bs]`` int64 (unused).

        Returns:
            Sampled token ids ``[bs]`` int32.
        """
        del seeds, offsets  # Unused in pure-torch path

        probs = torch.softmax(logits, dim=-1)
        bs, V = probs.shape

        # Top-k filtering
        k = top_ks.int()
        sorted_vals, sorted_idx = probs.sort(dim=-1, descending=True)
        batch_idx = torch.arange(bs, device=probs.device).unsqueeze(-1)
        topk_mask = torch.arange(V, device=probs.device).unsqueeze(0) < k.unsqueeze(-1)
        filtered = torch.zeros_like(probs)
        filtered[batch_idx, sorted_idx] = sorted_vals * topk_mask
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Top-p filtering
        sorted_vals, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_vals.cumsum(dim=-1)
        topp_mask = cumsum <= top_ps.unsqueeze(-1)
        topp_mask[..., 0] = True  # Always keep at least one token
        filtered = torch.zeros_like(probs)
        filtered[batch_idx, sorted_idx] = sorted_vals * topp_mask
        probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Gumbel-max
        uniform = torch.rand_like(probs)
        gumbel = -torch.log(-torch.log(uniform.clamp(min=1e-10)))
        scores = torch.log(probs.clamp(min=1e-10)) + gumbel
        return scores.argmax(dim=-1).to(torch.int32)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    @nvtx_range("sampling:ascend_verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]

        predict = self._predict_buf[: bs * num_tokens_per_req]
        accept_index = (
            self._accept_index_buf[: bs * num_tokens_per_req]
            .view(bs, num_tokens_per_req)
            .fill_(-1)
        )
        accept_length = self._accept_length_buf[:bs]

        logits = logits_output.next_token_logits.float()

        # Grammar bitmask
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=sampling_info.vocab_mask,
            )

        # Apply penalties
        logits = _apply_penalties_and_bias(
            logits,
            sampling_info,
            counts=self._counts,
            rep_pen_pool=self._rep_pen_pool,
            freq_pen_pool=self._freq_pen_pool,
            pres_pen_pool=self._pres_pen_pool,
            logit_bias=self._logit_bias,
            num_tokens_per_req=num_tokens_per_req,
        )

        # Temperature scaling via _gather_scalars (pure torch)
        temperatures = _gather_scalars(
            sampling_info.req_pool_indices,
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            n=num_tokens_per_req,
        )[0]
        logits = logits / temperatures[:, None]

        # Argmax target predictions
        target_predict = (
            logits.reshape(-1, logits.shape[-1])
            .argmax(dim=-1)
            .to(torch.int32)
            .reshape(bs, num_tokens_per_req)
        )

        # Chain-greedy verify (pure torch)
        _verify_chain_greedy_torch(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int32),
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=num_tokens_per_req,
        )

        accept_length += 1

        # TP-rank sync
        self.maybe_broadcast(predict, accept_index, accept_length)

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, predict
            )

        return predict, accept_length


register_backend("ascend", AscendSamplingBackend)
