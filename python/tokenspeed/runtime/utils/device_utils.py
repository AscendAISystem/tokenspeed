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

"""Device abstraction utilities for NPU/CUDA interoperability.

Provides a unified interface for device detection, selection, and tensor
placement.  All runtime code should go through these helpers instead of
directly referencing ``torch.cuda`` or ``torch.npu``.
"""

from __future__ import annotations

import torch


def is_npu_available() -> bool:
    """Check whether an Ascend NPU is available.

    Returns:
        ``True`` if ``torch_npu`` is installed and at least one NPU is
        visible to the current process.
    """
    try:
        # Direct import ensures torch.npu is patched in
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def get_device_module():
    """Return the active device module (``torch.npu`` or ``torch.cuda``).

    On a system with a usable NPU :func:`is_npu_available` returns
    ``torch.npu``; otherwise falls back to ``torch.cuda`` (which may also
    be absent if no accelerator is present).
    """
    if is_npu_available():
        return torch.npu

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda

    # Last resort – return torch.cuda even if unavailable; callers should
    # guard with is_npu_available() / torch.cuda.is_available().
    return torch.cuda


def get_current_device() -> torch.device:
    """Return the current active device as a :class:`torch.device`.

    The device type is ``"npu"`` on NPU platforms and ``"cuda"`` otherwise.
    """
    dev = get_device_module()
    idx = dev.current_device()
    if is_npu_available():
        return torch.device(f"npu:{idx}")
    return torch.device(f"cuda:{idx}")


def to_device(
    tensor: torch.Tensor,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Move *tensor* to the appropriate accelerator device.

    When *device* is ``None`` the destination is chosen automatically:
    NPU if available, otherwise CUDA, otherwise CPU.
    """
    if device is not None:
        return tensor.to(device)

    if is_npu_available():
        return tensor.to("npu")
    if torch.cuda.is_available():
        return tensor.to("cuda")
    return tensor


def device_context(device: torch.device | None = None):
    """Context manager that sets the default device.

    Signature matches ``torch.cuda.device`` / ``torch.npu.device``.
    """
    dev_mod = get_device_module()
    return dev_mod.device(device)
