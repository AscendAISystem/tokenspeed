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

from __future__ import annotations

from contextlib import contextmanager

import torch

from tokenspeed.runtime.utils.device_utils import get_device_module


class StreamFork:
    def __init__(self, aux_stream: torch.npu.Stream | torch.cuda.Stream | None):
        self.aux_stream = aux_stream
        self._dev = get_device_module()
        self.fork_event = self._dev.Event() if aux_stream is not None else None
        self.join_event = self._dev.Event() if aux_stream is not None else None
        self._active = False
        self._current: torch.npu.Stream | torch.cuda.Stream | None = None

    @contextmanager
    def scope(self, *, enable: bool):
        self._active = enable and self.aux_stream is not None
        if self._active:
            self._current = self._dev.current_stream()
            self.fork_event.record(self._current)
        try:
            yield self
        finally:
            if self._active:
                self.join_event.wait(self._current)
                self._active = False
                self._current = None

    @contextmanager
    def branch(self):
        if not self._active:
            yield
            return
        with self._dev.stream(self.aux_stream):
            self.fork_event.wait(self.aux_stream)
            yield
            self.join_event.record(self.aux_stream)
