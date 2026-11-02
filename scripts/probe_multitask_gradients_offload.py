#!/usr/bin/env python3
"""Run the unchanged gradient probe with selective CPU saved-tensor storage.

Use the same arguments as scripts/probe_multitask_gradients.py. The offload is
active only while the loaded model's forward runs, including both forwards in
the optional total.backward check. Only contiguous, zero-offset CUDA tensors
outside model parameter storage are copied. This keeps checkpoint boundary
inputs with other strides intact when PyTorch replays their forward.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import probe_multitask_gradients as original_probe
from unilip.train import gradient_probe_loader


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offload_metadata() -> dict:
    """Describe the exact offload boundary without changing the report schema."""
    return {
        "enabled": True,
        "mechanism": "torch.autograd.graph.saved_tensors_hooks",
        "pin_memory": True,
        "scope": "model.forward saved contiguous zero-offset CUDA non-parameter tensors, including total.backward replay forwards",
        "resident_passthrough": "noncontiguous, nonzero-offset, non-CUDA, and model-parameter-storage tensors",
        "checkpoint_internals": "PyTorch non-reentrant checkpoint hooks take precedence",
        "entrypoint": "scripts/probe_multitask_gradients_offload.py",
        "entrypoint_sha256": _sha256(Path(__file__)),
        "original_probe_sha256": _sha256(Path(original_probe.__file__)),
        "torch_version": torch.__version__,
    }


@contextmanager
def selective_saved_tensors_offload(model: torch.nn.Module, *, device_type="cuda"):
    """Keep stride-sensitive saved tensors on their device for checkpoint replay.

    Non-reentrant checkpoint saves its input tensors under the surrounding
    hooks. PyTorch 2.11 save_on_cpu(pin_memory=True) packs those using
    torch.empty(size), which turns noncontiguous inputs into contiguous ones.
    Recomputed operations can then choose a different graph and fail the
    checkpoint metadata check. Passing those inputs through retains strides.
    """
    parameter_storages = {(p.device, p.untyped_storage().data_ptr())
                          for p in model.parameters()}

    def pack(tensor):
        if (tensor.device.type != device_type or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tensor.storage_offset() != 0
                or (tensor.device, tensor.untyped_storage().data_ptr()) in parameter_storages):
            return (False, tensor.detach())
        pinnable = tensor.device.type == "cuda" and torch.cuda.is_available()
        packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                             device="cpu", pin_memory=pinnable)
        packed.copy_(tensor)
        return (True, tensor.device, packed)

    def unpack(packed):
        offloaded, *rest = packed
        if not offloaded:
            return rest[0]
        device, cpu_tensor = rest
        return cpu_tensor.to(device, non_blocking=cpu_tensor.is_pinned())

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@contextmanager
def offload_model_forward(model: torch.nn.Module):
    """Wrap only the outer model forward, then restore its original attribute."""
    prior_instance_forward = model.__dict__.get("forward", _MISSING)
    original_forward = model.forward

    @wraps(original_forward)
    def forward_with_offload(*args, **kwargs):
        with selective_saved_tensors_offload(model):
            return original_forward(*args, **kwargs)

    model.forward = forward_with_offload
    try:
        yield model
    finally:
        if prior_instance_forward is _MISSING:
            delattr(model, "forward")
        else:
            model.forward = prior_instance_forward


_MISSING = object()


def main(argv=None):
    original_loader = gradient_probe_loader.load_gradient_probe_context
    with ExitStack() as stack:
        def load_with_offload(*args, **kwargs):
            context = original_loader(*args, **kwargs)
            stack.enter_context(offload_model_forward(context.model))
            context.metadata["saved_tensor_offload"] = offload_metadata()
            return context

        # The original probe imports this function within main(). Keep every
        # argument, measurement, check, vector, and report writer unchanged.
        with patch.object(gradient_probe_loader, "load_gradient_probe_context",
                          load_with_offload):
            return original_probe.main(argv)


if __name__ == "__main__":
    main()
