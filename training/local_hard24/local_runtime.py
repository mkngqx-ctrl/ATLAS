"""Windows backend compatibility for the installed GPU environment."""
import functools
import os


def configure_dynamic_deephit():
    """Keep DDH on CUDA, using PyTorch's native kernels instead of cuDNN.

    On this Windows / torch 2.11.0+cu130 installation, a tiny DDH CUDA run
    completes but exits with 0xC0000409 when cuDNN is enabled. The same test
    exits normally with cuDNN disabled. Scope the fallback to DDH's forward
    graph; its backward graph consequently uses the corresponding kernels.
    Do not bypass interpreter cleanup or suppress process failures.
    """
    if os.name != 'nt':
        return {'dynamic_deephit_cudnn': 'default'}
    import torch
    from history_models import DynamicDeepHitHistory
    original = DynamicDeepHitHistory.forward
    if not getattr(original, '_atlas_native_cuda', False):
        @functools.wraps(original)
        def forward(self, *args, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return original(self, *args, **kwargs)
        forward._atlas_native_cuda = True
        DynamicDeepHitHistory.forward = forward
    return {'dynamic_deephit_cudnn': False,
            'reason': 'Windows GPU compatibility; DDH uses native PyTorch CUDA kernels'}
