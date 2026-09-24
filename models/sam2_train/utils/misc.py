import warnings

import torch


def get_sdpa_settings():
    if not torch.cuda.is_available():
        return True, False, True
    capability = torch.cuda.get_device_properties(0).major
    old_gpu = capability < 7
    use_flash_attention = capability >= 8
    version = tuple(int(value) for value in torch.__version__.split(".")[:2])
    if not use_flash_attention:
        warnings.warn("Flash Attention requires an Ampere-or-newer GPU.", stacklevel=2)
    math_kernel_on = version < (2, 2) or not use_flash_attention
    return old_gpu, use_flash_attention, math_kernel_on

