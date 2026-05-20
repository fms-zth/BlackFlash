import torch
import math
import blackflash_cuda

def blackflash(Q, K, V, scale=None):
    """
    BlackFlash: Flash Attention forward pass.

    Args:
        Q, K, V: [B, H, N, d] bfloat16 CUDA tensors, d must be 64
        scale: float, default 1/sqrt(d)

    Returns:
        O: [B, H, N, d] bfloat16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))
    return blackflash_cuda.forward(Q, K, V, scale)