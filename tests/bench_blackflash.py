import torch
import math

from blackflash import blackflash
from torch.nn.functional import scaled_dot_product_attention

def benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / iters
    return avg_ms

def main():
    B, H, N, d = 64,128,1024,64
    scale = 1.0 / math.sqrt(d)

    Q = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")

    # 运行一次拿输出
    O_black = blackflash(Q, K, V, scale)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        O_cudnn = scaled_dot_product_attention(Q, K, V, is_causal=False, scale=scale)

    # 精度对比：前4行8列
    print("=" * 50)
    print(f"Output comparison (B=0, H=0, first 4 rows x 8 cols)")
    print("=" * 50)
    print("BlackFlash:")
    print(O_black[0, 0, :4, :8].float())
    print("cuDNN:")
    print(O_cudnn[0, 0, :4, :8].float())
    print("Diff:")
    print((O_black[0, 0, :4, :8].float() - O_cudnn[0, 0, :4, :8].float()).abs())

    max_diff = (O_black.float() - O_cudnn.float()).abs().max().item()
    print(f"\nMax abs error (full tensor): {max_diff:.6f}")

    # 性能对比
    flops = 4.0 * B * H * N * N * d

    ms_black = benchmark(lambda: blackflash(Q, K, V, scale))
    tflops_black = flops / (ms_black * 1e-3) / 1e12

    def cudnn_fn():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)

    ms_cudnn = benchmark(cudnn_fn)
    tflops_cudnn = flops / (ms_cudnn * 1e-3) / 1e12

    print("\n" + "=" * 50)
    print(f"Performance (B={B}, H={H}, N={N}, d={d})")
    print("=" * 50)
    print(f"BlackFlash : {ms_black:.3f} ms  |  {tflops_black:.2f} TFLOPS")
    print(f"cuDNN      : {ms_cudnn:.3f} ms  |  {tflops_cudnn:.2f} TFLOPS")
    print(f"Ratio      : {tflops_black/tflops_cudnn:.2f}x")

if __name__ == "__main__":
    main()