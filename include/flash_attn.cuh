#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================
// FlashAttention Kernel 接口
// ============================================================
// 所有 kernel 版本共用同一个接口，方便切换和对比
//
// 参数说明:
//   Q, K, V: 输入，shape = [B, H, N, d]，BF16 格式
//   O:       输出，shape = [B, H, N, d]，BF16 格式
//   l, m:    online softmax 的中间量（logsumexp, row max）
//   B:       batch size
//   H:       number of heads
//   N:       sequence length
//   d:       head dimension
//   scale:   softmax 缩放因子，通常是 1/sqrt(d)

// Phase 1: Naive (global memory load + mma.sync)
void flash_attn_naive_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* l,    // [B, H, N] logsumexp (for backward)
    float* m,    // [B, H, N] row max (for backward)
    int B, int H, int N, int d,
    float scale,
    cudaStream_t stream = 0
);

// Phase 2: TMA version (后面加)
// void flash_attn_tma_launch(...);

// Phase 3: Pipelined version (后面加)
// void flash_attn_pipe_launch(...);
