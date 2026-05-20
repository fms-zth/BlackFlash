#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
//ldmatric宏函数
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"     \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr)                                                            \
    )

#define LDMATRIX_X2(R0, R1, addr)                                              \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"           \
        : "=r"(R0), "=r"(R1)                                                  \
        : "r"(addr)                                                            \
    )

#define LDMATRIX_X4_TRANS(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr) \
    )    
#define HMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
        : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3) \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), \
          "r"(RB0), "r"(RB1), \
          "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3) \
    )


#define LDMATRIX_X2_TRANS(R0, R1, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1) \
        : "r"(addr) \
    )
// ============================================================
// CUDA 错误检查宏
// ============================================================
// 每次调 CUDA API 后用这个包一下，出错立刻报错退出
// 用法: CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// Kernel launch 错误检查
// ============================================================
// kernel<<<...>>>(...) 后面紧跟这个，检查 launch 有没有出错
#define CUDA_KERNEL_CHECK()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// 计时工具（用 CUDA Event，比 CPU 计时更准）
// ============================================================
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    // 返回经过的时间（毫秒）
    float end(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================
// 简单的数值比较工具
// ============================================================
// 比较两个 float 数组，返回最大绝对误差和平均绝对误差
struct CompareResult {
    float max_abs_err;
    float avg_abs_err;
    int   total_elements;
    bool  passed;        // max_abs_err < tolerance
};

inline CompareResult compare_arrays(const float* ref, const float* test,
                                     int n, float tolerance = 1e-2f) {
    CompareResult result = {0.0f, 0.0f, n, true};
    double sum_err = 0.0;

    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > result.max_abs_err) result.max_abs_err = err;
        sum_err += err;
    }

    result.avg_abs_err = (float)(sum_err / n);
    result.passed = (result.max_abs_err < tolerance);
    return result;
}

// ============================================================
// 随机初始化（用 host 端生成，简单够用）
// ============================================================
inline void fill_random(float* data, int n, float low = -1.0f, float high = 1.0f) {
    for (int i = 0; i < n; i++) {
        data[i] = low + static_cast<float>(rand()) / RAND_MAX * (high - low);
    }
}

// ============================================================
// 打印矩阵（debug 用，只打印左上角一小块）
// ============================================================
inline void print_matrix(const float* data, int rows, int cols,
                         int max_rows = 4, int max_cols = 4,
                         const char* name = "Matrix") {
    printf("\n%s [%d x %d]:\n", name, rows, cols);
    for (int i = 0; i < (rows < max_rows ? rows : max_rows); i++) {
        for (int j = 0; j < (cols < max_cols ? cols : max_cols); j++) {
            printf("%8.4f ", data[i * cols + j]);
        }
        if (cols > max_cols) printf(" ...");
        printf("\n");
    }
    if (rows > max_rows) printf("  ...\n");

    
}
