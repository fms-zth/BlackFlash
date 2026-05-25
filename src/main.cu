#include "flash_attn.cuh"
#include "utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>

// ============================================================
// bf16 <-> float host helpers
// ============================================================
static void host_bf16_to_float(const __nv_bfloat16 *src, float *dst, int n)
{
    for (int i = 0; i < n; i++)
    {
        uint16_t bf16_bits;
        memcpy(&bf16_bits, &src[i], 2);
        uint32_t bits = ((uint32_t)bf16_bits) << 16;
        memcpy(&dst[i], &bits, 4);
    }
}

// ============================================================
// 读取 meta.txt 获取 B, H, N, d
// ============================================================
static bool read_meta(const char *dir, int &B, int &H, int &N, int &d, float &scale)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return false;
    }

    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        sscanf(line, "B=%d", &B);
        sscanf(line, "H=%d", &H);
        sscanf(line, "N=%d", &N);
        sscanf(line, "d=%d", &d);
        sscanf(line, "scale=%f", &scale);
    }
    fclose(f);
    return true;
}

// ============================================================
// 读取 binary 文件
// ============================================================
static bool read_bin(const char *dir, const char *name, void *dst, size_t bytes)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return false;
    }
    size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes)
    {
        fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, bytes, got);
        return false;
    }
    return true;
}

// ============================================================
// 对比函数: GPU output vs reference
// ============================================================
static void compare(const char *label, const float *gpu, const float *ref, int rows, int cols, int total, float tol)
{
    float max_err = 0.0f;
    int max_r = 0, max_c = 0;
    double sum_err = 0.0, sum_abs_ref = 0.0;
    int bad_count = 0;

    for (int idx = 0; idx < total; idx++)
    {
        float err = fabsf(gpu[idx] - ref[idx]);
        if (err > max_err)
        {
            max_err = err;
            max_r = idx / cols;
            max_c = idx % cols;
        }
        sum_err += err;
        sum_abs_ref += fabsf(ref[idx]);
        if (err > tol)
            bad_count++;
    }

    printf("\n--- %s comparison (%d elements) ---\n", label, total);
    printf("Max abs error : %.6f  (at row=%d, col=%d)\n", max_err, max_r, max_c);
    printf("Avg abs error : %.6f\n", (float)(sum_err / total));
    printf("Mean |ref|    : %.6f  (relative error: %.4f%%)\n", (float)(sum_abs_ref / total),
           sum_abs_ref > 0 ? (float)(sum_err / sum_abs_ref * 100.0) : 0.0f);
    printf("Bad elements  : %d / %d (tol=%.3f)\n", bad_count, total, tol);
    printf("Status: %s\n", max_err < tol ? "PASS ✓" : "FAIL ✗");

    // 打印前 4 行对比
    printf("\nRef[0:4][0:8]:\n");
    int print_rows = (rows < 4) ? rows : 4;
    int print_cols = (cols < 8) ? cols : 8;
    for (int i = 0; i < print_rows; i++)
    {
        for (int j = 0; j < print_cols; j++)
            printf("%9.4f ", ref[i * cols + j]);
        printf("...\n");
    }
    printf("GPU[0:4][0:8]:\n");
    for (int i = 0; i < print_rows; i++)
    {
        for (int j = 0; j < print_cols; j++)
            printf("%9.4f ", gpu[i * cols + j]);
        printf("...\n");
    }

    // 失败时打印 max error 那一行
    if (max_err >= tol)
    {
        int dbg_cols = (cols < 16) ? cols : 16;
        printf("\n[Debug] Row %d (first %d cols):\n", max_r, dbg_cols);
        printf("  Ref: ");
        for (int j = 0; j < dbg_cols; j++)
            printf("%8.4f ", ref[max_r * cols + j]);
        printf("\n  GPU: ");
        for (int j = 0; j < dbg_cols; j++)
            printf("%8.4f ", gpu[max_r * cols + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    // ref_data 目录路径, 默认 "ref_data"
    const char *ref_dir = "ref_data";
    if (argc >= 2)
        ref_dir = argv[1];

    // ============================================================
    // 1. 读取 meta
    // ============================================================
    int B = 0, H = 0, N = 0, d = 0;
    float scale = 0.0f;
    if (!read_meta(ref_dir, B, H, N, d, scale))
        return 1;

    printf("============================================\n");
    printf("BlackFlash TEST (vs PyTorch FA2 reference)\n");
    printf("B=%d, H=%d, N=%d, d=%d, scale=%.6f\n", B, H, N, d, scale);
    printf("ref_dir=%s\n", ref_dir);
    printf("============================================\n");

    int qkv_elems = B * H * N * d;
    int s_elems = B * H * N * N; // S 矩阵
    int lm_elems = B * H * N;

    // ============================================================
    // 2. 读取 binary 文件
    // ============================================================
    std::vector<__nv_bfloat16> h_Q(qkv_elems), h_K(qkv_elems), h_V(qkv_elems);
    std::vector<__nv_bfloat16> h_O_gpu_bf16(qkv_elems);
    std::vector<float> h_O_gpu(qkv_elems);

    // reference 数据
    std::vector<__nv_bfloat16> h_O_ref_bf16(qkv_elems);
    std::vector<float> h_O_ref(qkv_elems);
    std::vector<float> h_S_ref(s_elems);

    printf("\nLoading binary files...\n");
    if (!read_bin(ref_dir, "q.bin", h_Q.data(), qkv_elems * sizeof(__nv_bfloat16)))
        return 1;
    if (!read_bin(ref_dir, "k.bin", h_K.data(), qkv_elems * sizeof(__nv_bfloat16)))
        return 1;
    if (!read_bin(ref_dir, "v.bin", h_V.data(), qkv_elems * sizeof(__nv_bfloat16)))
        return 1;
    if (!read_bin(ref_dir, "o_ref.bin", h_O_ref_bf16.data(), qkv_elems * sizeof(__nv_bfloat16)))
        return 1;
    if (!read_bin(ref_dir, "s_ref.bin", h_S_ref.data(), s_elems * sizeof(float)))
        return 1;
    printf("All files loaded.\n");

    // O_ref: bf16 -> float
    host_bf16_to_float(h_O_ref_bf16.data(), h_O_ref.data(), qkv_elems);

    // ============================================================
    // 3. GPU: 运行 kernel

    printf("\nQ[0:4] from bin file:\n");
    float tmp[8];
    host_bf16_to_float(h_Q.data(), tmp, 8);
    for (int i = 0; i < 8; i++)
        printf("  Q[%d] = %.6f\n", i, tmp[i]);

    // ============================================================
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    float *d_l, *d_m;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_l, lm_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, lm_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, qkv_elems * sizeof(__nv_bfloat16)));

    printf("\nRunning kernel...\n");
    flash_attn_naive_launch(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, d, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel done.\n");

    // ============================================================
    // Benchmark: warmup + timed runs
    // ============================================================
    {
        const int warmup_iters = 2;
        const int bench_iters = 10;

        // warmup
        for (int i = 0; i < warmup_iters; i++)
        {
            flash_attn_naive_launch(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, d, scale);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // timed runs
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; i++)
        {
            flash_attn_naive_launch(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, d, scale);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / bench_iters;

        // FLOPs for standard attention: 2 matmuls of [N,d]x[d,N] and [N,N]x[N,d]
        // each matmul = 2*M*N*K FLOPs => total = 2*(2*N*N*d) = 4*N*N*d per head
        double flops_per_run = 4.0 * B * H * (double)N * N * d;
        double tflops = flops_per_run / (avg_ms * 1e-3) / 1e12;

        double peak_tflops = 36.13;

        double utilization = tflops / peak_tflops * 100.0;

        printf("\n============================================\n");
        printf("Performance Benchmark (%d iters)\n", bench_iters);
        printf("============================================\n");
        printf("Avg latency      : %.3f ms\n", avg_ms);
        printf("Achieved         : %.2f TFLOPS\n", tflops);
        printf("Peak (bf16 dense): %.1f TFLOPS\n", peak_tflops);
        printf("Utilization      : %.2f%%\n", utilization);
        printf("============================================\n");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaMemcpy(h_O_gpu_bf16.data(), d_O, qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    host_bf16_to_float(h_O_gpu_bf16.data(), h_O_gpu.data(), qkv_elems);

    // ============================================================
    // 4. 对比
    // ============================================================
    // 检测当前 kernel 模式:
    //   - 如果 kernel 有 early return (debug S 模式), O 输出的其实是 S
    //     此时 S 是 [N, d=64] 的 layout, 但 S_ref 是 [N, N] 的 layout
    //     只有 N==d 时 shape 才一致，否则需要特殊处理
    //   - 如果 kernel 跑完整 FA2, O 输出就是 attention output
    //
    // 这里两个都比：
    //   (a) 把 GPU output 当作 S, 与 S_ref 的前 d 列比 (debug 模式)
    //   (b) 把 GPU output 当作 O, 与 O_ref 比 (完整模式)

    // --- (a) S 对比 (debug 模式: kernel 把 S 写进了 O 的位置) ---
    // GPU S 的 layout: [N, d] (因为 O 的 buffer 是 [N, d])
    // Ref S 的 layout: [N, N]
    // 当 N >= d 时, GPU 只写了 S 的前 d 列
    // 我们比较 S_ref[i][0..d-1] vs GPU[i][0..d-1]
    printf("\n========== S matrix comparison (debug mode) ==========\n");
    printf("(Compares GPU output as S against S_ref, first %d columns)\n", d);
    {
        // 提取 S_ref 的前 d 列, 拼成 [N, d] 的 flat array
        int compare_cols = (N < d) ? N : d;
        std::vector<float> s_ref_trimmed(N * compare_cols);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < compare_cols; j++)
            {
                s_ref_trimmed[i * compare_cols + j] = h_S_ref[i * N + j];
            }
        }
        // GPU output 也只取前 compare_cols 列
        std::vector<float> s_gpu_trimmed(N * compare_cols);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < compare_cols; j++)
            {
                s_gpu_trimmed[i * compare_cols + j] = h_O_gpu[i * d + j];
            }
        }
        compare("S (debug)", s_gpu_trimmed.data(), s_ref_trimmed.data(), N, compare_cols, N * compare_cols, 0.05f);
    }

    // --- (b) O 对比 (完整 FA2 模式) ---
    printf("\n========== O matrix comparison (full FA2 mode) ==========\n");
    compare("O (FA2)", h_O_gpu.data(), h_O_ref.data(), N, d, qkv_elems, 0.02f);

    // ============================================================
    // 5. 清理
    // ============================================================
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_l));
    CUDA_CHECK(cudaFree(d_m));

    return 0;
}
