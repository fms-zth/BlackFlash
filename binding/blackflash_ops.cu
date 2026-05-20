#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
// 声明你的 launch 函数（在 flash_attn_mma.cu 中定义）
void flash_attn_naive_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* l,
    float* m,
    int B, int H, int N, int d,
    float scale,
    cudaStream_t stream
);

torch::Tensor blackflash_forward(
    torch::Tensor Q,   // [B, H, N, d], bfloat16
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(K.is_cuda() && K.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(V.is_cuda() && V.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    TORCH_CHECK(d == 64, "BlackFlash currently only supports d=64");

    auto O = torch::empty_like(Q);
    auto l = torch::zeros({B, H, N}, Q.options().dtype(torch::kFloat32));
    auto m = torch::full({B, H, N}, -INFINITY, Q.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    flash_attn_naive_launch(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        B, H, N, d,
        scale,
        stream
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blackflash_forward, "BlackFlash forward (BF16, d=64)");
}