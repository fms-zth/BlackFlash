import torch
from blackflash import blackflash

# 参数
B, H, N, d = 1, 1, 256, 64

# 随机生成 QKV，bfloat16，CUDA
Q = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")
K = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")
V = torch.randn(B, H, N, d, dtype=torch.bfloat16, device="cuda")

# BlackFlash
O_blackflash = blackflash(Q, K, V)

# PyTorch reference
scale = 1.0 / (d ** 0.5)
S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scale
P = torch.softmax(S, dim=-1)
O_ref = torch.matmul(P, V.float()).to(torch.bfloat16)

# 比较
diff = (O_blackflash.float() - O_ref.float()).abs()
print(f"Max abs error : {diff.max().item():.6f}")
print(f"Mean abs error: {diff.mean().item():.6f}")
print(f"Status: {'PASS' if diff.max().item() < 0.05 else 'FAIL'}")