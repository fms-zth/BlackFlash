"""
gen_ref.py  —  BlackFlash 精度测试: 生成 reference 数据
==========================================================
功能:
  1. 生成随机 bf16 的 Q, K, V  (shape: [B, H, N, d])
  2. 用 PyTorch 官方 FA2 计算 O_ref
  3. 用 naive matmul 计算 S_ref = Q @ K^T * scale  (fp32)
  4. 用 naive softmax 计算 P_ref = softmax(S_ref)    (fp32)
  5. 全部存成 raw binary 文件, 供 C++ 侧 fread 读取

输出文件 (默认在 ./ref_data/ 目录):
  q.bin       — [B*H*N*d] 个 bf16
  k.bin       — [B*H*N*d] 个 bf16
  v.bin       — [B*H*N*d] 个 bf16
  o_ref.bin   — [B*H*N*d] 个 bf16   (FA2 输出)
  s_ref.bin   — [B*H*N*N] 个 fp32   (naive Q@K^T * scale)
  p_ref.bin   — [B*H*N*N] 个 fp32   (naive softmax(S))

用法:
  python gen_ref.py                        # 默认 B=1 H=1 N=64 d=64
  python gen_ref.py --B 1 --H 1 --N 256 --d 64
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Generate FA2 reference data")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="ref_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    B, H, N, d = args.B, args.H, args.N, args.d
    scale = 1.0 / math.sqrt(d)

    print(f"Generating reference data: B={B}, H={H}, N={N}, d={d}, scale={scale:.6f}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, FA2 will fallback to math implementation")

    # ----------------------------------------------------------------
    # 1. 生成随机 bf16 输入, 范围 [-0.5, 0.5] (与你 C++ 端 fill_random 一致)
    # ----------------------------------------------------------------
    Q = (torch.rand(B, H, N, d, device=device) - 0.5).to(torch.bfloat16)
    K = (torch.rand(B, H, N, d, device=device) - 0.5).to(torch.bfloat16)
    V = (torch.rand(B, H, N, d, device=device) - 0.5).to(torch.bfloat16)

    print(f"  Q shape: {Q.shape}, dtype: {Q.dtype}")
    print(f"  K shape: {K.shape}, dtype: {K.dtype}")
    print(f"  V shape: {V.shape}, dtype: {V.dtype}")

    # ----------------------------------------------------------------
    # 2. 用 PyTorch FA2 算 O_ref
    #    enable_math=False 强制走 flash / memory-efficient kernel
    # ----------------------------------------------------------------
    with torch.nn.attention.sdpa_kernel([
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    ]):
        O_ref = F.scaled_dot_product_attention(Q, K, V, scale=scale)

    O_ref_bf16 = O_ref.to(torch.bfloat16)
    print(f"  O_ref shape: {O_ref_bf16.shape}, dtype: {O_ref_bf16.dtype}")

    # ----------------------------------------------------------------
    # 3. Naive 计算 S 和 P (用 fp32, 精度最高)
    #    S = Q @ K^T * scale    shape: [B, H, N, N]
    #    P = softmax(S, dim=-1) shape: [B, H, N, N]
    # ----------------------------------------------------------------
    Q_f32 = Q.float()
    K_f32 = K.float()

    S_ref = torch.matmul(Q_f32, K_f32.transpose(-2, -1)) * scale  # [B, H, N, N]
    P_ref = torch.softmax(S_ref, dim=-1)                          # [B, H, N, N]

    print(f"\n  K[0,0, 0:2, 0:4] (raw bf16 values):")
    print(f"  K[0,0,0,:] = {K[0,0,0,:4].float().tolist()}")
    print(f"  K[0,0,1,:] = {K[0,0,1,:4].float().tolist()}")

    S = (Q[0,0].float() @ K[0,0].float().T) * scale
    print("S[0,:8] =", S[0,:8].tolist())   # query行0，KV列0~7
    print("S[8,:8] =", S[8,:8].tolist())   # query行8，KV列0~7


    S = (Q[0,0].float() @ K[0,0].float().T) * scale
    print(f"S[0][0]={S[0][0].item():.4f}, S[0][1]={S[0][1].item():.4f}")
    print(f"S[8][0]={S[8][0].item():.4f}, S[8][1]={S[8][1].item():.4f}")


    print("Q行0, 列0~1:", Q[0,0,0,0].item(), Q[0,0,0,1].item())
    print("Q行8, 列0~1:", Q[0,0,8,0].item(), Q[0,0,8,1].item())

    print("K行0, 列0~1:", K[0,0,0,0].item(), K[0,0,0,1].item())
    print("K行1, 列0~1:", K[0,0,1,0].item(), K[0,0,1,1].item())


    import numpy as np
    S = (Q[0,0].float() @ K[0,0].float().T) * scale
    S_np = S.cpu().numpy()
    target = [0.1093, -0.0380, -0.1902, -0.0798]
    tol = 0.002

    # 找每个target在S中的候选位置
    def find_candidates(val, tol):
        rows, cols = np.where(np.abs(S_np - val) < tol)
        return list(zip(rows.tolist(), cols.tolist()))

    c0 = find_candidates(target[0], tol)
    c1 = find_candidates(target[1], tol)
    c2 = find_candidates(target[2], tol)
    c3 = find_candidates(target[3], tol)

    print(f"target[0]={target[0]} candidates: {c0}")
    print(f"target[1]={target[1]} candidates: {c1}")
    print(f"target[2]={target[2]} candidates: {c2}")
    print(f"target[3]={target[3]} candidates: {c3}")

    # 找满足 row0相同, row1相同, col0相同, col1相同 的组合
    for r0,ca0 in c0:
        for r0b,ca1 in c1:
            if r0 != r0b: continue
            for r1,cb0 in c2:
                if ca0 != cb0: continue
                for r1b,cb1 in c3:
                    if r1 != r1b or ca1 != cb1: continue
                    print(f"MATCH: row0={r0}, row1={r1}, col0={ca0}, col1={ca1}")


    print(f"  S_ref shape: {S_ref.shape}, dtype: {S_ref.dtype}")
    print(f"  P_ref shape: {P_ref.shape}, dtype: {P_ref.dtype}")

    # ----------------------------------------------------------------
    # 4. 打印前几个值, 方便 sanity check
    # ----------------------------------------------------------------
    print(f"\n  S_ref[0,0, 0:4, 0:8]:")
    for i in range(min(4, N)):
        vals = " ".join(f"{S_ref[0,0,i,j].item():9.4f}" for j in range(min(8, N)))
        print(f"    {vals}")

    print(f"\n  O_ref[0,0, 0:4, 0:8]:")
    for i in range(min(4, N)):
        vals = " ".join(f"{O_ref_bf16[0,0,i,j].float().item():9.4f}" for j in range(min(8, d)))
        print(f"    {vals}")

    # ----------------------------------------------------------------
    # 5. 存成 raw binary
    #    内存布局: [B, H, N, d] contiguous, C-order → 与你 C++ 的 BHND 一致
    # ----------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    def save_tensor(tensor, filename, dtype_name):
        t = tensor.contiguous().cpu()
        path = os.path.join(args.outdir, filename)
        # 用 numpy 的 tofile 写 raw binary
        t.numpy().tofile(path)
        nbytes = os.path.getsize(path)
        print(f"  Saved {filename:16s}  {dtype_name:8s}  {nbytes:>10d} bytes")

    print(f"\nSaving to {args.outdir}/")

    # bf16 tensors: 需要先 view 成 uint16 再转 numpy (numpy 不支持 bf16)
    def save_bf16(tensor, filename):
        t = tensor.contiguous().cpu()
        raw = t.view(torch.uint16).numpy()
        path = os.path.join(args.outdir, filename)
        raw.tofile(path)
        nbytes = os.path.getsize(path)
        print(f"  Saved {filename:16s}  bf16      {nbytes:>10d} bytes")

    save_bf16(Q, "q.bin")
    save_bf16(K, "k.bin")
    save_bf16(V, "v.bin")
    save_bf16(O_ref_bf16, "o_ref.bin")
    save_tensor(S_ref.float(), "s_ref.bin", "fp32")
    save_tensor(P_ref.float(), "p_ref.bin", "fp32")

    # ----------------------------------------------------------------
    # 6. 保存元信息, 方便 C++ 侧校验
    # ----------------------------------------------------------------
    meta_path = os.path.join(args.outdir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"B={B}\n")
        f.write(f"H={H}\n")
        f.write(f"N={N}\n")
        f.write(f"d={d}\n")
        f.write(f"scale={scale:.10f}\n")
        f.write(f"seed={args.seed}\n")
    print(f"  Saved {'meta.txt':16s}")

    print("\nDone! Next step: C++ 侧 fread 这些 .bin 文件来对比")


if __name__ == "__main__":
    main()
