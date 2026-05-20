# BlackFlash

A from-scratch, handwritten Flash Attention 2 CUDA kernel targeting NVIDIA Blackwell architecture (RTX 5060 Laptop, SM120). Fully exploits hardware features including TMA, MMA, shared memory swizzle, double buffering, and warp specialization — progressively optimized to near-peak performance.

## Performance

| Config (B, H, N, d) | BlackFlash (TFLOPS) | cuDNN (TFLOPS) | Ratio |
|---|---|---|---|
| 2, 2, 512, 64 | **4.54** | **4.42** | **1.05x** |
| 4, 8, 1024, 64 | 29.8 | 41.2 | 0.73x |
| 8, 16, 1024, 64 | **30.6** | **48.1** | **0.64x** |
| 64, 64, 1024, 64 | **31.45** | **51.14** | **0.62x** |

----

## Features

- Flash Attention 2 forward pass with online softmax and tiled computation
- TMA (Tensor Memory Accelerator) for asynchronous global-to-shared memory data movement
- MMA instructions for matrix multiply-accumulate on Tensor Cores
- 128B shared memory swizzle to eliminate bank conflicts
- Double buffering pipeline overlapping computation with data movement
- Warp specialization with producer/consumer division of labor

## Architecture & Pipeline

**Tile and thread configuration:**

- Br = Bc = d = 64
- Block size = 288 threads (9 warps total): 1 producer warp (32 threads) + 8 consumer warps (256 threads)
- 2-stage double buffering for K/V tiles (sK0/sV0 and sK1/sV1)

**Shared memory layout (per block):**

```
|--- sQ ---|--- sK0 ---|--- sV0 ---|--- sK1 ---|--- sV1 ---|--- barriers & scratch ---|
  Br × d      Bc × d      Bc × d      Bc × d      Bc × d       5 mbarriers + 4·Br floats
 (64×64)    (64×64)     (64×64)     (64×64)     (64×64)        (max/sum reduction)
```

All Q/K/V tiles are loaded via TMA with 128B swizzle addressing. The P matrix (softmax output) reuses the current K buffer (`sP = sK_cur`) to avoid extra shared memory allocation.

**Consumer warp organization:**

The 8 consumer warps are organized as a 4×2 grid. Each `warp_pair` (pair index 0–3) owns 16 rows of the Q tile. Within each pair, `warp_half` 0 and 1 split the column dimension, so every element of the 64×64 output tile is covered.

```
             half=0   half=1
  pair=0:   warp 0   warp 1    → Q rows  0–15
  pair=1:   warp 2   warp 3    → Q rows 16–31
  pair=2:   warp 4   warp 5    → Q rows 32–47
  pair=3:   warp 6   warp 7    → Q rows 48–63
```

**Pipeline execution flow:**

1. **Q load** — The producer warp (warp 0) issues a TMA load for the Q tile into `sQ`, then immediately begins the K/V loading pipeline. Consumers wait on `mbar_q` for Q data to arrive.

2. **K/V streaming (producer)** — The producer iterates over all K/V tile pairs. For each iteration `j`:
   - If `j >= 2`, wait on `mbar_empty` to ensure consumers have finished reading the buffer from 2 iterations ago (double buffering).
   - Issue TMA loads for K and V into the current stage (`sK0/sV0` or `sK1/sV1`), signaling `mbar_full` upon completion.
   - After all K/V tiles are dispatched, the producer warp exits.

3. **S = Q × Kᵀ (consumers)** — Consumers wait on `mbar_full` for the current K/V stage. Each consumer warp loads Q and K fragments from shared memory via `ldmatrix` (with swizzle-aware addressing), then performs `m16n8k16` HMMA to accumulate S tiles. The result is scaled by `1/√d`.

4. **Online softmax** — Each warp computes a local row-max across its S fragment using warp shuffle reductions (`__shfl_xor_sync`). Partial max values are exchanged between `warp_half` pairs through shared memory scratch space to obtain the true row-max `m_ij`. Then:
   - Compute `m_new = max(m_i, m_ij)` (running max across all K/V tiles so far)
   - Compute `α = exp(m_i − m_new)` (rescale factor for prior accumulator)
   - Compute `P = exp(S − m_new)` (softmax numerator)
   - Row-sum of P is similarly reduced across warp halves

5. **P writeback to SMEM** — P values are converted from fp32 to bf16 and written to `sP` (which aliases `sK_cur`) with manually applied swizzle addressing. This is necessary because P is not loaded via TMA — the swizzle layout must match the subsequent `ldmatrix` reads.

6. **O = P × V (consumers)** — P and V fragments are loaded via `ldmatrix` (x4 for P, x2.trans for V) with swizzle addressing, then accumulated through HMMA. The running output accumulator is rescaled: `acc = α · acc + P × V`.

7. **Output writeback** — After all K/V tiles are processed, each consumer thread writes its final output: `O = acc / l_i` (fp32 → bf16 conversion). Results are first staged in `sQ` (reused as output buffer), then cooperatively copied to global memory via 8-byte vectorized stores.

**Synchronization mechanisms:**

- `mbar_q` (arrive count = 1): signals Q tile arrival from TMA
- `mbar_full0/1` (arrive count = 1): signals K/V tile arrival in each double buffer stage
- `mbar_empty0/1` (arrive count = 256): signals consumers have finished reading a K/V stage, allowing the producer to reuse the buffer
- `bar.sync 1, 256`: intra-consumer-group barriers for shared memory data exchange (max/sum reduction, P writeback)

## Optimization History

Each version was entirely handwritten and profiled with `ncu` to guide the next optimization:

1. **Naive FA2** — Baseline tiled Flash Attention implementation
2. **SMEM vectorized loads** — Vectorized shared memory read/write
3. **MMA compute** — Replaced scalar arithmetic with Tensor Core MMA instructions
4. **TMA data movement** — Switched global→shared loads to asynchronous TMA
5. **Swizzle** — 128B swizzle address mapping to eliminate shared memory bank conflicts
6. **Double buffering** — Overlapped compute and data movement via pipeline
7. **Warp specialization** — Producer/consumer warp partitioning to further hide latency

## Build & Run

### Requirements

- NVIDIA GPU: Blackwell architecture (SM120), e.g. RTX 5060 Laptop
- CUDA Toolkit >= 12.8
- Compiler: nvcc

### Compile & Run

#run
make clean && make run

#look
./build/flash_attn profile/ref_data

#compare blackflash and cuDNN
python3 tests/bench_blackflash.py

