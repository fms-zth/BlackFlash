#include "flash_attn.cuh"
#include "utils.cuh"
#include <cuda.h>
#include <cuda/barrier> // mbarrier 相关
#include <cuda_bf16.h>

constexpr int Br = 64;          // query tile rows
constexpr int Bc = 64;          // key/value tile rows
constexpr int d_model = 64;     // head dimension
constexpr int BLOCK_SIZE = 288; // threads per block
// constexpr int num_consumer=8;

// swizzle 的索引写法：
__device__ __forceinline__ uint32_t swizzle_addr(const __nv_bfloat16 *base, int row, int col)
{
    int chunk = col / 8;
    int in_chunk = col % 8;
    int swizzle_chunk_col = chunk ^ (row % 8);
    return __cvta_generic_to_shared(base + row * 64 + swizzle_chunk_col * 8 + in_chunk);
}

// ============================================================
// Kernel
// ============================================================
__global__ void flash_attn_naive_kernel(const __grid_constant__ CUtensorMap tma_desc_q,
                                        const __grid_constant__ CUtensorMap tma_desc_k,
                                        const __grid_constant__ CUtensorMap tma_desc_v, __nv_bfloat16 *__restrict__ O,
                                        float *__restrict__ l_out, float *__restrict__ m_out, int N, float scale)
{
    // ======================================
    // Step 1: 索引计算
    // ======================================
    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    // const int stride=d_model;

    // 当前 query block 在 N 维的起始行
    const int q_start = q_block_idx * Br;

    // 提前退出：如果这个 block 完全超出 sequence length
    if (q_start >= N)
        return;

    // 基地址：Q/K/V/O 都是 [B, H, N, d] layout → 连续存储
    // offset = ((b * H + h) * N) * d
    const int bh_offset =
        (b * gridDim.y + h) *
        N; // 注意这个是行单位！少了d列的！后面会乘，这里的索引不要想象成三维坐标，这里是一维的，根据B,H,N,来思考索引地址
    // 还是那句话，griddim代表什么要看下面dim3代码传的什么  dim3 grid(num_q_blocks, H,
    // B);。所以这里的griddim.y代表的是H的数量
    __nv_bfloat16 *O_bh = O + bh_offset * d_model;
    // =====================================
    // Step 2: Shared Memory 分配和barrier的初始化
    // ======================================

    extern __shared__ char smem[];
    __nv_bfloat16 *sQ =
        reinterpret_cast<__nv_bfloat16 *>(smem); // 这里要把内存类型转换成bf16，前面就要加reinterpret_cast
    // 跳过sq的内存，所以用br
    __nv_bfloat16 *sK0 = sQ + Br * d_model;
    __nv_bfloat16 *sV0 = sK0 + Bc * d_model;
    __nv_bfloat16 *sK1 = sV0 + Bc * d_model;
    __nv_bfloat16 *sV1 = sK1 + Bc * d_model;

    uint64_t *mbar_q = reinterpret_cast<uint64_t *>(sV1 + Bc * d_model);
    uint64_t *mbar_full0 = mbar_q + 1;
    uint64_t *mbar_full1 = mbar_q + 2; // 这里q用了reinterpret，将每个元素的字节间隔变成了8字节了
    uint64_t *mbar_empty0 = mbar_q + 3;
    uint64_t *mbar_empty1 = mbar_q + 4;
    float *s_p_max = reinterpret_cast<float *>(mbar_q + 5); // 5个mbar之后
    float *s_p_sum = s_p_max + 2 * Br;
    uint32_t phase_full0 = 0, phase_full1 = 0;
    uint32_t phase_empty0 = 0, phase_empty1 = 0;
    if (tid == 0)
    { // 定义了的barrier，每个都要初始化
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_q)),
                     "r"(1) // expected arrive count = 1（只有1个线程发TMA）
        );
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_full0)),
                     "r"(1) // expected arrive count = 1（只有1个线程发TMA）
        );
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_full1)),
                     "r"(1) // expected arrive count = 1（只有1个线程发TMA）
        );
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_empty0)),
                     "r"(256) // 消费者要等待256个线程到齐
        );
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_empty1)),
                     "r"(256));
        // 初始化时设置一个 expected arrive count，
        //  之后每个参与的线程调用 mbarrier.arrive，内部计数器减一。
        //  当计数器归零时，barrier phase 翻转，所有等待的线程被放行。
    }
    __syncthreads();

    // ======================================
    // Step 3:加载Q，同时在warp specialization 下  生产者在stage2下加载KV
    // ======================================
    // 现在用的是TMA加载
    // 初始化mbarrier
    int warp_idx_all = tid / 32;
    int num_kv_blocks = (N + Bc - 1) / Bc;
    int g_row = bh_offset + q_start;
    if (warp_idx_all == 0 && tid % 32 == 0)
    {
        // 第一步先确定预期的字节数
        uint32_t expected_bytes = Br * d_model * sizeof(__nv_bfloat16);
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_q)),
            "r"(expected_bytes));
        // 第二步开始搬运
        //  发起 TMA load
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];\n" ::"r"((uint32_t)__cvta_generic_to_shared(sQ)), // smem 目标地址
                     "l"(&tma_desc_q),                                                               // TMA descriptor
                     "r"(0),                                         // coord[0] = col起始 = 0
                     "r"(g_row),                                     // coord[1] = row起始
                     "r"((uint32_t)__cvta_generic_to_shared(mbar_q)) // barrier
        );
    }

    if (warp_idx_all == 0)
    {

        // 只让lane0发射TMA
        if (tid % 32 == 0)
        {
            for (int j = 0; j < num_kv_blocks; j++)
            {
                int round = j % 2;
                uint64_t *mbar_full = (round == 0) ? mbar_full0 : mbar_full1;
                uint64_t *mbar_empty = (round == 0) ? mbar_empty0 : mbar_empty1;
                __nv_bfloat16 *sK_r = (round == 0) ? sK0 : sK1;
                __nv_bfloat16 *sV_r = (round == 0) ? sV0 : sV1;

                if (j >= 2)
                {
                    // 当j=2的时候，就要等待消费者算完才能继续搬运了
                    // 所以要先判断有没有算完
                    uint32_t phase_cur = (round == 0) ? phase_empty0 : phase_empty1;
                    // 当j刚到2的时候，phase_empty0=0，此时消费者开始计算，算完后会给phase加一，phase_empty0=1
                    asm volatile(
                        "{\n.reg .pred P;\n"
                        "WAIT_E_%=:\n"
                        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
                        "@!P bra WAIT_E_%=;\n}\n" // 你要明白的是mbar内部自动有一个phase进行计数，我们自己定义出来的
                        // phase就是为什么去和它内部的相比较，当内部的phase和我们定义的不同时，说明完成了。
                        ::"r"((uint32_t)__cvta_generic_to_shared(mbar_empty)),
                        "r"(phase_cur)); // 当phase_empty0和cur_phase不一样的时候，代码就不会被mbar阻碍，会继续走下去
                    if (round == 0)
                        phase_empty0++;
                    else
                        phase_empty1++; // 加一重新到达当前的状态
                }

                // 上面的if过去后，就是对应的j块已经空了，可以搬运了。
                uint32_t expected_bytes = 2 * Bc * d_model * sizeof(__nv_bfloat16);
                int g_kv_row = bh_offset + j * Bc;
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                                 (uint32_t)__cvta_generic_to_shared(mbar_full)),
                             "r"(expected_bytes));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];\n" ::"r"((uint32_t)__cvta_generic_to_shared(sK_r)), // smem 目标地址
                    "l"(&tma_desc_k),                                                                 // TMA descriptor
                    "r"(0),                                            // coord[0] = col起始 = 0
                    "r"(g_kv_row),                                     // coord[1] = row起始
                    "r"((uint32_t)__cvta_generic_to_shared(mbar_full)) // barrier
                );
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];\n" ::"r"((uint32_t)__cvta_generic_to_shared(sV_r)), // smem 目标地址
                    "l"(&tma_desc_v),                                                                 // TMA descriptor
                    "r"(0),                                            // coord[0] = col起始 = 0
                    "r"(g_kv_row),                                     // coord[1] = row起始
                    "r"((uint32_t)__cvta_generic_to_shared(mbar_full)) // barrier
                );
            }
        }
        return; // 当所有的kv搬运完后，生产者直接退出
    }

    // ======================================
    // Step 4: 初始化 Online Softmax 状态（寄存器）和 消费者
    // ======================================

    float m_i[2] = {-INFINITY, -INFINITY}; // running max
    float l_i[2] = {0.0f, 0.0f};           // running sum (denominator)

    //////////////////////////////////////////////
    // ldmatric和mma都是以warp为单位操作的，所以要先把warp单位构建出来
    /////////////////////////////////////////////

    int warp_idx = warp_idx_all - 1; // 对一个block里面128个线程进行分配
    int warp_pair = warp_idx / 2;    // 0..3，决定处理 Q 的哪 16 行
    int warp_half = warp_idx % 2;    // 0 或 1，决定 N 维的哪一半
    int lane_idx = tid % 32;
    //                    half:0  1
    // 8个消费者，分成       0  1    --0
    //                     2  3    --1
    //                     4  5    --2
    //                     6  7    --3    这样分配

    float acc[4][4] = {}; // 这个是o_i的累加器
    // 这种结果的累加器一定要定义在外面，因为for里面的话，外面就拿不到了。
    float RC[4][4] = {}; // 必须清零，HMMA用RC作为累加器C
    uint32_t RB[2];
    uint32_t RA[4];
    // ======================================
    // Step 5: 主循环 — 开始计算
    // ======================================

    // const int kv_global_row0=bh_offset+0; 到时候直接用bh_offset  能省下一个寄存器
    { // 开始之前要等待Q搬运完
        uint32_t ph = 0;
        asm volatile("{\n.reg .pred P;\n"
                     "WAIT_Q_%=:\n"
                     "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
                     "@!P bra WAIT_Q_%=;\n}\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_q)),
                     "r"(ph));
    }

    // 真正的主循环计算部分

    for (int j = 0; j < num_kv_blocks; j++)
    {
        int round = j % 2;
        uint64_t *mbar_full = (round == 0) ? mbar_full0 : mbar_full1;
        uint64_t *mbar_empty = (round == 0) ? mbar_empty0 : mbar_empty1;
        uint32_t phase_cur = (round == 0) ? phase_full0 : phase_full1;
        __nv_bfloat16 *sK_cur = (round == 0) ? sK0 : sK1;
        __nv_bfloat16 *sV_cur = (round == 0) ? sV0 : sV1;

        // 等待KV搬运完成
        asm volatile("{\n.reg .pred P;\n"
                     "WAIT_F_%=:\n"
                     "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
                     "@!P bra WAIT_F_%=;\n}\n" ::"r"((uint32_t)__cvta_generic_to_shared(mbar_full)),
                     "r"(phase_cur));
        if (round == 0)
            phase_full0++;
        else
            phase_full1++;

#pragma unroll
        for (int nc = 0; nc < 4; nc++)
        {
            RC[nc][0] = 0.0f;
            RC[nc][1] = 0.0f;
            RC[nc][2] = 0.0f;
            RC[nc][3] = 0.0f;
        }

        for (int nc = 0; nc < 4; nc++)
        {

            for (int kc = 0; kc < 4; kc++)
            {

                // 不要rowbase了，直接写进去，注意将warp分成两块后用warp_pair

                int q_frag_row = warp_pair * 16 + (lane_idx % 16);
                int q_frag_col = kc * 16 + (lane_idx / 16) * 8; // k 维度的偏移
                uint32_t q_add = swizzle_addr(sQ, q_frag_row, q_frag_col);

                LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], q_add);

                int k_frag_row = lane_idx % 8 + warp_half * 32 + nc * 8;
                int k_frag_col = kc * 16 + (lane_idx / 8) % 2 * 8;
                uint32_t k_add = swizzle_addr(sK_cur, k_frag_row, k_frag_col);

                LDMATRIX_X2(RB[0], RB[1], k_add);

                // === DEBUG: 打印 ldmatrix 加载的实际值 ===
                HMMA16816(RC[nc][0], RC[nc][1], RC[nc][2], RC[nc][3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1],
                          RC[nc][0], RC[nc][1], RC[nc][2], RC[nc][3]);
            }
        }

#pragma unroll
        for (int nc = 0; nc < 4; nc++)
        {
            RC[nc][0] *= scale;
            RC[nc][1] *= scale;
            RC[nc][2] *= scale;
            RC[nc][3] *= scale;
        } // 本来S就是要乘scale的，只是上面做QK的没乘；

        float p_max0 = -INFINITY, p_max1 = -INFINITY; // 上面说过了将warp分成了两个大部分
        for (int nc = 0; nc < 4; nc++)
        {
            p_max0 = fmaxf(p_max0, fmaxf(RC[nc][0], RC[nc][1]));
            p_max1 = fmaxf(p_max1, fmaxf(RC[nc][2], RC[nc][3]));
        }
        // shuffle reduce across lane_id % 4 的 4 个线程
        p_max0 = fmaxf(p_max0, __shfl_xor_sync(0xffffffff, p_max0, 1));
        p_max0 = fmaxf(p_max0, __shfl_xor_sync(0xffffffff, p_max0, 2));
        p_max1 = fmaxf(p_max1, __shfl_xor_sync(0xffffffff, p_max1, 1));
        p_max1 = fmaxf(p_max1, __shfl_xor_sync(0xffffffff, p_max1, 2));

        // Phase 3: smem exchange
        int smem_row0 = warp_pair * 16 + (lane_idx / 4) % 8;
        int smem_row1 = smem_row0 + 8;

        if (lane_idx % 4 == 0)
        {
            s_p_max[warp_half * Br + smem_row0] = p_max0;
            s_p_max[warp_half * Br + smem_row1] = p_max1;
        }
        asm volatile("bar.sync 1, 256;\n"); // 这里让八个warp全部将部分最大值填入SMEM
        // 然后才让取相反的部分的最大值other_pmax
        float other_p_max0 = s_p_max[(1 - warp_half) * Br + smem_row0];
        float other_p_max1 = s_p_max[(1 - warp_half) * Br + smem_row1];
        float m_ij[2] = {-INFINITY, -INFINITY};
        m_ij[0] = fmaxf(other_p_max0, p_max0);
        m_ij[1] = fmaxf(other_p_max1, p_max1);

        // 注意：这里 m_i 的索引含义变了。之前 m_i[0]/m_i[1] 对应 warp 的上8行/下8行，
        // 现在一样，但 warp_pair 决定具体是哪 16 行。
        float m_new[2] = {-INFINITY, -INFINITY};
        m_new[0] = fmaxf(m_i[0], m_ij[0]);
        m_new[1] = fmaxf(m_i[1], m_ij[1]);
        float alpha[2] = {-INFINITY, -INFINITY};
        alpha[0] = expf(m_i[0] - m_new[0]); // 注意这里是m_i-new
        alpha[1] = expf(m_i[1] - m_new[1]);

        float p[4][4], p_rowsum[2] = {0.0f, 0.0f};
        for (int nc = 0; nc < 4; nc++)
        {
            p[nc][0] = expf(RC[nc][0] - m_new[0]);
            p[nc][1] = expf(RC[nc][1] - m_new[0]);
            p[nc][2] = expf(RC[nc][2] - m_new[1]);
            p[nc][3] = expf(RC[nc][3] - m_new[1]);
            p_rowsum[0] += p[nc][0] + p[nc][1];
            p_rowsum[1] += p[nc][2] + p[nc][3];
        }
        p_rowsum[0] += __shfl_xor_sync(0xffffffff, p_rowsum[0], 1);
        p_rowsum[0] += __shfl_xor_sync(0xffffffff, p_rowsum[0], 2);
        p_rowsum[1] += __shfl_xor_sync(0xffffffff, p_rowsum[1], 1);
        p_rowsum[1] += __shfl_xor_sync(0xffffffff, p_rowsum[1], 2);

        if (lane_idx % 4 == 0)
        {
            s_p_sum[warp_half * Br + smem_row0] = p_rowsum[0];
            s_p_sum[warp_half * Br + smem_row1] = p_rowsum[1];
        }
        asm volatile("bar.sync 1, 256;\n");

        float other_p_rowsum[2] = {0.0f, 0.0f};
        other_p_rowsum[0] = s_p_sum[(1 - warp_half) * Br + smem_row0];
        other_p_rowsum[1] = s_p_sum[(1 - warp_half) * Br + smem_row1];
        float rowsum[2] = {0.0f, 0.0f};
        rowsum[0] = p_rowsum[0] + other_p_rowsum[0];
        rowsum[1] = p_rowsum[1] + other_p_rowsum[1];

        __nv_bfloat16 *sP = sK_cur;

        // int group_idx=lane_idx/4;
        // int tidin_group_idx=lane_idx%4;
        for (int nc = 0; nc < 4; nc++)
        {
            // int rowbase=warp_pair*16;
            int row = warp_pair * 16 + lane_idx / 4;
            int col = nc * 8 + lane_idx % 4 * 2 + warp_half * 32;
            int chunk = col / 8;
            int in_chunk = col % 8;
            int swizzle_chunk0 = chunk ^ (row % 8);
            int swizzle_chunk1 = chunk ^ ((row + 8) % 8);

            sP[row * 64 + swizzle_chunk0 * 8 + in_chunk] = __float2bfloat16(p[nc][0]);
            sP[row * 64 + swizzle_chunk0 * 8 + in_chunk + 1] = __float2bfloat16(p[nc][1]);
            sP[(row + 8) * 64 + swizzle_chunk1 * 8 + in_chunk] = __float2bfloat16(p[nc][2]);
            sP[(row + 8) * 64 + swizzle_chunk1 * 8 + in_chunk + 1] = __float2bfloat16(p[nc][3]);
        }
        asm volatile("bar.sync 1, 256;\n"); // 在生产者消费者中不能用syncthread

        float ROC[4][4] = {};
        // 搬运完了，重新计算O=P*V；
        //  for (int set=0;set<2;set++)//这里set要放外面,当然这个是4个warp的时候，当8个warp的时候就不用了
        { // 每一次kc后，移动后，都要进行一次mma计算留住，然后系统会自己把全部加起来，所以set放外面
            for (int nc = 0; nc < 4; nc++)
            {
                for (int kc = 0; kc < 4; kc++)
                {

                    int p_row = lane_idx % 16 + warp_pair * 16;
                    int p_col = (lane_idx / 16) * 8 + kc * 16;
                    uint32_t p_add = swizzle_addr(sP, p_row, p_col);
                    uint32_t RPA[4];
                    LDMATRIX_X4(RPA[0], RPA[1], RPA[2], RPA[3], p_add);

                    int v_row = kc * 16 + (lane_idx / 8) % 2 * 8 + lane_idx % 8;
                    int v_col = warp_half * 32 + nc * 8;
                    uint32_t v_add = swizzle_addr(sV_cur, v_row, v_col);
                    uint32_t RVB[2];
                    LDMATRIX_X2_TRANS(RVB[0], RVB[1], v_add);

                    HMMA16816(ROC[nc][0], ROC[nc][1], ROC[nc][2], ROC[nc][3], RPA[0], RPA[1], RPA[2], RPA[3], RVB[0],
                              RVB[1], ROC[nc][0], ROC[nc][1], ROC[nc][2], ROC[nc][3]);
                }
            }
        }

        for (int nc = 0; nc < 4; nc++)
        {
            acc[nc][0] = alpha[0] * acc[nc][0] + ROC[nc][0];
            acc[nc][1] = alpha[0] * acc[nc][1] + ROC[nc][1];
            acc[nc][2] = alpha[1] * acc[nc][2] + ROC[nc][2];
            acc[nc][3] = alpha[1] * acc[nc][3] + ROC[nc][3];
        }
        // 这样就将所有的O_i的数据存下来得到完整的o数据。
        // 接下来就是将寄存器acc中的数据写入global中
        float l_new[2];
        l_new[0] = alpha[0] * l_i[0] + rowsum[0];
        l_new[1] = alpha[1] * l_i[1] + rowsum[1];

        m_i[0] = m_new[0];
        m_i[1] = m_new[1];
        l_i[0] = l_new[0];
        l_i[1] = l_new[1];

        asm volatile("bar.sync 1, 256;\n");
        asm volatile(
            "mbarrier.arrive.shared.b64 _, [%0];\n" // 这个是arrive代码，线程到了以后减去一个计数，当计数为0的时候，翻转phase
            ::"r"((uint32_t)__cvta_generic_to_shared(mbar_empty)));
    }

    // ======================================
    // Step 6 & 7: 最终 rescale + 写回 global memory
    // ======================================
    // 主循环KV遍历完后就把算出来的o放进全局内存，放入的方法就是上面将P寄存器中的数据放入SMEM是一样的，只是地址改成全局内存

    __nv_bfloat16 *smem_O = sQ;

    for (int nc = 0; nc < 4; nc++)
    {
        int local_row0 = warp_pair * 16 + lane_idx / 4;     // 去掉 q_start
        int local_row1 = warp_pair * 16 + lane_idx / 4 + 8; // 去掉 q_start
        int col = (lane_idx % 4) * 2 + (warp_half * 4 + nc) * 8;

        smem_O[local_row0 * d_model + col] = __float2bfloat16(acc[nc][0] * (1.0f / l_i[0]));
        smem_O[local_row0 * d_model + col + 1] = __float2bfloat16(acc[nc][1] * (1.0f / l_i[0]));
        smem_O[local_row1 * d_model + col] = __float2bfloat16(acc[nc][2] * (1.0f / l_i[1]));
        smem_O[local_row1 * d_model + col + 1] = __float2bfloat16(acc[nc][3] * (1.0f / l_i[1]));
    }

    asm volatile("bar.sync 1, 256;\n");

    for (int i = tid - 32; i < (Br * d_model) / 4; i += 256)
    {
        // int elem_idx = i * 4;
        int row = i * 4 / d_model;
        int col = i * 4 % d_model;
        int global_row = q_start + row;
        if (global_row < N)
        {
            // 从 smem 读 8 bytes，向 global 写 8 bytes
            uint2 val = *reinterpret_cast<uint2 *>(&smem_O[i * 4]);
            *reinterpret_cast<uint2 *>(&O_bh[global_row * d_model + col]) = val;
        }
    }
}

// ============================================================
// Launch 函数
// ============================================================
void flash_attn_naive_launch(const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V, __nv_bfloat16 *O,
                             float *l, float *m, int B, int H, int N, int d, float scale, cudaStream_t stream)
{

    // Grid: 每个 block 处理一个 (query_tile, head, batch) 组合
    int num_q_blocks = (N + Br - 1) / Br;
    dim3 grid(num_q_blocks, H, B);

    dim3 block(BLOCK_SIZE);

    // Shared memory: sQ[Br*d] + sK[Bc*d] + sV[Bc*d]，全是 bf16
    size_t smem_size = (Br * d_model + Bc * d_model * 2 + 2 * Bc * d_model) * sizeof(__nv_bfloat16) +
                       5 * sizeof(uint64_t) + 4 * Br * sizeof(float);

    // TMA引入
    //-----------------------------------------------------------------------
    CUtensorMap tma_desc_q, tma_desc_k, tma_desc_v;
    cuuint64_t q_g_dim[2] = {(cuuint64_t)d, (cuuint64_t)B * H * N};
    cuuint64_t q_g_stride[1] = {(cuuint64_t)d * sizeof(__nv_bfloat16)};
    cuuint32_t q_b_dim[2] = {(cuuint32_t)d, (cuuint32_t)Br};
    cuuint32_t q_element_stride[2] = {
        1,
        1,
    };

    cuTensorMapEncodeTiled(&tma_desc_q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)Q, q_g_dim, q_g_stride, q_b_dim,
                           q_element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    cuuint32_t K_boxDim[2] = {(cuuint32_t)d, (cuuint32_t)Bc};
    cuTensorMapEncodeTiled(&tma_desc_k, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)K, q_g_dim, q_g_stride, K_boxDim,
                           q_element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    // V: 和 K 一模一样的 shape
    cuTensorMapEncodeTiled(&tma_desc_v, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)V, q_g_dim, q_g_stride, K_boxDim,
                           q_element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

    // 如果超过 48KB 默认限制，需要 opt-in
    if (smem_size > 48 * 1024)
    {
        CUDA_CHECK(
            cudaFuncSetAttribute(flash_attn_naive_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    flash_attn_naive_kernel<<<grid, block, smem_size, stream>>>(tma_desc_q, tma_desc_k, tma_desc_v, O, l, m, N, scale);
    CUDA_KERNEL_CHECK();
}
