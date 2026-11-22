#pragma once

// Included by the pinned fattn.cu after its existing CUDA helpers.
namespace {

constexpr int gqa_dim = 128;
constexpr int gqa_heads = 4;
constexpr int gqa_tile = 32;
constexpr int gqa_split = 128;
static_assert(QK4_0 == 32 && sizeof(block_q4_0) == 18);
static_assert(QK8_0 == 32 && sizeof(block_q8_0) == 34);
static_assert(FATTN_KQ_STRIDE % gqa_split == 0);

template<ggml_type Type, typename Block>
bool grouped_decode_supported(const ggml_tensor * dst, int cc) {
    static const bool enabled = [] {
        const auto * value = std::getenv(Type == GGML_TYPE_Q4_0 ? "LLM_OPT_ATTENTION_Q4" : "LLM_OPT_ATTENTION_Q8");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (!enabled || cc != GGML_CUDA_CC_ADA_LOVELACE || dst->src[4] || dst->src[5]) {
        return false;
    }

    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    const auto aligned_rows = [](const ggml_tensor * tensor) {
        return tensor->nb[0] == sizeof(Block) &&
               (tensor->nb[1] | tensor->nb[2] | tensor->view_offs) % alignof(uint32_t) == 0;
    };
    return q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && q->ne[0] == gqa_dim &&
           q->ne[1] == 1 && q->ne[2] == gqa_heads * k->ne[2] && q->ne[3] == 1 &&
           k->type == Type && v->type == Type &&
           k->ne[0] == gqa_dim && v->ne[0] == gqa_dim && k->ne[2] == v->ne[2] &&
           k->ne[3] == 1 && v->ne[3] == 1 && k->ne[1] == v->ne[1] &&
           k->ne[1] >= 4096 && k->ne[1] <= 65536 && k->ne[1] % FATTN_KQ_STRIDE == 0 &&
           aligned_rows(k) && aligned_rows(v) && ggml_is_contiguous(dst) &&
           mask && mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(half) &&
           mask->ne[0] == k->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
           ggml_get_op_params_f32(dst, 1) == 0.0f && ggml_get_op_params_f32(dst, 2) == 0.0f;
}

template<typename Block>
__device__ __forceinline__ uint32_t gqa_word(const uint32_t * row, int block, int word) {
    const int offset = block * sizeof(Block) + sizeof(half) + word * sizeof(uint32_t);
    const uint32_t first = row[offset / 4];
    return offset % 4 == 0 ? first : __funnelshift_r(first, row[offset / 4 + 1], 16);
}

template<typename Block>
__device__ __forceinline__ float gqa_scale(const uint32_t * row, int block) {
    const int offset = block * sizeof(Block);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(row[offset / 4] >> (8 * (offset % 4)))));
}

template<typename Block>
__device__ __forceinline__ uint32_t gqa_quants(const uint32_t * row, int block, int group_of_four) {
    if constexpr (std::is_same_v<Block, block_q4_0>) {
        // Each byte holds values 16 positions apart. Select the appropriate half of the block.
        const auto packed = gqa_word<Block>(row, block, group_of_four % 4);
        return (packed >> (4 * (group_of_four / 4))) & 0x0f0f0f0fu;
    } else {
        return gqa_word<Block>(row, block, group_of_four);
    }
}

template<typename Block>
__global__ __launch_bounds__(128) void flash_attn_grouped_decode(
        const char * query, const char * key, const char * value, const half * mask,
        float * partial, float2 * stats, size_t q_head_stride,
        size_t k_row_stride, size_t k_head_stride, size_t v_row_stride, size_t v_head_stride,
        int depth, float scale) {
    const int lane = threadIdx.x;
    const int head_in_group = threadIdx.y;
    const int group = blockIdx.y;
    const int head = group * gqa_heads + head_in_group;
    const int split = blockIdx.x;
    const int first = split * gqa_split;
    key += group * k_head_stride;
    value += group * v_head_stride;

    constexpr bool q4 = std::is_same_v<Block, block_q4_0>;
    constexpr int row_words = gqa_dim / 32 * sizeof(Block) / sizeof(uint32_t);
    __shared__ uint32_t keys[gqa_tile][row_words];
    __shared__ uint32_t values[gqa_tile][row_words];
    __shared__ int query_words[gqa_heads][32];
    __shared__ float2 query_scales[gqa_heads][4];
    __shared__ float scores[gqa_heads][gqa_tile];

    quantize_q8_1_to_shared<float2, 32>(
        reinterpret_cast<const float *>(query + head * q_head_stride), scale,
        query_words[head_in_group], query_scales[head_in_group]);
    __syncwarp();
    int q_words[4];
    float2 q_scale_sum[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_words[i] = query_words[head_in_group][8 * i + lane % 8];
        q_scale_sum[i] = query_scales[head_in_group][i];
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float out[4] = {};
    for (int start = first; start < min(depth, first + gqa_split); start += gqa_tile) {
        // All four warps share the same packed tile. Each word is loaded once from global memory.
        for (int row = head_in_group; row < gqa_tile; row += gqa_heads) {
            const auto * k_row = reinterpret_cast<const uint32_t *>(key + (start + row) * k_row_stride);
            const auto * v_row = reinterpret_cast<const uint32_t *>(value + (start + row) * v_row_stride);
#pragma unroll
            for (int offset = 0; offset < row_words; offset += 32) {
                const int word = offset + lane;
                if (word < row_words) {
                    keys[row][word] = k_row[word];
                    values[row][word] = v_row[word];
                }
            }
        }
        __syncthreads();

        // Eight lanes produce a dot product, so each warp scores four positions at once.
#pragma unroll
        for (int base = 0; base < gqa_tile; base += 4) {
            const int row = base + lane / 8;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto word = static_cast<int>(gqa_quants<Block>(keys[row], i, lane % 8));
                const float key_scale = gqa_scale<Block>(keys[row], i);
                const float product = float(ggml_cuda_dp4a(word, q_words[i], 0));
                if constexpr (q4) {
                    // Stock Q4_0 subtracts the offset of eight using the original scaled query sum.
                    dot += key_scale * (product * q_scale_sum[i].x - q_scale_sum[i].y);
                } else {
                    dot += product * (key_scale * q_scale_sum[i].x);
                }
            }
            dot += __shfl_xor_sync(0xffffffff, dot, 4, 8);
            dot += __shfl_xor_sync(0xffffffff, dot, 2, 8);
            dot += __shfl_xor_sync(0xffffffff, dot, 1, 8);
            if (lane % 8 == 0) {
                scores[head_in_group][row] = dot + __half2float(mask[start + row]);
            }
        }
        __syncwarp();

        const float score = scores[head_in_group][lane];
        float next_max = fmaxf(score, running_max);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, offset));
        }
        const float old_weight = expf(running_max - next_max);
        const float weight = expf(score - next_max);
        float sum = weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        running_sum = running_sum * old_weight + sum;
        running_max = next_max;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] *= old_weight;
        }

#pragma unroll 4
        for (int row = 0; row < gqa_tile; ++row) {
            auto packed = gqa_quants<Block>(values[row], lane / 8, lane % 8);
            if constexpr (q4) {
                packed = __vsub4(packed, 0x08080808u);
            }
            const float factor = __shfl_sync(0xffffffff, weight, row) * gqa_scale<Block>(values[row], lane / 8);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] += factor * float(static_cast<int8_t>(packed >> (8 * i)));
            }
        }
        __syncthreads();
    }

    const int index = head * gridDim.x + split;
    if (lane == 0) {
        stats[index] = make_float2(running_max, running_sum);
    }
    reinterpret_cast<float4 *>(partial + index * gqa_dim)[lane] =
        make_float4(out[0], out[1], out[2], out[3]);
}

__global__ void flash_attn_grouped_decode_merge(
        const float * partial, const float2 * stats, float * output, int splits) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const auto * head_stats = stats + head * splits;
    extern __shared__ float factors[];
    __shared__ float total;
    if (lane < 32) {
        float largest = -FLT_MAX;
        for (int i = lane; i < splits; i += 32) {
            largest = fmaxf(largest, head_stats[i].x);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, offset));
        }
        float sum = 0.0f;
        for (int i = lane; i < splits; i += 32) {
            const float factor = expf(head_stats[i].x - largest);
            factors[i] = factor;
            sum += factor * head_stats[i].y;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            total = sum;
        }
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < splits; ++i) {
        sum += factors[i] * partial[(head * splits + i) * gqa_dim + lane];
    }
    output[head * gqa_dim + lane] = total > 0.0f ? sum / total : 0.0f;
}

template<typename Block>
void launch_grouped_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    const int splits = k->ne[1] / gqa_split;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), q->ne[2] * splits * gqa_dim);
    ggml_cuda_pool_alloc<float2> stats(ctx.pool(), q->ne[2] * splits);
    const auto stream = ctx.stream();
    flash_attn_grouped_decode<Block><<<dim3(splits, k->ne[2]), dim3(32, gqa_heads), 0, stream>>>(
        static_cast<const char *>(q->data), static_cast<const char *>(k->data), static_cast<const char *>(v->data),
        static_cast<const half *>(mask->data), partial.ptr, stats.ptr, q->nb[2],
        k->nb[1], k->nb[2], v->nb[1], v->nb[2], k->ne[1], ggml_get_op_params_f32(dst, 0));
    flash_attn_grouped_decode_merge<<<q->ne[2], gqa_dim, splits * sizeof(float), stream>>>(
        partial.ptr, stats.ptr, static_cast<float *>(dst->data), splits);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace
