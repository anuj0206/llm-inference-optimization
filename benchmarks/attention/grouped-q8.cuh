#pragma once

// Included by the pinned fattn.cu after its existing CUDA helpers.
namespace {

constexpr int q8_gqa_dim = 128;
constexpr int q8_gqa_heads = 4;
constexpr int q8_gqa_tile = 32;
constexpr int q8_gqa_split = 128;
constexpr int q8_gqa_row_words = q8_gqa_dim / QK8_0 * sizeof(block_q8_0) / sizeof(uint32_t);
static_assert(QK8_0 == 32 && sizeof(block_q8_0) == 34 && q8_gqa_row_words == 34);

bool q8_gqa_supported(const ggml_tensor * dst, int cc) {
    static const bool enabled = [] {
        const auto * value = std::getenv("LLM_OPT_ATTENTION_Q8");
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
        return tensor->nb[0] == sizeof(block_q8_0) &&
               (tensor->nb[1] | tensor->nb[2] | tensor->view_offs) % alignof(uint32_t) == 0;
    };
    return q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && q->ne[0] == q8_gqa_dim &&
           q->ne[1] == 1 && q->ne[2] == q8_gqa_heads * k->ne[2] && q->ne[3] == 1 &&
           k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0 &&
           k->ne[0] == q8_gqa_dim && v->ne[0] == q8_gqa_dim && k->ne[2] == v->ne[2] &&
           k->ne[3] == 1 && v->ne[3] == 1 && k->ne[1] == v->ne[1] &&
           k->ne[1] >= 4096 && k->ne[1] <= 65536 && k->ne[1] % FATTN_KQ_STRIDE == 0 &&
           aligned_rows(k) && aligned_rows(v) && ggml_is_contiguous(dst) &&
           mask && mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(half) &&
           mask->ne[0] == k->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
           ggml_get_op_params_f32(dst, 1) == 0.0f && ggml_get_op_params_f32(dst, 2) == 0.0f;
}

__device__ __forceinline__ uint32_t q8_gqa_word(const uint32_t * row, int block, int word) {
    const int offset = block * sizeof(block_q8_0) + sizeof(half) + word * sizeof(uint32_t);
    const uint32_t first = row[offset / 4];
    return offset % 4 == 0 ? first : __funnelshift_r(first, row[offset / 4 + 1], 16);
}

__device__ __forceinline__ float q8_gqa_scale(const uint32_t * row, int block) {
    const int offset = block * sizeof(block_q8_0);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(row[offset / 4] >> (8 * (offset % 4)))));
}

__global__ __launch_bounds__(128) void flash_attn_q8_grouped(
        const char * query, const char * key, const char * value, const half * mask,
        float * partial, float2 * stats, size_t q_head_stride,
        size_t k_row_stride, size_t k_head_stride, size_t v_row_stride, size_t v_head_stride,
        int depth, float scale) {
    const int lane = threadIdx.x;
    const int head_in_group = threadIdx.y;
    const int group = blockIdx.y;
    const int head = group * q8_gqa_heads + head_in_group;
    const int split = blockIdx.x;
    const int first = split * q8_gqa_split;
    key += group * k_head_stride;
    value += group * v_head_stride;

    __shared__ uint32_t keys[q8_gqa_tile][q8_gqa_row_words];
    __shared__ uint32_t values[q8_gqa_tile][q8_gqa_row_words];
    __shared__ int query_words[q8_gqa_heads][32];
    __shared__ float2 query_scales[q8_gqa_heads][4];
    __shared__ float scores[q8_gqa_heads][q8_gqa_tile];

    quantize_q8_1_to_shared<float2, 32>(
        reinterpret_cast<const float *>(query + head * q_head_stride), scale,
        query_words[head_in_group], query_scales[head_in_group]);
    __syncwarp();
    int q_words[4];
    float q_scales[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q_words[i] = query_words[head_in_group][8 * i + lane % 8];
        q_scales[i] = query_scales[head_in_group][i].x;
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float out[4] = {};
    for (int start = first; start < min(depth, first + q8_gqa_split); start += q8_gqa_tile) {
        // All four warps share the same packed tile. Each word is loaded once from global memory.
        for (int row = head_in_group; row < q8_gqa_tile; row += q8_gqa_heads) {
            const auto * k_row = reinterpret_cast<const uint32_t *>(key + (start + row) * k_row_stride);
            const auto * v_row = reinterpret_cast<const uint32_t *>(value + (start + row) * v_row_stride);
            keys[row][lane] = k_row[lane];
            values[row][lane] = v_row[lane];
            if (lane < q8_gqa_row_words - 32) {
                keys[row][lane + 32] = k_row[lane + 32];
                values[row][lane + 32] = v_row[lane + 32];
            }
        }
        __syncthreads();

        // Eight lanes produce a dot product, so each warp scores four positions at once.
#pragma unroll
        for (int base = 0; base < q8_gqa_tile; base += 4) {
            const int row = base + lane / 8;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto word = static_cast<int>(q8_gqa_word(keys[row], i, lane % 8));
                const float product_scale = q8_gqa_scale(keys[row], i) * q_scales[i];
                dot += float(ggml_cuda_dp4a(word, q_words[i], 0)) * product_scale;
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
        for (int row = 0; row < q8_gqa_tile; ++row) {
            const auto packed = q8_gqa_word(values[row], lane / 8, lane % 8);
            const float factor = __shfl_sync(0xffffffff, weight, row) * q8_gqa_scale(values[row], lane / 8);
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
    reinterpret_cast<float4 *>(partial + index * q8_gqa_dim)[lane] =
        make_float4(out[0], out[1], out[2], out[3]);
}

__global__ void flash_attn_q8_grouped_merge(
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
        sum += factors[i] * partial[(head * splits + i) * q8_gqa_dim + lane];
    }
    output[head * q8_gqa_dim + lane] = total > 0.0f ? sum / total : 0.0f;
}

void launch_q8_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    const int splits = (k->ne[1] + q8_gqa_split - 1) / q8_gqa_split;
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), q->ne[2] * splits * q8_gqa_dim);
    ggml_cuda_pool_alloc<float2> stats(ctx.pool(), q->ne[2] * splits);
    const auto stream = ctx.stream();
    flash_attn_q8_grouped<<<dim3(splits, k->ne[2]), dim3(32, q8_gqa_heads), 0, stream>>>(
        static_cast<const char *>(q->data), static_cast<const char *>(k->data), static_cast<const char *>(v->data),
        static_cast<const half *>(mask->data), partial.ptr, stats.ptr, q->nb[2],
        k->nb[1], k->nb[2], v->nb[1], v->nb[2], k->ne[1], ggml_get_op_params_f32(dst, 0));
    flash_attn_q8_grouped_merge<<<q->ne[2], q8_gqa_dim, splits * sizeof(float), stream>>>(
        partial.ptr, stats.ptr, static_cast<float *>(dst->data), splits);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace
