#include "dequantize.cuh"
#include "fp8.cuh"
#include "unary.cuh"

// Use the endpoints of each Q4_K subgroup to bound the reconstructed weights.
static __global__ void q4_k_fp8_bound(const block_q4_K * weights, int blocks, unsigned int * bound) {
    float maximum = 0;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < blocks; index += blockDim.x * gridDim.x) {
        const auto & block = weights[index];
        const float d = __low2float(block.dm);
        const float m = __high2float(block.dm);
        for (int subgroup = 0; subgroup < 8; ++subgroup) {
            uint8_t scale, offset;
            get_scale_min_k4(subgroup, block.scales, scale, offset);
            maximum = fmaxf(maximum, fmaxf(fabsf(m * offset), fabsf(15.0f * d * scale - m * offset)));
        }
    }
    __shared__ float reductions[32];
    maximum = block_reduce<block_reduce_method::MAX, 256>(maximum, reductions);
    if (threadIdx.x == 0) {
        atomicMax(bound, __float_as_uint(maximum));
    }
}

static __global__ void q4_k_pack_fp8(const block_q4_K * weights, uint8_t * packed,
        const unsigned int * bound, float * weight_scale) {
    const int block_index = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int group = lane / 8;
    const int column = 4 * (lane % 8);
    const auto & block = weights[block_index];
    const float maximum = __uint_as_float(*bound);
    const float scale = maximum > 0 ? maximum / 448.0f : 1.0f;
    if (block_index == 0 && lane == 0) {
        *weight_scale = scale;
    }
    uint8_t scale_low, min_low, scale_high, min_high;
    get_scale_min_k4(2 * group, block.scales, scale_low, min_low);
    get_scale_min_k4(2 * group + 1, block.scales, scale_high, min_high);
    const float d = __low2float(block.dm);
    const float m = __high2float(block.dm);
    uint8_t * output = packed + block_index * QK_K + 64 * group + column;
    for (int i = 0; i < 4; ++i) {
        const uint8_t code = block.qs[32 * group + column + i];
        output[i] = ggml_cuda_fp32_to_f8_e4m3((d * scale_low * (code & 15) - m * min_low) / scale);
        output[i + 32] = ggml_cuda_fp32_to_f8_e4m3((d * scale_high * (code >> 4) - m * min_high) / scale);
    }
}

static __global__ void prefill_fp8_swiglu(float * up, const float * gate, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        up[i] *= ggml_cuda_op_silu_single(gate[i]);
    }
}

static void prefill_fp8_gate_up(ggml_backend_cuda_context & ctx, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * input, ggml_tensor * output) {
    const int elements = ggml_nelements(up);
    const int blocks = elements / QK_K;
    const int output_elements = ggml_nelements(output);
    auto & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint8_t> packed(pool, elements);
    ggml_cuda_pool_alloc<unsigned int> bound(pool, 1);
    ggml_cuda_pool_alloc<float> scale(pool, 1);
    ggml_cuda_pool_alloc<float> gate_data(pool, output_elements);

    ggml_tensor weight = *up;
    weight.type = GGML_TYPE_F8_E4M3;
    weight.data = packed.ptr;
    weight.nb[0] = 1;
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        weight.nb[dim] = weight.nb[dim - 1] * weight.ne[dim - 1];
    }
    ggml_tensor weight_scale = {};
    weight_scale.type = GGML_TYPE_F32;
    weight_scale.data = scale.ptr;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        weight_scale.ne[dim] = 1;
        weight_scale.nb[dim] = sizeof(float);
    }
    ggml_tensor gate_result = *output;
    gate_result.data = gate_data.ptr;

    const ggml_tensor * weights[] = {up, gate};
    ggml_tensor * outputs[] = {output, &gate_result};
    for (int projection = 0; projection < 2; ++projection) {
        const auto * source = static_cast<const block_q4_K *>(weights[projection]->data);
        CUDA_CHECK(cudaMemsetAsync(bound.ptr, 0, sizeof(unsigned int), stream));
        q4_k_fp8_bound<<<256, 256, 0, stream>>>(source, blocks, bound.ptr);
        q4_k_pack_fp8<<<blocks / 4, 128, 0, stream>>>(source, packed.ptr, bound.ptr, scale.ptr);
        CUDA_CHECK(cudaGetLastError());
        if (!ggml_cuda_mul_mat_fp8(ctx, &weight, input, outputs[projection], &weight_scale)) {
            GGML_ABORT("No native FP8 cuBLASLt algorithm for the prefill experiment");
        }
    }
    prefill_fp8_swiglu<<<(output_elements + 255) / 256, 256, 0, stream>>>(
        static_cast<float *>(output->data), gate_data.ptr, output_elements);
    CUDA_CHECK(cudaGetLastError());
}
