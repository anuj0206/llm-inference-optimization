#include "ggml-backend-impl.h"

static ggml_backend_cuda_context & prefill_mmq_context(ggml_backend_t backend) {
    return *static_cast<ggml_backend_cuda_context *>(backend->context);
}

extern "C" __declspec(dllexport) cudaStream_t prefill_mmq_stream(ggml_backend_t backend) {
    return prefill_mmq_context(backend).stream();
}

extern "C" __declspec(dllexport) size_t prefill_mmq_packed_bytes() {
    return token_columns * input_features * sizeof(block_q8_1) / QK8_1 + 128 * sizeof(block_q8_1_mmq);
}

extern "C" __declspec(dllexport) void prefill_mmq_pack(ggml_backend_t backend, const float * input, void * packed) {
    constexpr int elements = input_features * token_columns;
    quantize_mmq_q8_1_cuda(input, nullptr, packed, GGML_TYPE_Q4_K,
        input_features, input_features, elements, elements,
        input_features, token_columns, 1, 1, prefill_mmq_stream(backend));
}

extern "C" __declspec(dllexport) void prefill_mmq_launch(ggml_backend_t backend, int width,
        const char * up, const char * gate, const void * packed, float * output) {
    GGML_ASSERT(width == 32 || width == 64 || width == 128);
    constexpr int weight_row = input_features / QK_K;
    constexpr int weight_stride = weight_row * output_features;
    constexpr int activation_stride = input_features * token_columns * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    constexpr int output_stride = output_features * token_columns;
    ggml_cuda_mm_fusion_args_device fusion = {};
    fusion.gate = gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    const mmq_args args = {
        up, GGML_TYPE_Q4_K, static_cast<const int *>(packed), nullptr, nullptr, output, nullptr,
        input_features, output_features, token_columns, weight_row, token_columns, output_features,
        1, 1, weight_stride, activation_stride, output_stride,
        1, 1, weight_stride, activation_stride, output_stride,
        token_columns / width, width, fusion,
    };
    auto & ctx = prefill_mmq_context(backend);
    ggml_cuda_mul_mat_q_switch_type(ctx, args, ctx.stream(), GGML_PREC_Q8);
}
