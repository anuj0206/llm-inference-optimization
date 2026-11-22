cmake_minimum_required(VERSION 3.25)
project(attention_bench LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)

set(cupti_samples "${CUDAToolkit_BIN_DIR}/../samples/common")
if(NOT TARGET CUDA::cupti OR NOT EXISTS "${cupti_samples}/cupti_range_profiler_util.h")
  message(FATAL_ERROR "The installed toolkit needs its cuda_cupti component.")
endif()

add_executable(attention-bench benchmark.cpp)
target_compile_features(attention-bench PRIVATE cxx_std_17)
target_compile_definitions(attention-bench PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
target_include_directories(attention-bench PRIVATE
  "${STOCK_SOURCE}/ggml/include" "${STOCK_SOURCE}/vendor" "${cupti_samples}")
target_link_libraries(attention-bench PRIVATE
  CUDA::cudart CUDA::cupti CUDA::cuda_driver
  "${STOCK_RUNTIME_BUILD}/ggml/src/ggml-base.lib"
  "${ATTENTION_BUILD}/ggml-cuda.lib")
set_target_properties(attention-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ATTENTION_BUILD}/bin")
