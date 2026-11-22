cmake_minimum_required(VERSION 3.25)
project(prefill_mmq_bench LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(prefill-mmq-bench benchmark.cpp)
target_compile_features(prefill-mmq-bench PRIVATE cxx_std_17)
target_compile_definitions(prefill-mmq-bench PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
target_include_directories(prefill-mmq-bench PRIVATE "${STOCK_SOURCE}/ggml/include" "${STOCK_SOURCE}/vendor")
target_link_libraries(prefill-mmq-bench PRIVATE CUDA::cudart
  "${STOCK_RUNTIME_BUILD}/ggml/src/ggml-base.lib" "${EXPERIMENT_BUILD}/ggml-cuda.lib")
set_target_properties(prefill-mmq-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${EXPERIMENT_BUILD}/bin")
