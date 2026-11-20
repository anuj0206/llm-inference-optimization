cmake_minimum_required(VERSION 3.25)
project(q4_microbench LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(NOT STOCK_SOURCE OR NOT STOCK_GENERATED OR NOT STOCK_RUNTIME_BUILD)
  message(FATAL_ERROR "Supply STOCK_SOURCE, STOCK_GENERATED and STOCK_RUNTIME_BUILD")
endif()
add_executable(q4-microbench benchmark.cu)
target_compile_features(q4-microbench PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(q4-microbench PRIVATE GGML_SHARED)
target_include_directories(q4-microbench PRIVATE
  "${STOCK_GENERATED}" "${STOCK_SOURCE}/ggml/src/ggml-cuda"
  "${STOCK_SOURCE}/ggml/src" "${STOCK_SOURCE}/ggml/include"
  "${STOCK_SOURCE}/vendor")
target_compile_options(q4-microbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=/openmp;-Xcompiler=/Zc:preprocessor;-Xcompiler=/Gy>)
target_link_options(q4-microbench PRIVATE /OPT:REF)
target_link_libraries(q4-microbench PRIVATE CUDA::cudart "${STOCK_RUNTIME_BUILD}/ggml/src/ggml-base.lib")
set_target_properties(q4-microbench PROPERTIES CUDA_ARCHITECTURES 89-real)
