// Stock CUDA excerpts are generated from the pinned MIT-licensed llama.cpp source.
#include "stock_mmvq.cuh"
#include "two_warp_mmvq.cuh"
#include "one_warp_mmvq.cuh"
#include "two_row_mmvq.cuh"
#include "stock_quantize.cuh"
#include <nlohmann/json.hpp>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

using json = nlohmann::ordered_json;

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct Device {
    T *p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&p, count*sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device &) = delete;
};

struct Weights {
    int layer, n, k;
    std::vector<block_q4_K> up, gate;
    Device<block_q4_K> gpu_up, gpu_gate;
    Weights(int layer_, int n_, int k_): layer(layer_), n(n_), k(k_),
        up(size_t(n)*k/256), gate(up.size()), gpu_up(up.size()), gpu_gate(gate.size()) {}
    void upload() {
        check(cudaMemcpy(gpu_up.p, up.data(), up.size()*sizeof(block_q4_K), cudaMemcpyHostToDevice));
        check(cudaMemcpy(gpu_gate.p, gate.data(), gate.size()*sizeof(block_q4_K), cudaMemcpyHostToDevice));
    }
};

static void quantize(const float *x, block_q8_1 *q, int k, cudaStream_t stream) {
    quantize_q8_1<<<dim3((k+255)/256,1,1),256,0,stream>>>(x,q,k,k,k,k,k,1,init_fastdiv_values(1));
}

template<bool Fused, int Warps, int Rows = 1> static constexpr auto matvec_kernel() {
    static_assert(Warps == 4 || (Fused && (Warps == 2 || Warps == 1)));
    static_assert(Rows == 1 || (Fused && Warps == 4 && Rows == 2));
    if constexpr (Rows == 2) return mul_mat_vec_q_two_rows<GGML_TYPE_Q4_K,1,Fused>;
    if constexpr (Warps == 1) return mul_mat_vec_q_one_warp<GGML_TYPE_Q4_K,1,Fused>;
    if constexpr (Warps == 2) return mul_mat_vec_q_two_warps<GGML_TYPE_Q4_K,1,Fused>;
    return mul_mat_vec_q<GGML_TYPE_Q4_K,1,Fused>;
}

template<bool Fused, int Warps = 4, int Rows = 1> static void matvec(const block_q4_K *w, const block_q4_K *gate,
        const block_q8_1 *x, float *y, int n, int k, cudaStream_t stream) {
    ggml_cuda_mm_fusion_args_device fusion{};
    if constexpr (Fused) { fusion.gate = gate; fusion.glu_op = GGML_GLU_OP_SWIGLU; }
    if constexpr (Rows == 2) {
        if (n % Rows != 0) throw std::invalid_argument("The two-row kernel requires an even output width");
    }
    constexpr auto kernel = matvec_kernel<Fused, Warps, Rows>();
    kernel<<<dim3(n/Rows,1,1),dim3(32,Warps,1),0,stream>>>(
        w,x,nullptr,fusion,y,k,init_fastdiv_values(1),k/256,k/32,n,init_fastdiv_values(1),
        n*(k/256),k/32,n,init_fastdiv_values(1),n*(k/256),k/32,n,0);
}

template<int Warps, int Rows = 1> static json launch_resources(int max_threads_per_sm) {
    constexpr auto kernel = matvec_kernel<true, Warps, Rows>();
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, kernel));
    int blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, Warps * 32, 0));
    return {{"warps_per_block", Warps}, {"rows_per_block", Rows}, {"registers_per_thread", attributes.numRegs},
        {"local_bytes_per_thread", attributes.localSizeBytes}, {"shared_bytes_per_block", attributes.sharedSizeBytes},
        {"max_resident_blocks_per_sm", blocks_per_sm}, {"max_resident_warps_per_sm", blocks_per_sm * Warps},
        {"theoretical_occupancy", double(blocks_per_sm * Warps * 32) / max_threads_per_sm}};
}

static __global__ void combine(const float *up, const float *gate, float *out, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i] = up[i]*ggml_cuda_op_silu_single(gate[i]);
}

static void unfused(Weights &w, block_q8_1 *x, float *up, float *gate, float *out, cudaStream_t stream) {
    matvec<false>(w.gpu_up.p,nullptr,x,up,w.n,w.k,stream);
    matvec<false>(w.gpu_gate.p,nullptr,x,gate,w.n,w.k,stream);
    combine<<<(w.n+255)/256,256,0,stream>>>(up,gate,out,w.n);
}

static double silu(double x) {
    return x >= 0 ? x/(1+std::exp(-x)) : x*std::exp(x)/(1+std::exp(x));
}

// Scalar layout decoding, independent of the CUDA packed-dot implementation.
static float weight_at(const block_q4_K &b, int index) {
    int group=index/32, pos=index%32;
    int scale, minimum;
    if (group<4) { scale=b.scales[group]&63; minimum=b.scales[group+4]&63; }
    else {
        scale=(b.scales[group+4]&15)|((b.scales[group-4]>>6)<<4);
        minimum=(b.scales[group+4]>>4)|((b.scales[group]>>6)<<4);
    }
    int packed=b.qs[(group/2)*32+pos];
    int q=group%2 ? packed>>4 : packed&15;
    float2 factors=__half22float2(b.dm);
    return (factors.x*scale)*q-factors.y*minimum;
}

struct Reference { std::vector<double> up,gate,out,l1_up,l1_gate; };

static Reference reference(const Weights &w, const std::vector<block_q8_1> &x) {
    Reference r;
    for (auto *v: {&r.up,&r.gate,&r.out,&r.l1_up,&r.l1_gate}) v->resize(w.n);
    #pragma omp parallel for
    for (int row=0;row<w.n;row++) {
        double up=0,gate=0,lu=0,lg=0;
        for (int col=0;col<w.k;col++) {
            double xv=double(__low2float(x[col/32].ds))*x[col/32].qs[col%32];
            size_t index=size_t(row)*(w.k/256)+col/256;
            double u=double(weight_at(w.up[index],col%256))*xv;
            double g=double(weight_at(w.gate[index],col%256))*xv;
            up+=u; gate+=g; lu+=std::abs(u); lg+=std::abs(g);
        }
        r.up[row]=up;r.gate[row]=gate;r.out[row]=up*silu(gate);r.l1_up[row]=lu;r.l1_gate[row]=lg;
    }
    return r;
}

struct Errors {
    double max_abs=0, sum_sq=0, ref_sq=0, max_ratio=0;
    int failures=0, count=0;
    void add(double actual,double expected,double tolerance) {
        count++;
        if (!std::isfinite(actual)||!std::isfinite(expected)) { failures++;max_ratio=1e300;return; }
        double error=std::abs(actual-expected);
        max_abs=std::max(max_abs,error);sum_sq+=error*error;ref_sq+=expected*expected;
        max_ratio=std::max(max_ratio,error/tolerance);
        if(error>tolerance)failures++;
    }
    json result() const { return {{"count",count},{"failures",failures},{"max_abs_error",max_abs},
        {"rmse",std::sqrt(sum_sq/std::max(1,count))},{"relative_l2_error",std::sqrt(sum_sq/std::max(1e-30,ref_sq))},
        {"max_error_over_tolerance",max_ratio}}; }
};

static std::vector<float> activation(int k,int scenario) {
    std::mt19937 rng(4200+scenario);
    std::uniform_real_distribution<float> random(-0.2f,0.2f);
    std::vector<float> x(k);
    for(int i=0;i<k;i++) {
        switch(scenario) {
            case 0: x[i]=random(rng);break;
            case 1: x[i]=0;break;
            case 2: x[i]=(i%2 ? -1.0f:1.0f)*((i%31)+1)/32;break;
            case 3: x[i]=i==k/2 ? 1.0f:0;break;
            case 4: x[i]=random(rng)*1e-3f;break;
            default:x[i]=random(rng)*100;break;
        }
    }
    return x;
}

static json correctness(Weights &w,int scenario,cudaStream_t stream) {
    Device<float> dx(w.k),out(w.n),candidate(w.n),single(w.n),up(w.n),gate(w.n),control(w.n);
    Device<float> paired_rows(w.n);
    Device<block_q8_1> q(w.k/32);
    auto x=activation(w.k,scenario);
    check(cudaMemcpyAsync(dx.p,x.data(),w.k*sizeof(float),cudaMemcpyHostToDevice,stream));
    quantize(dx.p,q.p,w.k,stream);
    matvec<true>(w.gpu_up.p,w.gpu_gate.p,q.p,out.p,w.n,w.k,stream);
    matvec<true,2>(w.gpu_up.p,w.gpu_gate.p,q.p,candidate.p,w.n,w.k,stream);
    matvec<true,1>(w.gpu_up.p,w.gpu_gate.p,q.p,single.p,w.n,w.k,stream);
    matvec<true,4,2>(w.gpu_up.p,w.gpu_gate.p,q.p,paired_rows.p,w.n,w.k,stream);
    unfused(w,q.p,up.p,gate.p,control.p,stream);
    check(cudaGetLastError());check(cudaStreamSynchronize(stream));
    std::vector<block_q8_1> qhost(w.k/32);
    std::vector<float> actual(w.n),two_warp(w.n),one_warp(w.n),hu(w.n),hg(w.n),hc(w.n);
    std::vector<float> two_row(w.n);
    check(cudaMemcpy(qhost.data(),q.p,qhost.size()*sizeof(block_q8_1),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(two_warp.data(),candidate.p,w.n*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(one_warp.data(),single.p,w.n*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(two_row.data(),paired_rows.p,w.n*sizeof(float),cudaMemcpyDeviceToHost));
    for(auto item:{std::make_pair(actual.data(),out.p),std::make_pair(hu.data(),up.p),std::make_pair(hg.data(),gate.p),std::make_pair(hc.data(),control.p)})
        check(cudaMemcpy(item.first,item.second,w.n*sizeof(float),cudaMemcpyDeviceToHost));
    auto expected=reference(w,qhost);
    int quant_failures=0;
    double activation_error_sq=0,activation_ref_sq=0,max_activation_error=0;
    for(int block=0;block<w.k/32;block++) {
        float maximum=0;for(int j=0;j<32;j++)maximum=std::max(maximum,std::abs(x[block*32+j]));
        float scale=maximum/127.0f;
        float actual_scale=__low2float(qhost[block].ds);
        float expected_scale=__half2float(__float2half(scale));
        if(std::abs(actual_scale-expected_scale)>6e-8+0.001*std::abs(expected_scale))quant_failures++;
        int sum=0;
        for(int j=0;j<32;j++) {
            float ratio=maximum==0 ? 0:x[block*32+j]/scale;
            int expected_q=maximum==0 ? 0:int(std::round(ratio));
            int actual_q=qhost[block].qs[j];sum+=actual_q;
            float magnitude=std::abs(ratio);
            bool near_tie=std::abs(magnitude-std::floor(magnitude)-0.5f)<1e-3f;
            if(actual_q!=expected_q && !(near_tie && std::abs(actual_q-expected_q)==1))quant_failures++;
            double decoded=double(actual_scale)*actual_q,original=x[block*32+j],error=decoded-original;
            activation_error_sq+=error*error;activation_ref_sq+=original*original;
            max_activation_error=std::max(max_activation_error,std::abs(error));
        }
        float expected_sum=__half2float(__float2half(scale*sum));
        if(std::abs(__high2float(qhost[block].ds)-expected_sum)>6e-8+0.001*std::abs(expected_sum))quant_failures++;
    }
    Errors eu,eg,ef,ec,candidate_cpu,candidate_stock,candidate_stock_tight;
    Errors single_cpu, single_stock, single_stock_tight;
    Errors two_row_cpu, two_row_stock, two_row_stock_tight;
    for(int i=0;i<w.n;i++) {
        double tu=2e-5*expected.l1_up[i]+2e-5;
        double tg=2e-5*expected.l1_gate[i]+2e-5;
        double tf=std::abs(silu(expected.gate[i]))*tu+1.1*(std::abs(expected.up[i])+tu)*tg+5e-5*(1+std::abs(expected.out[i]));
        eu.add(hu[i],expected.up[i],tu);eg.add(hg[i],expected.gate[i],tg);ef.add(actual[i],expected.out[i],tf);
        ec.add(actual[i],hc[i],5e-5*(1+std::abs(hc[i])));
        candidate_cpu.add(two_warp[i],expected.out[i],tf);
        // Both results can differ from the CPU reference by up to tf.
        candidate_stock.add(two_warp[i],actual[i],2*tf);
        candidate_stock_tight.add(two_warp[i],actual[i],5e-5*(1+std::abs(actual[i])));
        single_cpu.add(one_warp[i],expected.out[i],tf);
        single_stock.add(one_warp[i],actual[i],2*tf);
        single_stock_tight.add(one_warp[i],actual[i],5e-5*(1+std::abs(actual[i])));
        two_row_cpu.add(two_row[i], expected.out[i], tf);
        two_row_stock.add(two_row[i], actual[i], 2*tf);
        two_row_stock_tight.add(two_row[i], actual[i], 5e-5*(1+std::abs(actual[i])));
    }
    Errors negative;negative.add(std::numeric_limits<double>::quiet_NaN(),expected.out[0],1);
    negative.add(expected.out[0]+10,expected.out[0],1);
    bool passed=eu.failures+eg.failures+ef.failures+ec.failures+quant_failures==0 && negative.failures==2
        && candidate_cpu.failures == 0 && candidate_stock.failures == 0
        && single_cpu.failures == 0 && single_stock.failures == 0
        && two_row_cpu.failures == 0 && two_row_stock.failures == 0;
    std::cout<<"check layer="<<w.layer<<" n="<<w.n<<" k="<<w.k<<" scenario="<<scenario<<" "<<(passed?"PASS":"FAIL")<<std::endl;
    return {{"layer",w.layer},{"n",w.n},{"k",w.k},{"activation_scenario",scenario},{"passed",passed},
        {"up_dot",eu.result()},{"gate_dot",eg.result()},{"fused_vs_cpu",ef.result()},{"fused_vs_unfused",ec.result()},
        {"two_warp_vs_cpu",candidate_cpu.result()},{"two_warp_vs_stock",candidate_stock.result()},
        {"two_warp_vs_stock_tight_diagnostic",candidate_stock_tight.result()},
        {"one_warp_vs_cpu",single_cpu.result()},{"one_warp_vs_stock",single_stock.result()},
        {"one_warp_vs_stock_tight_diagnostic",single_stock_tight.result()},
        {"two_row_vs_cpu",two_row_cpu.result()},{"two_row_vs_stock",two_row_stock.result()},
        {"two_row_vs_stock_tight_diagnostic",two_row_stock_tight.result()},
        {"activation_quantization",{{"failures",quant_failures},{"max_abs_reconstruction_error",max_activation_error},
            {"relative_l2_reconstruction_error",std::sqrt(activation_error_sq/std::max(1e-30,activation_ref_sq))}}},
        {"negative_controls_detected",negative.failures}};
}

static json timing(std::vector<std::unique_ptr<Weights>> &weights,bool rotate,bool include_quant,bool split,cudaStream_t stream) {
    int n=weights[0]->n,k=weights[0]->k;
    Device<float> x(k),out(n),up(n),gate(n);Device<block_q8_1> q(k/32);
    auto hx=activation(k,0);check(cudaMemcpyAsync(x.p,hx.data(),k*sizeof(float),cudaMemcpyHostToDevice,stream));
    quantize(x.p,q.p,k,stream);
    auto launch_batch = [&](int variant, int count, int start_layer) {
        for (int i = 0; i < count; ++i) {
            auto &w = *weights[rotate ? (start_layer + i) % weights.size() : 0];
            if (include_quant) quantize(x.p,q.p,k,stream);
            if (split) unfused(w,q.p,up.p,gate.p,out.p,stream);
            else if (variant == 1) matvec<true,2>(w.gpu_up.p,w.gpu_gate.p,q.p,out.p,n,k,stream);
            else if (variant == 2) matvec<true,1>(w.gpu_up.p,w.gpu_gate.p,q.p,out.p,n,k,stream);
            else if (variant == 3) matvec<true,4,2>(w.gpu_up.p,w.gpu_gate.p,q.p,out.p,n,k,stream);
            else matvec<true>(w.gpu_up.p,w.gpu_gate.p,q.p,out.p,n,k,stream);
        }
    };
    constexpr int warmup_calls = 200;
    constexpr int calls_per_sample = 108;
    constexpr int sample_count = 24;
    const int variants = split ? 1 : 4;
    for (int variant = 0; variant < variants; ++variant) launch_batch(variant, warmup_calls, 0);
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    // Visit every ordering of the four fused variants once.
    std::array<int, 4> order = {0, 1, 2, 3};
    std::array<std::vector<double>, 4> samples;
    for (int trial = 0; trial < sample_count; ++trial) {
        for (int arm = 0; arm < variants; ++arm) {
            const int variant = order[arm];
            check(cudaEventRecord(begin,stream));
            launch_batch(variant, calls_per_sample, trial);
            check(cudaEventRecord(end,stream));
            check(cudaEventSynchronize(end));
            float ms;
            check(cudaEventElapsedTime(&ms,begin,end));
            samples[variant].push_back(ms * 1000.0 / calls_per_sample);
        }
        if (!split) std::next_permutation(order.begin(), order.end());
    }
    check(cudaGetLastError());check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    json rows = json::array();
    constexpr const char *fused_labels[] = {"stock_fused", "two_warp_fused", "one_warp_fused", "two_row_fused"};
    for (int arm = 0; arm < variants; ++arm) {
        const auto &values = samples[arm];
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0;
        for (double value : values) variance += (value-mean)*(value-mean);
        const char *implementation = split ? "unfused_control" : fused_labels[arm];
        std::cout << "timing " << (rotate ? "36-layer rotation" : "one-pair reuse") << " "
                  << (include_quant ? "quant+operation" : "prepared activation") << " " << implementation
                  << " " << mean << " us" << std::endl;
        rows.push_back({{"mode",rotate?"36_layer_rotation":"one_pair_reuse"},
            {"path",include_quant?"quantize_plus_operation":"prepared_activation"},
            {"implementation",implementation},{"mean_us",mean},
            {"median_us",(sorted[sample_count/2-1]+sorted[sample_count/2])/2},
            {"p95_batch_mean_us",sorted[(95*sample_count+99)/100-1]},
            {"stddev_batch_mean_us",std::sqrt(variance/(sample_count-1))},
            {"warmup_calls",warmup_calls},{"calls_per_sample",calls_per_sample},{"samples_us",values}});
    }
    return rows;
}

int main(int argc,char **argv) {
    json result;
    try {
        if(argc!=3)throw std::runtime_error("usage: q4-microbench fixture.json output.json");
        std::ifstream description(argv[1]);json fixture;description>>fixture;
        result["fixture"]=fixture;result["correctness"]=json::array();result["timings"]=json::array();
        omp_set_num_threads(8);
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
        result["device"]={{"name",prop.name},{"l2_bytes",prop.l2CacheSize},{"compute_major",prop.major},{"compute_minor",prop.minor}};
        result["device"]["streaming_multiprocessors"] = prop.multiProcessorCount;
        result["device"]["max_threads_per_sm"] = prop.maxThreadsPerMultiProcessor;
        result["device"]["max_blocks_per_sm"] = prop.maxBlocksPerMultiProcessor;
        result["device"]["total_vram_bytes"] = prop.totalGlobalMem;
        result["device"]["memory_bus_bits"] = prop.memoryBusWidth;
        result["launch_resources"] = json::array({launch_resources<4>(prop.maxThreadsPerMultiProcessor),
            launch_resources<2>(prop.maxThreadsPerMultiProcessor), launch_resources<1>(prop.maxThreadsPerMultiProcessor),
            launch_resources<4,2>(prop.maxThreadsPerMultiProcessor)});
        int runtime_version=0,driver_version=0;check(cudaRuntimeGetVersion(&runtime_version));check(cudaDriverGetVersion(&driver_version));
        result["cuda_runtime_version"]=runtime_version;result["cuda_driver_version"]=driver_version;
        result["tolerances"]={{"dot_absolute_floor",2e-5},{"dot_l1_factor",2e-5},{"output_relative_floor",5e-5},
            {"output_absolute_floor",5e-5},{"silu_derivative_bound",1.1},{"candidate_stock_cpu_bound_multiplier",2}};
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        std::ifstream model(fixture.at("model_path").get<std::string>(),std::ios::binary);
        if(!model)throw std::runtime_error("Cannot open model");
        std::vector<std::unique_ptr<Weights>> weights;
        for(const auto &entry:fixture.at("pairs")) {
            auto w=std::make_unique<Weights>(entry.at("layer"),fixture.at("n"),fixture.at("k"));
            for(auto pair:{std::make_pair("up_offset",&w->up),std::make_pair("gate_offset",&w->gate)}) {
                model.seekg(entry.at(pair.first).get<uint64_t>());
                model.read(reinterpret_cast<char *>(pair.second->data()),pair.second->size()*sizeof(block_q4_K));
                if(!model)throw std::runtime_error("Model tensor read failed");
            }
            w->upload();
            if(w->layer==0||w->layer==17||w->layer==35) {
                for(int scenario=0;scenario<6;scenario++)result["correctness"].push_back(correctness(*w,scenario,stream));
            }
            // Keep real tensors on the GPU; only validation layers need host data temporarily.
            std::vector<block_q4_K>().swap(w->up);std::vector<block_q4_K>().swap(w->gate);
            weights.push_back(std::move(w));
        }
        for(int pattern=0;pattern<3;pattern++) {
            Weights w(-1-pattern,128,512);
            for(size_t i=0;i<w.up.size();i++)for(auto *b:{&w.up[i],&w.gate[i]}) {
                b->dm=make_half2(pattern==0?0.0f:1.0f/64,pattern==2?1.0f/128:0);
                for(int j=0;j<12;j++)b->scales[j]=pattern==1?255:uint8_t((i*37+j*53)&255);
                for(int j=0;j<128;j++)b->qs[j]=pattern==2?uint8_t((i+j)%2?0x0f:0xf0):uint8_t((i*13+j*17)&255);
            }
            w.upload();
            for(int scenario=0;scenario<6;scenario++)result["correctness"].push_back(correctness(w,scenario,stream));
        }
        bool all=true;for(const auto &c:result["correctness"])all=all&&c.at("passed").get<bool>();
        result["all_correctness_passed"]=all;
        if(!all)throw std::runtime_error("Correctness failed; performance tests skipped");
        std::vector<std::array<bool,3>> modes;
        for(bool rotate:{false,true})for(bool include_quant:{false,true})for(bool split:{false,true})modes.push_back({rotate,include_quant,split});
        std::mt19937 ordering(42);std::shuffle(modes.begin(),modes.end(),ordering);
        result["timing_order_seed"]=42;result["device_weight_working_set_bytes"]=uint64_t(36)*2*12288*4096/256*144;
        result["fused_sample_order"] = "All 24 permutations of stock, two warps, one warp and two rows, once each";
        for (const auto &mode : modes) {
            for (auto &row : timing(weights,mode[0],mode[1],mode[2],stream)) {
                result["timings"].push_back(std::move(row));
            }
        }
        result["status"]="passed";check(cudaStreamDestroy(stream));
        std::ofstream(argv[2])<<result.dump(2)<<"\n";
        return 0;
    } catch(const std::exception &error) {
        result["status"]="failed";result["error"]=error.what();
        if(argc==3)std::ofstream(argv[2])<<result.dump(2)<<"\n";
        std::cerr<<error.what()<<std::endl;return 1;
    }
}
