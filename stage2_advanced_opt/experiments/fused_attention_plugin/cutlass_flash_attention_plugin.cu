// Standalone CUTLASS FlashAttention-style TensorRT plugin.
//
// The TensorRT graph exposes Q/V as [B,H,S,D] and K as [B,H,D,S], matching the
// exported RF-DETR graph.  CUTLASS's FMHA example uses BMHK storage, so this
// plugin transposes into a workspace, runs the fused online-softmax kernel,
// and transposes the result back.  It has no PyTorch/libtorch dependency.

#include "cutlass_kernel_forward.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

using namespace nvinfer1;

namespace rfdetr_plugin_internal {
constexpr int kHeads = 12;
constexpr int kTokens = 1226;
constexpr int kHeadDim = 64;
using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>;

__global__ void bhsd_to_bmhk(__half const* src, cutlass::half_t* dst, int batch, int heads, int seq, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * dim;
    if (index >= total) return;
    int d = index % dim;
    int t = index / dim;
    int s = t % seq;
    int h = (t / seq) % heads;
    int b = t / (seq * heads);
    // Source is [B,H,S,D], destination is [B,S,H,D].
    int src_index = ((b * heads + h) * seq + s) * dim + d;
    int dst_index = ((b * seq + s) * heads + h) * dim + d;
    dst[dst_index] = reinterpret_cast<cutlass::half_t const*>(src)[src_index];
}

__global__ void bhds_to_bmhk(__half const* src, cutlass::half_t* dst, int batch, int heads, int seq, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * dim;
    if (index >= total) return;
    int d = index % dim;
    int t = index / dim;
    int s = t % seq;
    int h = (t / seq) % heads;
    int b = t / (seq * heads);
    // K source is [B,H,D,S], destination is [B,S,H,D].
    int src_index = ((b * heads + h) * dim + d) * seq + s;
    int dst_index = ((b * seq + s) * heads + h) * dim + d;
    dst[dst_index] = reinterpret_cast<cutlass::half_t const*>(src)[src_index];
}

__global__ void bmhk_to_bhsd(cutlass::half_t const* src, __half* dst, int batch, int heads, int seq, int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * dim;
    if (index >= total) return;
    int d = index % dim;
    int t = index / dim;
    int s = t % seq;
    int h = (t / seq) % heads;
    int b = t / (seq * heads);
    int src_index = ((b * seq + s) * heads + h) * dim + d;
    int dst_index = ((b * heads + h) * seq + s) * dim + d;
    reinterpret_cast<cutlass::half_t*>(dst)[dst_index] = src[src_index];
}
} // namespace rfdetr_plugin_internal

using rfdetr_plugin_internal::bhsd_to_bmhk;
using rfdetr_plugin_internal::bhds_to_bmhk;
using rfdetr_plugin_internal::bmhk_to_bhsd;
using rfdetr_plugin_internal::kHeadDim;
using rfdetr_plugin_internal::kHeads;
using rfdetr_plugin_internal::kTokens;
using Attention = rfdetr_plugin_internal::Attention;

class CutlassFlashAttentionPlugin final : public IPluginV2DynamicExt
{
public:
    char const* getPluginType() const noexcept override { return "RFDetrCutlassFlashAttention"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder&) noexcept override
    {
        return nbInputs == 3 ? inputs[0] : DimsExprs{};
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 3 || nbOutputs != 1) return false;
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kHALF;
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const*, int32_t) const noexcept override
    {
        if (nbInputs != 3 || inputs[0].dims.nbDims != 4) return 0;
        int64_t elements = 1;
        for (int i = 0; i < inputs[0].dims.nbDims; ++i) elements *= inputs[0].dims.d[i];
        // Q, K, V in BMHK plus output in BMHK.
        return static_cast<size_t>(elements) * sizeof(cutlass::half_t) * 4;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        auto const& dims = inputDesc[0].dims;
        if (dims.nbDims != 4 || dims.d[1] != kHeads || dims.d[2] != kTokens || dims.d[3] != kHeadDim)
            return 1;
        int batch = dims.d[0];
        size_t elements = static_cast<size_t>(batch) * kHeads * kTokens * kHeadDim;
        auto* q_bmhk = reinterpret_cast<cutlass::half_t*>(workspace);
        auto* k_bmhk = q_bmhk + elements;
        auto* v_bmhk = k_bmhk + elements;
        auto* o_bmhk = v_bmhk + elements;
        int blocks = static_cast<int>((elements + 255) / 256);

        bhsd_to_bmhk<<<blocks, 256, 0, stream>>>(static_cast<__half const*>(inputs[0]), q_bmhk, batch, kHeads,
            kTokens, kHeadDim);
        bhds_to_bmhk<<<blocks, 256, 0, stream>>>(static_cast<__half const*>(inputs[1]), k_bmhk, batch, kHeads,
            kTokens, kHeadDim);
        bhsd_to_bmhk<<<blocks, 256, 0, stream>>>(static_cast<__half const*>(inputs[2]), v_bmhk, batch, kHeads,
            kTokens, kHeadDim);

        Attention::Params p;
        p.query_ptr = q_bmhk;
        p.key_ptr = k_bmhk;
        p.value_ptr = v_bmhk;
        p.output_ptr = o_bmhk;
        p.scale = 1.0f;
        p.num_heads = kHeads;
        p.num_batches = batch;
        p.head_dim = kHeadDim;
        p.head_dim_value = kHeadDim;
        p.num_queries = kTokens;
        p.num_keys = kTokens;
        p.q_strideH = kHeadDim;
        p.k_strideH = kHeadDim;
        p.v_strideH = kHeadDim;
        p.q_strideM = kHeads * kHeadDim;
        p.k_strideM = kHeads * kHeadDim;
        p.v_strideM = kHeads * kHeadDim;
        p.q_strideB = static_cast<int64_t>(kTokens) * kHeads * kHeadDim;
        p.k_strideB = static_cast<int64_t>(kTokens) * kHeads * kHeadDim;
        p.v_strideB = static_cast<int64_t>(kTokens) * kHeads * kHeadDim;
        p.o_strideM = kHeads * kHeadDim;
        if (!Attention::check_supported(p)) return 1;

        constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
        constexpr int smem_bytes = sizeof(typename Attention::SharedStorage);
        if (smem_bytes > 0xc000)
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
        bmhk_to_bhsd<<<blocks, 256, 0, stream>>>(o_bmhk, static_cast<__half*>(outputs[0]), batch, kHeads,
            kTokens, kHeadDim);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    DataType getOutputDataType(int32_t, DataType const*, int32_t) const noexcept override { return DataType::kHALF; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    CutlassFlashAttentionPlugin* clone() const noexcept override { return new CutlassFlashAttentionPlugin(*this); }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* ns) noexcept override { namespace_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    std::string namespace_;
};

class CutlassFlashAttentionPluginCreator final : public IPluginCreator
{
public:
    char const* getPluginName() const noexcept override { return "RFDetrCutlassFlashAttention"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    IPluginV2* createPlugin(char const*, PluginFieldCollection const*) noexcept override
    {
        return new CutlassFlashAttentionPlugin();
    }
    IPluginV2* deserializePlugin(char const*, void const*, size_t) noexcept override
    {
        return new CutlassFlashAttentionPlugin();
    }
    void setPluginNamespace(char const* ns) noexcept override { namespace_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    PluginFieldCollection fields_{0, nullptr};
    std::string namespace_;
};

extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    static CutlassFlashAttentionPluginCreator creator;
    static IPluginCreatorInterface* creators[] = {&creator};
    nbCreators = 1;
    return creators;
}

extern "C" TENSORRTAPI void setLoggerFinder(ILoggerFinder*) {}

REGISTER_TENSORRT_PLUGIN(CutlassFlashAttentionPluginCreator);
