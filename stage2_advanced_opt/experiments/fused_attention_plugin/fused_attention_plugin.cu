// Fixed-shape reference plugin for RF-DETR encoder attention.
//
// This file contains two experimental implementations of a fixed-shape
// TensorRT attention plugin. The native path is a tiled online-softmax CUDA
// kernel. When compiled with TORCH_FLASH, the plugin dispatches to PyTorch's
// CUDA scaled-dot-product-attention operator, which selects FlashAttention for
// supported FP16/BF16 tensors. Both paths are research prototypes, not a
// standalone production runtime.

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef TORCH_FLASH
#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <optional>
#endif

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>

using namespace nvinfer1;

constexpr int kHeads = 12;
constexpr int kTokens = 1226;
constexpr int kHeadDim = 64;
constexpr int kTileTokens = 64;

template <typename T>
__device__ inline float load_value(T const* p, int i)
{
    return static_cast<float>(p[i]);
}

template <>
__device__ inline float load_value<__half>(__half const* p, int i)
{
    return __half2float(p[i]);
}

template <typename T>
__device__ inline void store_value(T* p, int i, float v)
{
    p[i] = static_cast<T>(v);
}

template <>
__device__ inline void store_value<__half>(__half* p, int i, float v)
{
    p[i] = __float2half(v);
}

// One block computes one (batch, head, query) row. Threads correspond to D.
// The implementation deliberately keeps only the online softmax state and
// never materializes the SxS score/probability matrix.
template <typename T>
__global__ void fused_attention_kernel(T const* q, T const* k, T const* v, T* out, int batch, int heads, int seq,
    int dim)
{
    int row = blockIdx.x;
    int totalRows = batch * heads * seq;
    if (row >= totalRows || dim > blockDim.x)
        return;
    int d = threadIdx.x;
    int qBase = row * dim;
    int batchIndex = row / (heads * seq);
    int headIndex = (row / seq) % heads;
    float acc = 0.0F;
    __shared__ float partial[64];
    __shared__ float state[2]; // running max, running exp sum
    if (d == 0) { state[0] = -INFINITY; state[1] = 0.0F; }
    __syncthreads();

    // One online pass: update the running max/sum and the unnormalized V
    // accumulator. This avoids materializing the SxS score matrix.
    for (int key = 0; key < seq; ++key)
    {
        int kBase = batchIndex * heads * dim * seq + headIndex * dim * seq + d * seq + key;
        partial[d] = load_value(q, qBase + d) * load_value(k, kBase);
        __syncthreads();
        if (d == 0)
        {
            float dot = 0.0F;
            for (int j = 0; j < dim; ++j) dot += partial[j];
            float oldMax = state[0];
            float newMax = fmaxf(oldMax, dot);
            float weight = expf(dot - newMax);
            state[1] = state[1] * expf(oldMax - newMax) + weight;
            state[0] = newMax;
            partial[0] = weight;
            partial[1] = oldMax;
        }
        __syncthreads();
        float oldMax = partial[1];
        float newMax = state[0];
        float weight = partial[0];
        int vBase = (row / seq * seq * dim) + key * dim;
        acc = acc * expf(oldMax - newMax) + weight * load_value(v, vBase + d);
        __syncthreads();
    }
    store_value(out, qBase + d, acc / state[1]);
}

// FlashAttention-style tiled kernel. Four warps process four query rows per
// block. K/V are loaded once per token tile into shared memory; the online
// softmax state stays in registers and no SxS score tensor is materialized.
template <typename T>
__global__ void flash_attention_kernel(T const* q, T const* k, T const* v, T* out, int batch, int heads, int seq,
    int dim)
{
    constexpr int kWarps = 4;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    int query = blockIdx.x * kWarps + warp;
    int headRow = blockIdx.y;
    int b = headRow / heads;
    int h = headRow % heads;
    bool valid = b < batch && query < seq;
    int qRow = (b * heads + h) * seq + query;
    int qBase = qRow * dim;
    float q0 = valid ? load_value(q, qBase + lane) : 0.0F;
    float q1 = valid ? load_value(q, qBase + lane + 32) : 0.0F;
    float acc0 = 0.0F, acc1 = 0.0F;
    float runningMax = -INFINITY;
    float runningSum = 0.0F;
    __shared__ float tileK[kTileTokens * kHeadDim];
    __shared__ float tileV[kTileTokens * kHeadDim];
    unsigned mask = 0xffffffffu;

    int keyTiles = (seq + kTileTokens - 1) / kTileTokens;
    for (int tile = 0; tile < keyTiles; ++tile)
    {
        int start = tile * kTileTokens;
        int count = min(kTileTokens, seq - start);
        int tileElems = count * dim;
        for (int i = tid; i < tileElems; i += blockDim.x)
        {
            int key = i / dim;
            int d = i - key * dim;
            int globalKey = start + key;
            int kIndex = (headRow * dim + d) * seq + globalKey;
            int vIndex = (headRow * seq + globalKey) * dim + d;
            tileK[key * dim + d] = load_value(k, kIndex);
            tileV[key * dim + d] = load_value(v, vIndex);
        }
        __syncthreads();

        for (int key = 0; key < count; ++key)
        {
            float dot = q0 * tileK[key * dim + lane] + q1 * tileK[key * dim + lane + 32];
            dot += __shfl_down_sync(mask, dot, 16);
            dot += __shfl_down_sync(mask, dot, 8);
            dot += __shfl_down_sync(mask, dot, 4);
            dot += __shfl_down_sync(mask, dot, 2);
            dot += __shfl_down_sync(mask, dot, 1);
            float score = __shfl_sync(mask, dot, 0);

            float oldMax = runningMax;
            float weight = 0.0F;
            if (lane == 0)
            {
                float newMax = fmaxf(oldMax, score);
                weight = expf(score - newMax);
                runningSum = runningSum * expf(oldMax - newMax) + weight;
                runningMax = newMax;
            }
            float newMax = __shfl_sync(mask, runningMax, 0);
            weight = __shfl_sync(mask, weight, 0);
            runningMax = newMax;
            runningSum = __shfl_sync(mask, runningSum, 0);
            float rescale = expf(oldMax - newMax);
            acc0 = acc0 * rescale + weight * tileV[key * dim + lane];
            acc1 = acc1 * rescale + weight * tileV[key * dim + lane + 32];
        }
        __syncthreads();
    }

    float denom = __shfl_sync(mask, runningSum, 0);
    if (valid)
    {
        store_value(out, qBase + lane, acc0 / denom);
        store_value(out, qBase + lane + 32, acc1 / denom);
    }
}

class FusedAttentionPlugin final : public IPluginV2DynamicExt
{
public:
    FusedAttentionPlugin() = default;
    FusedAttentionPlugin(FusedAttentionPlugin const&) = default;

    char const* getPluginType() const noexcept override
    {
#ifdef TORCH_FLASH
        return "RFDetrTorchFlashAttention";
#else
        return "RFDetrFusedAttention";
#endif
    }
    char const* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder&) noexcept override
    {
        return (nbInputs == 3) ? inputs[0] : DimsExprs{};
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 3 || nbOutputs != 1)
            return false;
        if (pos == 0)
            return inOut[pos].format == TensorFormat::kLINEAR
                && (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == inOut[0].type;
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        auto const& dims = inputDesc[0].dims;
        if (dims.nbDims != 4 || dims.d[1] != kHeads || dims.d[2] != kTokens || dims.d[3] != kHeadDim)
        {
            std::fprintf(stderr, "RFDetrFusedAttention unexpected dims nb=%d [%lld,%lld,%lld,%lld]\n", dims.nbDims,
                static_cast<long long>(dims.nbDims > 0 ? dims.d[0] : -1),
                static_cast<long long>(dims.nbDims > 1 ? dims.d[1] : -1),
                static_cast<long long>(dims.nbDims > 2 ? dims.d[2] : -1),
                static_cast<long long>(dims.nbDims > 3 ? dims.d[3] : -1));
            return 1;
        }
        int batch = dims.d[0];
#ifdef TORCH_FLASH
        try
        {
            auto scalarType = inputDesc[0].type == DataType::kHALF ? at::kHalf : at::kFloat;
            auto options = at::TensorOptions().device(at::kCUDA).dtype(scalarType);
            auto qTensor = at::from_blob(const_cast<void*>(inputs[0]), {batch, kHeads, kTokens, kHeadDim},
                [](void*) {}, options);
            auto kTensor = at::from_blob(const_cast<void*>(inputs[1]), {batch, kHeads, kTokens, kHeadDim},
                [](void*) {}, options);
            auto vTensor = at::from_blob(const_cast<void*>(inputs[2]), {batch, kHeads, kTokens, kHeadDim},
                [](void*) {}, options);
            auto outTensor = at::from_blob(outputs[0], {batch, kHeads, kTokens, kHeadDim}, [](void*) {}, options);
            c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, 0));
            // The public SDPA dispatcher consumes the graph's [B,H,S,D]
            // layout and selects the CUDA FlashAttention backend when the
            // dtype and shape are supported. The lower-level
            // _flash_attention_forward_no_dropout_inplace API expects a
            // different packed layout ([B,S,H,D]); calling it directly here
            // silently produced incorrect attention values.
            auto result = at::scaled_dot_product_attention(qTensor, kTensor, vTensor, std::nullopt, 0.0, false, 1.0,
                false);
            outTensor.copy_(result);
            auto status = cudaGetLastError();
            if (status != cudaSuccess)
                std::fprintf(stderr, "RFDetrTorchFlashAttention failed: %s\n", cudaGetErrorString(status));
            return status == cudaSuccess ? 0 : 1;
        }
        catch (std::exception const& e)
        {
            std::fprintf(stderr, "RFDetrTorchFlashAttention exception: %s\n", e.what());
            return 1;
        }
#else
        int rows = batch * kHeads * kTokens;
        if (inputDesc[0].type == DataType::kHALF)
            flash_attention_kernel<<<dim3((kTokens + 4 - 1) / 4, batch * kHeads),
                dim3(128), 0, stream>>>(static_cast<__half const*>(inputs[0]), static_cast<__half const*>(inputs[1]),
                static_cast<__half const*>(inputs[2]), static_cast<__half*>(outputs[0]), batch, kHeads, kTokens,
                kHeadDim);
        else
            flash_attention_kernel<<<dim3((kTokens + 4 - 1) / 4, batch * kHeads),
                dim3(128), 0, stream>>>(static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
                static_cast<float const*>(inputs[2]), static_cast<float*>(outputs[0]), batch, kHeads, kTokens,
                kHeadDim);
        auto status = cudaGetLastError();
        if (status != cudaSuccess)
            std::fprintf(stderr, "RFDetrFusedAttention launch failed: %s\n", cudaGetErrorString(status));
        return status == cudaSuccess ? 0 : 1;
#endif
    }

    DataType getOutputDataType(int32_t, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        return nbInputs > 0 ? inputTypes[0] : DataType::kFLOAT;
    }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    FusedAttentionPlugin* clone() const noexcept override { return new FusedAttentionPlugin(*this); }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
};

class FusedAttentionPluginCreator final : public IPluginCreator
{
public:
    char const* getPluginName() const noexcept override
    {
#ifdef TORCH_FLASH
        return "RFDetrTorchFlashAttention";
#else
        return "RFDetrFusedAttention";
#endif
    }
    char const* getPluginVersion() const noexcept override { return "1"; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFields; }
    IPluginV2* createPlugin(char const*, PluginFieldCollection const*) noexcept override
    {
        return new FusedAttentionPlugin();
    }
    IPluginV2* deserializePlugin(char const*, void const*, size_t) noexcept override
    {
        return new FusedAttentionPlugin();
    }
    void setPluginNamespace(char const* ns) noexcept override { mNamespace = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    PluginFieldCollection mFields{0, nullptr};
    std::string mNamespace;
};

// TensorRT's --dynamicPlugins loader looks for this C symbol. Static
// REGISTER_TENSORRT_PLUGIN registration is kept as well for applications
// that use the global plugin registry.
extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    static FusedAttentionPluginCreator creator;
    static IPluginCreatorInterface* creators[] = {&creator};
    nbCreators = 1;
    return creators;
}

extern "C" TENSORRTAPI void setLoggerFinder(ILoggerFinder*) {}

REGISTER_TENSORRT_PLUGIN(FusedAttentionPluginCreator);
