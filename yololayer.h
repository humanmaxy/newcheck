#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include <vector>
#include <string>
#include <iostream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << " at line " << __LINE__ << std::endl; \
            abort(); \
        } \
    } while (0)

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 17;
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };

    struct Detection
    {
        float bbox[4];  // x, y, w, h
        float conf;     // bbox_conf * cls_conf
        float class_id;
    };
}

class YoloLayerPlugin : public nvinfer1::IPluginV2IOExt
{
public:
    YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
    YoloLayerPlugin(const void* data, size_t length);
    ~YoloLayerPlugin();

    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    virtual void terminate() noexcept override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

    virtual int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    virtual size_t getSerializationSize() const noexcept override;

    virtual void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override;

    void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2IOExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

    int mClassCount;
    int mYoloV5NetWidth;
    int mYoloV5NetHeight;
    int mMaxOutObject;
    int mKernelCount;
    std::vector<Yolo::YoloKernel> mYoloKernel;
    void* mAnchor{nullptr};
    std::string mPluginNamespace;
};

class YoloPluginCreator : public nvinfer1::IPluginCreator
{
public:
    YoloPluginCreator();

    ~YoloPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2IOExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

// Forward declaration for plugin registration
extern "C" {
    bool initLibNvInferPlugins(void* logger, const char* libNamespace);
}

#endif // YOLO_LAYER_H