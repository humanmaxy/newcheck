#include "yololayer.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

using namespace nvinfer1;

// CUDA kernel declarations
extern "C" {
    void CalDetection(const float *input, float *output, int noElements, int yoloWidth, int yoloHeight, int maxoutobject, 
                     float confthresh, float *anchors, int classes, int outputElem, cudaStream_t stream);
}

// Plugin implementation
YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
{
    mClassCount = classCount;
    mYoloV5NetWidth = netWidth;
    mYoloV5NetHeight = netHeight;
    mMaxOutObject = maxOut;
    mYoloKernel = vYoloKernel;
    mKernelCount = vYoloKernel.size();

    CUDA_CHECK(cudaMalloc(&mAnchor, mKernelCount * sizeof(Yolo::YoloKernel)));
    CUDA_CHECK(cudaMemcpy(mAnchor, &mYoloKernel[0], mKernelCount * sizeof(Yolo::YoloKernel), cudaMemcpyHostToDevice));
}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    
    mClassCount = read<int>(d);
    mYoloV5NetWidth = read<int>(d);
    mYoloV5NetHeight = read<int>(d);
    mMaxOutObject = read<int>(d);
    mKernelCount = read<int>(d);
    
    mYoloKernel.resize(mKernelCount);
    for (int i = 0; i < mKernelCount; ++i)
    {
        mYoloKernel[i].width = read<int>(d);
        mYoloKernel[i].height = read<int>(d);
        for (int j = 0; j < Yolo::CHECK_COUNT * 2; ++j)
        {
            mYoloKernel[i].anchors[j] = read<float>(d);
        }
    }
    
    CUDA_CHECK(cudaMalloc(&mAnchor, mKernelCount * sizeof(Yolo::YoloKernel)));
    CUDA_CHECK(cudaMemcpy(mAnchor, &mYoloKernel[0], mKernelCount * sizeof(Yolo::YoloKernel), cudaMemcpyHostToDevice));
}

YoloLayerPlugin::~YoloLayerPlugin()
{
    if (mAnchor)
    {
        CUDA_CHECK(cudaFree(mAnchor));
        mAnchor = nullptr;
    }
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    return Dims3(1, (mMaxOutObject < 1000 ? mMaxOutObject : 1000) * sizeof(Yolo::Detection) / sizeof(float) + 1, 1);
}

int YoloLayerPlugin::initialize() noexcept
{
    return 0;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    forwardGpu(reinterpret_cast<const float* const*>(inputs), reinterpret_cast<float*>(outputs[0]), stream, batchSize);
    return 0;
}

size_t YoloLayerPlugin::getSerializationSize() const noexcept
{
    return sizeof(mClassCount) + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject) + 
           sizeof(mKernelCount) + mKernelCount * (sizeof(int) * 2 + sizeof(float) * Yolo::CHECK_COUNT * 2);
}

void YoloLayerPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    
    write(d, mClassCount);
    write(d, mYoloV5NetWidth);
    write(d, mYoloV5NetHeight);
    write(d, mMaxOutObject);
    write(d, mKernelCount);
    
    for (int i = 0; i < mKernelCount; ++i)
    {
        write(d, mYoloKernel[i].width);
        write(d, mYoloKernel[i].height);
        for (int j = 0; j < Yolo::CHECK_COUNT * 2; ++j)
        {
            write(d, mYoloKernel[i].anchors[j]);
        }
    }
}

bool YoloLayerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept
{
    // No additional configuration needed
}

const char* YoloLayerPlugin::getPluginType() const noexcept
{
    return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const noexcept
{
    return "1";
}

void YoloLayerPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2IOExt* YoloLayerPlugin::clone() const noexcept
{
    YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

DataType YoloLayerPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return DataType::kFLOAT;
}

bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize)
{
    void* devAnchor = mAnchor;
    size_t numElem = 0;
    for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
    {
        numElem += mYoloKernel[i].width * mYoloKernel[i].height * 3 * (5 + mClassCount);
    }
    
    CalDetection(inputs[0], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, 
                 0.1f, reinterpret_cast<float*>(devAnchor), mClassCount, 
                 1 + mMaxOutObject * sizeof(Yolo::Detection) / sizeof(float), stream);
}

// Plugin Creator implementation
PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

YoloPluginCreator::YoloPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const noexcept
{
    return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    
    int* netinfo = nullptr;
    Yolo::YoloKernel* kernels = nullptr;
    int kernels_size = 0;
    
    for (int i = 0; i < nbFields; ++i)
    {
        if (strcmp(fields[i].name, "netinfo") == 0)
        {
            netinfo = (int*)fields[i].data;
        }
        else if (strcmp(fields[i].name, "kernels") == 0)
        {
            kernels = (Yolo::YoloKernel*)fields[i].data;
            kernels_size = fields[i].length;
        }
    }
    
    if (netinfo && kernels)
    {
        std::vector<Yolo::YoloKernel> kernels_vec(kernels, kernels + kernels_size);
        YoloLayerPlugin* obj = new YoloLayerPlugin(netinfo[0], netinfo[1], netinfo[2], netinfo[3], kernels_vec);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    
    return nullptr;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

// Plugin registration implementation
namespace
{
    // Static instance of the plugin creator
    static YoloPluginCreator* g_yoloPluginCreator = nullptr;
    
    // Initialize the plugin creator when needed
    YoloPluginCreator* getYoloPluginCreator()
    {
        if (!g_yoloPluginCreator)
        {
            g_yoloPluginCreator = new YoloPluginCreator();
        }
        return g_yoloPluginCreator;
    }
}

// Plugin registration function
extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace)
{
    try
    {
        auto* registry = getPluginRegistry();
        if (registry)
        {
            auto* creator = getYoloPluginCreator();
            if (creator)
            {
                registry->registerCreator(*creator, libNamespace ? libNamespace : "");
                return true;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error registering YOLO plugin: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error registering YOLO plugin" << std::endl;
    }
    return false;
}