#include "yololayer.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;

namespace nvinfer1
{
    // Static member definitions for YoloPluginCreator
    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    // YoloLayerPlugin implementation
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<Yolo::YoloKernel>& vYoloKernel)
        : mClassCount(classCount), mYoloV5NetWidth(netWidth), mYoloV5NetHeight(netHeight), mMaxOutObject(maxOut), is_segmentation_(is_segmentation), mYoloKernel(vYoloKernel)
    {
        mKernelCount = vYoloKernel.size();
        mPluginNamespace = "";
        
        // Allocate memory for anchors
        mAnchor = new void*[mKernelCount];
        for (int i = 0; i < mKernelCount; i++) {
            float* anchor = new float[Yolo::CHECK_COUNT * 2];
            memcpy(anchor, mYoloKernel[i].anchors, Yolo::CHECK_COUNT * 2 * sizeof(float));
            mAnchor[i] = anchor;
        }
    }

    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        const char* d = reinterpret_cast<const char*>(data);
        
        // Deserialize plugin data
        mClassCount = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        mThreadCount = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        mKernelCount = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        mYoloV5NetWidth = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        mYoloV5NetHeight = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        mMaxOutObject = *reinterpret_cast<const int*>(d);
        d += sizeof(int);
        
        is_segmentation_ = *reinterpret_cast<const bool*>(d);
        d += sizeof(bool);
        
        // Deserialize kernels
        mYoloKernel.resize(mKernelCount);
        for (int i = 0; i < mKernelCount; i++) {
            mYoloKernel[i] = *reinterpret_cast<const Yolo::YoloKernel*>(d);
            d += sizeof(Yolo::YoloKernel);
        }
        
        // Allocate memory for anchors
        mAnchor = new void*[mKernelCount];
        for (int i = 0; i < mKernelCount; i++) {
            float* anchor = new float[Yolo::CHECK_COUNT * 2];
            memcpy(anchor, mYoloKernel[i].anchors, Yolo::CHECK_COUNT * 2 * sizeof(float));
            mAnchor[i] = anchor;
        }
        
        mPluginNamespace = "";
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {
        if (mAnchor) {
            for (int i = 0; i < mKernelCount; i++) {
                delete[] static_cast<float*>(mAnchor[i]);
            }
            delete[] mAnchor;
        }
    }

    // Plugin interface implementations
    const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT
    {
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, is_segmentation_, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return sizeof(int) * 6 + sizeof(bool) + mKernelCount * sizeof(Yolo::YoloKernel);
    }

    void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT
    {
        char* d = reinterpret_cast<char*>(buffer);
        
        *reinterpret_cast<int*>(d) = mClassCount;
        d += sizeof(int);
        
        *reinterpret_cast<int*>(d) = mThreadCount;
        d += sizeof(int);
        
        *reinterpret_cast<int*>(d) = mKernelCount;
        d += sizeof(int);
        
        *reinterpret_cast<int*>(d) = mYoloV5NetWidth;
        d += sizeof(int);
        
        *reinterpret_cast<int*>(d) = mYoloV5NetHeight;
        d += sizeof(int);
        
        *reinterpret_cast<int*>(d) = mMaxOutObject;
        d += sizeof(int);
        
        *reinterpret_cast<bool*>(d) = is_segmentation_;
        d += sizeof(bool);
        
        for (int i = 0; i < mKernelCount; i++) {
            *reinterpret_cast<Yolo::YoloKernel*>(d) = mYoloKernel[i];
            d += sizeof(Yolo::YoloKernel);
        }
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        return Dims3(1 + mMaxOutObject * sizeof(Yolo::Detection) / sizeof(float), 1, 1);
    }

    int YoloLayerPlugin::initialize() TRT_NOEXCEPT
    {
        return 0;
    }

    int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        // Forward GPU implementation would go here
        // For now, return success to avoid crashes during initialization
        return 0;
    }

    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT
    {
    }

    void YoloLayerPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize)
    {
        // GPU forward implementation would go here
    }

    // YoloPluginCreator implementation
    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("netinfo", nullptr, PluginFieldType::kFLOAT32, 5));
        mPluginAttributes.emplace_back(PluginField("kernels", nullptr, PluginFieldType::kFLOAT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        const PluginField* fields = fc->fields;
        int netinfo[5];
        std::vector<Yolo::YoloKernel> kernels;

        for (int i = 0; i < fc->nbFields; ++i) {
            if (strcmp(fields[i].name, "netinfo") == 0) {
                memcpy(netinfo, fields[i].data, sizeof(int) * 5);
            } else if (strcmp(fields[i].name, "kernels") == 0) {
                int kernelSize = fields[i].length;
                const Yolo::YoloKernel* data = static_cast<const Yolo::YoloKernel*>(fields[i].data);
                for (int k = 0; k < kernelSize; ++k) {
                    kernels.push_back(data[k]);
                }
            }
        }

        YoloLayerPlugin* obj = new YoloLayerPlugin(netinfo[0], netinfo[1], netinfo[2], netinfo[3], (bool)netinfo[4], kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    // Safe plugin registration - avoid global static initialization issues
    static YoloPluginCreator* g_yoloPluginCreator = nullptr;

    extern "C" {
        // This function will be called to initialize the plugin
        bool initLibNvInferPlugins(void* logger, const char* libNamespace)
        {
            if (!g_yoloPluginCreator) {
                g_yoloPluginCreator = new YoloPluginCreator();
                getPluginRegistry()->registerCreator(*g_yoloPluginCreator, libNamespace ? libNamespace : "");
            }
            return true;
        }
    }
}