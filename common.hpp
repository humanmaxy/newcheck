#ifndef COMMON_HPP
#define COMMON_HPP

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>

// Common TensorRT utilities for CUDA 12.9 + TensorRT 10.12 compatibility

namespace common {

// Logger class for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only output warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// Helper function to create resize layer with proper namespace
inline nvinfer1::IResizeLayer* addResizeLayer(nvinfer1::INetworkDefinition* network, 
                                              nvinfer1::ITensor& input, 
                                              nvinfer1::Dims outputDims) {
    auto* resize = network->addResize(input);
    if (resize) {
        // Fix for TensorRT 10.x: Use proper namespace for ResizeMode
        resize->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
        resize->setOutputDimensions(outputDims);
    }
    return resize;
}

// Helper function to configure builder for TensorRT 10.x
inline void configureBuilder(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config) {
    // Note: setMaxBatchSize is deprecated in TensorRT 10.x
    // Batch size is now handled via optimization profiles
    
    // Set memory pool limit (replaces setMaxWorkspaceSize)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20)); // 16MB
    
    // Enable FP16 if needed
    #ifdef USE_FP16
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    #endif
    
    // Enable INT8 if needed
    #ifdef USE_INT8
    if (builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    #endif
}

// Helper function for proper TensorRT object cleanup
template<typename T>
inline void safeDelete(T*& ptr) {
    if (ptr) {
        delete ptr;
        ptr = nullptr;
    }
}

// Helper function to setup execution context for TensorRT 10.x
inline bool setupExecutionContext(nvinfer1::ICudaEngine* engine, 
                                  nvinfer1::IExecutionContext* context,
                                  void** buffers,
                                  const char* inputBlobName,
                                  const char* outputBlobName) {
    if (!engine || !context) return false;
    
    // Check tensor count (replaces getNbBindings)
    if (engine->getNbIOTensors() != 2) return false;
    
    // Set tensor addresses (replaces getBindingIndex approach)
    context->setTensorAddress(inputBlobName, buffers[0]);
    context->setTensorAddress(outputBlobName, buffers[1]);
    
    return true;
}

// Helper function for inference with TensorRT 10.x
inline bool doInferenceV3(nvinfer1::IExecutionContext* context, cudaStream_t stream) {
    if (!context) return false;
    
    // Use enqueueV3 instead of enqueueV2
    return context->enqueueV3(stream);
}

} // namespace common

#endif // COMMON_HPP