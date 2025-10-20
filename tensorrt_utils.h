#ifndef TENSORRT_UTILS_H
#define TENSORRT_UTILS_H

#include <NvInfer.h>
#include <iostream>

// Safe wrapper for addResize to handle TensorRT version differences
inline nvinfer1::IResizeLayer* safeAddResize(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, const char* layerName = "") {
    if (!network) {
        std::cerr << "Error: Network is null in safeAddResize" << std::endl;
        return nullptr;
    }
    
    // Check input tensor validity
    auto dims = input.getDimensions();
    if (dims.nbDims <= 0) {
        std::cerr << "Error: Invalid input tensor dimensions in " << layerName << std::endl;
        return nullptr;
    }
    
    // Print debug info
    std::cout << "Creating resize layer " << layerName << " with input dims: ";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << " ";
    }
    std::cout << std::endl;
    
    // Try to create the resize layer
    auto resizeLayer = network->addResize(input);
    if (!resizeLayer) {
        std::cerr << "Error: Failed to create resize layer " << layerName << std::endl;
        std::cerr << "This might be due to TensorRT version compatibility issues" << std::endl;
    }
    
    return resizeLayer;
}

// Safe wrapper for setting resize mode with error checking (InterpolationMode version)
inline bool safeSetResizeMode(nvinfer1::IResizeLayer* layer, nvinfer1::InterpolationMode mode, const char* layerName = "") {
    if (!layer) {
        std::cerr << "Error: Resize layer is null in safeSetResizeMode for " << layerName << std::endl;
        return false;
    }
    
    try {
        layer->setResizeMode(mode);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error setting resize mode for " << layerName << ": " << e.what() << std::endl;
        return false;
    }
}

// Safe wrapper for setting resize mode with error checking (ResizeMode version for compatibility)
inline bool safeSetResizeMode(nvinfer1::IResizeLayer* layer, nvinfer1::ResizeMode mode, const char* layerName = "") {
    if (!layer) {
        std::cerr << "Error: Resize layer is null in safeSetResizeMode for " << layerName << std::endl;
        return false;
    }
    
    try {
        layer->setResizeMode(mode);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error setting resize mode for " << layerName << ": " << e.what() << std::endl;
        return false;
    }
}

// Safe wrapper for setting output dimensions
inline bool safeSetOutputDimensions(nvinfer1::IResizeLayer* layer, const nvinfer1::Dims& dims, const char* layerName = "") {
    if (!layer) {
        std::cerr << "Error: Resize layer is null in safeSetOutputDimensions for " << layerName << std::endl;
        return false;
    }
    
    try {
        layer->setOutputDimensions(dims);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error setting output dimensions for " << layerName << ": " << e.what() << std::endl;
        return false;
    }
}

#endif // TENSORRT_UTILS_H