#ifndef YOLOLAYER_H
#define YOLOLAYER_H

#include <vector>
#include <NvInfer.h>

// YOLOv5 constants and structures
namespace Yolo {
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int CLASS_NUM = 80;  // COCO classes, adjust as needed
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    
    struct Detection {
        float bbox[4];  // x, y, w, h
        float conf;     // confidence
        float class_id; // class id
    };
}

// Forward declarations to avoid circular dependencies
class Logger;

// Function declarations
nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition *network, 
                                        std::map<std::string, nvinfer1::Weights>& weightMap, 
                                        std::string lname, 
                                        std::vector<nvinfer1::IConvolutionLayer*> dets);

#endif // YOLOLAYER_H