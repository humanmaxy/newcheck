#ifndef COMMONDATA_H
#define COMMONDATA_H

#include <string>
#include <vector>

// Status codes
enum StatusCode {
    SUCCESS = 0,
    ERROR_INIT_FAILED = 1,
    ERROR_INFERENCE_FAILED = 2,
    ERROR_INVALID_INPUT = 3
};

// Defect data structure
struct DefectData {
    int x, y, w, h;     // Bounding box
    float confidence;   // Detection confidence
    std::string type;   // Defect type
    int class_id;       // Class ID
};

// Defect data for C# interop
struct DefectDataCSharp {
    int x, y, w, h;
    float confidence;
    char type[64];
    int class_id;
};

// Tape position structure
struct TapePosition {
    int top_y;
    int bottom_y;
};

// AI Parameters structure
struct AI_Params {
    float threshold;
    float nms_threshold;
    int max_detections;
};

#endif // COMMONDATA_H