#ifndef LOGGING_H
#define LOGGING_H

#include <NvInfer.h>
#include <iostream>

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only output warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

#endif // LOGGING_H