#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <map>
#include <NvInfer.h>

// Utility functions
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);
int get_width(int x, float gw);
int get_depth(int x, float gd);

#endif // UTILS_H