#include "utils.h"
#include <fstream>
#include <iostream>

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file) {
    std::cout << "Loading weights from " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;
    
    std::ifstream input(file, std::ios::in | std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Unable to load weight file " << file << std::endl;
        return weightMap;
    }
    
    // Implementation would depend on your specific weight file format
    // This is a placeholder
    
    return weightMap;
}

int get_width(int x, float gw) {
    return static_cast<int>(x * gw);
}

int get_depth(int x, float gd) {
    return static_cast<int>(x * gd);
}