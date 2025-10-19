#include <iostream>
#include "yololayer.h"
#include "plugin_init.h"
#include <NvInfer.h>

int main() {
    std::cout << "Testing YoloLayer plugin initialization..." << std::endl;
    
    // Test the safe plugin initialization
    if (initializeYoloPlugin()) {
        std::cout << "✓ Plugin initialized successfully!" << std::endl;
        
        // Test plugin registry access
        auto* registry = nvinfer1::getPluginRegistry();
        if (registry) {
            auto* creator = registry->getPluginCreator("YoloLayer_TRT", "1");
            if (creator) {
                std::cout << "✓ Plugin creator found in registry!" << std::endl;
                std::cout << "Plugin name: " << creator->getPluginName() << std::endl;
                std::cout << "Plugin version: " << creator->getPluginVersion() << std::endl;
            } else {
                std::cout << "✗ Plugin creator not found in registry!" << std::endl;
                return 1;
            }
        } else {
            std::cout << "✗ Plugin registry not accessible!" << std::endl;
            return 1;
        }
    } else {
        std::cout << "✗ Plugin initialization failed!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ All tests passed! The access violation should be fixed." << std::endl;
    return 0;
}