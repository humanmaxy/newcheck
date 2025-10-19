#include "m_logger.h"
#include <iostream>

void LogWriterFlush(const std::string& message) {
    std::cout << "[LOG] " << message << std::endl;
}

void InitLogger() {
    // Initialize logging system
}

void CloseLogger() {
    // Close logging system
}