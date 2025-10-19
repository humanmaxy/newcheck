#ifndef M_LOGGER_H
#define M_LOGGER_H

#include <string>

// Simple logging functions
void LogWriterFlush(const std::string& message);
void InitLogger();
void CloseLogger();

#endif // M_LOGGER_H