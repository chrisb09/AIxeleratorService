#pragma once

#ifdef SCOREP
#include <nvml.h>

inline unsigned long long get_gpu_memory_used() {
    static bool initialized = false;
    static nvmlDevice_t device;
    if (!initialized) {
        if (nvmlInit() != NVML_SUCCESS) return 0;
        if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) return 0;
        initialized = true;
    }
    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
        return memory.used;
    }
    return 0;
}
#else
inline unsigned long long get_gpu_memory_used() { return 0; }
#endif
