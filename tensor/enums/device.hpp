#include <string.h>
#include "file_loaders/json.hpp"
#ifndef DEVICE_TYPE
#define DEVICE_TYPE
enum DeviceType
{
    kCPU,
    kGPU
};

std::ostream &operator<<(std::ostream &os, const DeviceType &dtype)
{
    std::string s;
    switch (dtype)
    {
    case DeviceType::kCPU:
        s = "CPU";
        break;
    case DeviceType::kGPU:
        s = "GPU";
        break;
    }
    os << s;
    return os;
}

NLOHMANN_JSON_SERIALIZE_ENUM(DeviceType, {
                                             {kCPU, "CPU"},
                                             {kGPU, "GPU"}
                                         })

#endif