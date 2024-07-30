#include <string>
#include <typeinfo> 
#include <cstdint>
#include "file_loaders/json.hpp"
#include "bfloat16/bf16.hpp"
#ifndef DATA_TYPE
#define DATA_TYPE
enum DataType
{
    /// Boolean type
    kBOOL,
    /// Unsigned byte
    kUINT_8,
    /// Signed byte
    kINT_8,
    /// Signed integer (16-bit)
    kINT_16,
    /// Unsigned integer (16-bit)
    kUINT_16,
    /// Half-precision floating point
    kFLOAT_16,
    /// Brain floating point
    kBFLOAT_16,
    /// Signed integer (32-bit)
    kINT_32,
    /// Unsigned integer (32-bit)
    kUINT_32,
    /// Floating point (32-bit)
    kFLOAT_32,
    /// Floating point (64-bit)
    kFLOAT_64,
    /// Signed integer (64-bit)
    kINT_64,
    /// Unsigned integer (64-bit)
    kUINT_64,

};

NLOHMANN_JSON_SERIALIZE_ENUM(DataType, {
                                             {kBOOL, "BOOL"},
                                             {kUINT_8, "U8"},
                                             {kINT_8, "I8"},
                                             {kINT_16, "I16"},
                                             {kUINT_16, "U16"},
                                             {kFLOAT_16, "F16"},
                                             {kBFLOAT_16, "BF16"},
                                             {kINT_32, "I32"},
                                             {kUINT_32, "U32"},
                                             {kFLOAT_32, "F32"},
                                             {kFLOAT_64, "F64"},
                                             {kINT_64, "I64"},
                                             {kUINT_64, "U64"},
                                         })



// map from enum DataType to size of data type
int dtype_size(enum DataType dtype){
    switch(dtype){
        case kUINT_8:
            return 1;
        case kINT_8:
            return 1;
        case kINT_16:
            return 2;
        case kUINT_16:
            return 2;
        case kFLOAT_16:
            return 2;
        case kBFLOAT_16:
            return 2;
        case kINT_32:
            return 4;
        case kUINT_32:
            return 4;
        case kFLOAT_32:
            return 4;
        case kFLOAT_64:
            return 8;
        case kINT_64:
            return 8;
        case kUINT_64:
            return 8;

    }
    return 0;
}

template <typename T>
DataType get_dtype(){
    if(typeid(T) == typeid(uint8_t)){
        return kUINT_8;
    }else if(typeid(T) == typeid(int8_t)){
        return kINT_8;
    }else if(typeid(T) == typeid(int16_t)){
        return kINT_16;
    }else if(typeid(T) == typeid(uint16_t)){
        return kUINT_16;
    }else if(typeid(T) == typeid(float16)){
        return kFLOAT_16;
    }else if(typeid(T) == typeid(bfloat16)){
        return kBFLOAT_16;
    }else if(typeid(T) == typeid(float)){
        return kFLOAT_32;
    }else if(typeid(T) == typeid(double)){
        return kFLOAT_64;
    }else if(typeid(T) == typeid(int32_t)){
        return kINT_32;
    }else if(typeid(T) == typeid(int64_t)){
        return kINT_64;
    }else if(typeid(T) == typeid(uint32_t)){
        return kUINT_32;
    }else if(typeid(T) == typeid(uint64_t)){
        return kUINT_64;
    }

    throw "Unsupported data type:" + std::string(typeid(T).name());
}
std::ostream &operator<<(std::ostream &os, const DataType &dtype)
{
    std::string s;
    switch (dtype)
    {
    case kUINT_8:
        s = "U8";
        break;
    case kINT_8:
        s = "I8";
        break;
    case kINT_16:
        s = "I16";
        break;
    case kUINT_16:
        s = "U16";
        break;
    case kFLOAT_16:
        s = "F16";
        break;
    case kBFLOAT_16:
        s = "BF16";
        break;
    case kINT_32:
        s = "I32";
        break;
    case kUINT_32:
        s = "U32";
        break;
    case kFLOAT_32:
        s = "F32";
        break;
    case kFLOAT_64:
        s = "F64";
        break;
    case kINT_64:
        s = "I64";
        break;
    case kUINT_64:
        s = "U64";
        break;
    case kBOOL:
        s = "BOOL";
        break;
    }

    os << s;
    return os;
}




#endif