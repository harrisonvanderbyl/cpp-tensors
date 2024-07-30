
// include malloc
#include "stdlib.h"
#include "enums/dtype.hpp"
#include "vector"
#include "enums/device.hpp"
#include <stdarg.h>
#include <string>
#include "shape.hpp"
#include <iostream>
#include <string.h>
#ifndef TENSOR
#define TENSOR

class Slice
{
public:
    int start;
    int end;
    int step;

    Slice(int start, int end = -1, int step = 1)
    {
        this->start = start;
        this->end = end;
        this->step = step;
    }
    
};

class Tensor
{
public:
    Shape shape;
    Shape strides;
    void *data = NULL;
    DataType dtype;
    size_t total_size = 0;
    size_t total_bytes = 0;
    DeviceType device_type = DeviceType::kCPU;

    void calculate_metadata()
    {
        total_bytes = shape.total_size() * dtype_size(dtype);
        
        if(shape.ndim == 0){
            return;
        }

        strides[shape.ndim - 1] = 1;
        total_size = shape.total_size();
        for (int i = shape.ndim - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    Tensor(Shape __a, DataType dtype = DataType::kFLOAT_32, DeviceType device_type = DeviceType::kCPU)
    {
        this->dtype = dtype;
        this->device_type = device_type;
        // printf("ndim: %d\n", __a.ndim);
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        data = malloc(total_bytes);
    }

    Tensor(Shape __a, void *datain, DataType dtype = DataType::kFLOAT_32, DeviceType device_type = DeviceType::kCPU)
    {
        this->dtype = dtype;
        this->device_type = device_type;
        // printf("ndim: %d\n", __a.ndim);
        this->shape = __a;
        this->strides = __a.clone();
        calculate_metadata();
        this->data = datain;
    }

    template <typename T>
    Tensor(T a)
    {
        this->dtype = get_dtype<T>();
        this->device_type = DeviceType::kCPU;
        this->shape = Shape(1);
        this->strides = Shape(1);
        calculate_metadata();
        data = malloc(total_bytes);
        *((T *)data) = a;
    }

    template <typename T>
    inline int operator=(T a)
    {
        switch (dtype)
        {
        case DataType::kUINT_8:
        {
            uint8_t uint8_taa = (a);
            std::fill_n((uint8_t *)data, total_size, uint8_taa);
            break;
        }
        case DataType::kINT_8:
        {
            int8_t int8_taa = (a);
            std::fill_n((int8_t *)data, total_size, int8_taa);
            break;
        }
        case DataType::kBFLOAT_16:
        {
            uint16_t int16_taa = (a);
            std::fill_n((uint16_t *)data, total_size, int16_taa);
            break;
        }
        case DataType::kFLOAT_16:
        {
            float16 fp16_taa = (a);
            std::fill_n((uint16_t *)data, total_size, fp16_taa);
            break;
        }
        case DataType::kFLOAT_32:
        {
            float floataa = (a);
            std::fill_n((float *)data, total_size, floataa);
            break;
        }
        case DataType::kFLOAT_64:
        {
            double doubleaa = (a);
            std::fill_n((double *)data, total_size, doubleaa);
            break;
        }
        case DataType::kINT_32:
        {
            int32_t int32_taa = (a);
            std::fill_n((int32_t *)data, total_size, int32_taa);
            break;
        }
        case DataType::kINT_64:
        {
            int64_t int64_taa = (a);
            std::fill_n((int64_t *)data, total_size, int64_taa);
            break;
        }
        case DataType::kUINT_32:
        {
            uint32_t int32_taa = (a);
            std::fill_n((uint32_t *)data, total_size, int32_taa);
            break;
        }
        case DataType::kUINT_64:
        {
            uint64_t uint64_taa = (a);
            std::fill_n((uint64_t *)data, total_size, uint64_taa);
            break;
        }
        }
        return 0;
    }

    inline int operator=(const Tensor& other)
    {
        if (data == NULL)
        {
            data = other.data;
            shape = other.shape;
            strides = other.strides;
            dtype = other.dtype;
            device_type = other.device_type;
            total_size = other.total_size;
            total_bytes = other.total_bytes;

            return 0;
        }
        if (other.dtype != dtype)
        {
            std::cerr << "Dtypes do not match: A: " << dtype << " B: " << other.dtype << "" << std::endl;
            throw std::runtime_error("Dtypes do not match");
        }

        if(
            shape.total_size() == ((Tensor*)(&other))->shape.total_size() && strides.total_size() == ((Tensor*)(&other))->strides.total_size()
        ){

            memcpy(data, other.data, total_bytes);
            return 0;
        }

        if(
            other.shape.ndim == 0 || (other.shape.shape[0] == 1 && other.shape.ndim == 1)
        ){
            auto bytes = dtype_size(dtype);
            std::cout << "Bytes: " << bytes * total_size << std::endl;
            std::cout << "Total bits: " << total_size << std::endl;
            switch (bytes)
            {   
                case 1:
                    std::fill_n((uint8_t *)data, total_size, *((uint8_t *)other.data));
                    break;
                case 2:
                    std::fill_n((uint16_t *)data, total_size, *((uint16_t *)other.data));
                    break;
                case 4:
                    std::fill_n((uint32_t *)data, total_size, *((uint32_t *)other.data));
                    break;
                case 8:
                    std::fill_n((uint64_t *)data, total_size, *((uint64_t *)other.data));
                    break;
                default:
                    break;
            }
                
            return 0;
        }

        if (shape.ndim * other.shape.ndim == 0)
        {
            
            auto bytes = dtype_size(dtype);
            memcpy(data, other.data, bytes);
            return 0;

        }
    
        auto bcast = ((Tensor*)(&other))->broadcast(shape);

        for (int i = 0; i < shape.shape[0]; i++)
        {
            auto bbx = bcast[i];
            auto aax = this->operator[](i);
            aax = bbx;
        }
        
        return 0;
    }

    inline Tensor operator[](Slice i)
    {
        

        if (i.start < -shape[0] || i.end <= -shape[0] || i.start >= shape[0] || i.end > shape[0])
        {
            std::cerr << "Index out of range" << std::endl;
            std::cerr << "Index: " << i.start << " to " << i.end << " Shape: " << shape << std::endl;
            throw std::runtime_error("Index out of range");
        }

        i.start = i.start % shape[0];
        if(i.end < 0){
            i.end = shape[0] + i.end + 1;
        }
        

        Tensor b = {shape.clone(), data, dtype, device_type};
        b.strides = strides.clone();
        // std::cout << (i.end - i.start) / i.step << std::endl;
        // std::cout << "Start: " << i.start << " End: " << i.end << " Step: " << i.step << std::endl;
        b.shape.shape[0] = (i.end - i.start) / i.step;
        // std::cout << "Shape: " << b.shape << std::endl;
        b.data = (void *)((uint8_t *)b.data + i.start * strides[0] * dtype_size(dtype));
        b.strides.shape[0] = strides[0] * i.step;
        return b;
    }

    inline Tensor operator[](int i)
    {
         if (i < -shape[0]  || i >= shape[0] )
        {
            std::cerr << "Index out of range" << std::endl;
            std::cerr << "Index: " << i << " Shape: " << shape << std::endl;
            throw std::runtime_error("Index out of range");
        }

        i = size_t(i) % shape[0];
        
        Tensor a;
        a.shape = Shape();
        a.strides = Shape();
        a.dtype = dtype;
        a.device_type = device_type;
        a.data = (void *)((uint8_t *)data + i * strides[0] * dtype_size(dtype));
        a.shape.ndim = shape.ndim - 1;
        a.shape.shape = shape.shape + 1;
        a.strides.ndim = strides.ndim - 1;
        a.strides.shape = strides.shape + 1;
        a.total_size = total_size / shape[0];
        a.total_bytes = total_bytes / shape[0];
        if(a.data == NULL){
            std::cerr << "Data is null" << std::endl;
            throw std::runtime_error("Data is null");
        }
        return a;
    }

    template <typename T>
    inline T& flattened_get(int i) const
    {
        if (i < 0 || i >= total_size)
        {
            std::cerr << "Index out of range" << std::endl;
            std::cerr << "Index: " << i << " Total size: " << total_size << std::endl;
            throw std::runtime_error("Index out of range");
        }

        size_t location = 0;
        size_t total_stride = shape.total_size();
        for (size_t j = 0; j < shape.ndim; j++)
        {
            total_stride /= shape.shape[j];
            location += (((i%(total_stride*shape.shape[j]))/total_stride)) * strides.shape[j];
            // i -= (i / total_stride) * total_stride;
        }
        // std::cout << "Location: " << location << std::endl;

        return *((T *)((uint8_t *)data + location * dtype_size(dtype)));
    }

    inline Tensor transpose()
    {
        Tensor a;
        a.shape = Shape(shape[1], shape[0]);
        a.strides = Shape(strides[1], strides[0]);
        a.dtype = dtype;
        a.device_type = device_type;
        a.total_size = total_size;
        a.total_bytes = total_bytes;
        a.data = data;
        
        return a;
    }

    inline Tensor contiguous()
    {
        Tensor a = {shape, dtype, device_type};
        a = *this;
        return a;
    }

    Tensor()
    {
    }

    Tensor broadcast(const Shape& a)
    {
        Tensor b{a, data, dtype, device_type};
        for (size_t i = 1; i < a.ndim+1; i++)
        {
            if(shape.ndim > i && a.shape[-i%a.ndim] != shape.shape[-i%shape.ndim] && shape.shape[-i%shape.ndim] != 1){
                std::cerr << "Incompatible shapes for broadcast" << std::endl;
                std::cerr << i << "\n";
                std::cerr << "Shape: " << shape << " Broadcast shape: " << a << std::endl;
                std::cerr << "Shape: " << shape.shape[-i%shape.ndim] << " Broadcast shape: " << a.shape[-i%a.ndim] << std::endl;
                throw std::runtime_error("Incompatible shapes for broadcast");
            }

            if (shape.ndim < i || shape[-i] == 1)
            {
                b.strides[i] = 0;
            }
        }

        return b;
    }

    template <typename T>
    inline T *as()
    {
        return (T *)data;
    }

    inline std::string value_string(){

        std::basic_stringstream<char> os;
        os << "\n[";
        if (shape.ndim > 1)
        {
            for (size_t i = 0; i < shape.shape[0]; i += 1)
            {
                os << this->operator[](i).value_string();
                if (i != shape.shape[0] - 1)
                {
                    os << ", ";
                }
            }
        }
        else
        {
            auto stride = 0;
            auto entries = 1;
            if(shape.ndim > 0){
                stride = strides.shape[0];
                entries = shape.shape[0];
                }

            for (size_t i = 0; i < entries; i += 1)
            {
                if (dtype == DataType::kUINT_8)
                {
                    os << *((uint8_t *)data + i * stride);
                }
                if (dtype == DataType::kINT_8)
                {
                    os << *((int8_t *)data + i * stride);
                }
                if (dtype == DataType::kINT_16)
                {
                    os << *((int16_t *)data + i * stride);
                }
                if (dtype == DataType::kUINT_16)
                {
                    os << *((uint16_t *)data + i * stride);
                }
                if (dtype == DataType::kFLOAT_16)
                {
                    os << float(*((float16 *)data + i * stride));
                }
                if (dtype == DataType::kBFLOAT_16)
                {
                    os << *((bfloat16 *)data + i * stride);
                }
                if (dtype == DataType::kINT_32)
                {
                    os << *((int32_t *)data + i * stride);
                }
                if (dtype == DataType::kUINT_32)
                {
                    os << *((uint32_t *)data + i * stride);
                }
                if (dtype == DataType::kFLOAT_32)
                {
                    os << *((float *)data + i * stride);
                }
                if (dtype == DataType::kFLOAT_64)
                {
                    os << *((double *)data + i * stride);
                }
                if (dtype == DataType::kINT_64)
                {
                    os << *((int64_t *)data + i * stride);
                }
                if (dtype == DataType::kUINT_64)
                {
                    os << *((uint64_t *)data + i * stride);
                }

                if (i != entries - 1)
                {
                    os << ", ";
                }
            }
        }
        os << "]";

        return os.str();
    }

    // print tensor
    friend std::ostream &operator<<(std::ostream &os, Tensor tensor)
    {
        std::cout << tensor.strides << std::endl;
        os << "Tensor(" << tensor.shape << ", " << tensor.dtype << ", " << tensor.device_type << ")\n";
        os << tensor.value_string();
        
        return os;
    }
};

#endif