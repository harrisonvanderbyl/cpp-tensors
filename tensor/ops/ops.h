
#include <iostream>
#include <map>
#include "enums/dtype.hpp"
#include "enums/device.hpp"
#include "tensor.hpp"
#include <stdarg.h>
// ops is an array of function pointers for dynamic library inclusion
#ifndef OPS
#define OPS

typedef void (*funcpointer)(void*, void*, void*, size_t&);

template <typename T>
Tensor applyOperation(Tensor a, Tensor b, void (*op)(T&, T&, T&, size_t&) )
{
    Shape output;
    auto longestOutputDims = std::max(a.shape.ndim, b.shape.ndim);
    output.ndim = longestOutputDims;
    output.shape = new int[output.ndim];
    for (int i = 0; i < output.ndim; i++)
    {
        int ashape = ((a.shape.ndim - output.ndim + i) < 0) ? 1 : a.shape.shape[a.shape.ndim - output.ndim + i];
        int bshape = ((b.shape.ndim - output.ndim + i) < 0) ? 1 : b.shape.shape[b.shape.ndim - output.ndim + i];
        if (ashape != bshape && ashape != 1 && bshape != 1)
        {
            std::cout << "Error: Incompatible shapes for multiplication\n"
                      << ashape << " " << bshape << "\n";
            exit(0);
        }
        output.shape[i] = std::max(ashape, bshape);
    }

    Tensor out = Tensor(output, a.dtype, a.device_type);

    auto aa = a.broadcast(output);
    auto bb = b.broadcast(output);

    for (size_t i = 0; i < out.total_size; )
    {
        op(aa.flattened_get<T>(i) , bb.flattened_get<T>(i), out.flattened_get<T>(i), i);
    }

    return out;
}

template <typename T>
void add(T& a, T& b, T& out, size_t& i)
{
    out = a + b;
    i++;
}

template <typename T>
void sub(T& a, T& b, T& out, size_t& i)
{
    out = a - b;
    i++;
}

template <typename T>
void mul(T& a, T& b, T& out, size_t& i)
{
    out = a * b;
    i++;
}

template <typename T>
void div(T& a, T& b, T& out, size_t& i)
{
    out = a / b;
    i++;
}

template <typename T>
void pow(T& a, T& b, T& out, size_t& i)
{
    out = pow(a, b);
    i++;
}


#define MakeOp(OP) \
{\
    switch (a.dtype)\
    {\
    case DataType::kUINT_8:\
        return applyOperation<uint8_t>(a, b, OP<uint8_t>);\
    case DataType::kINT_8:\
        return applyOperation<int8_t>(a, b, OP<int8_t>);\
    case DataType::kINT_16:\
        return applyOperation<int16_t>(a, b, OP<int16_t>);\
    case DataType::kUINT_16:\
        return applyOperation<uint16_t>(a, b, OP<uint16_t>);\
    case DataType::kFLOAT_16:\
        return applyOperation<float16>(a, b, OP<float16>);\
    case DataType::kBFLOAT_16:\
        return applyOperation<bfloat16>(a, b, OP<bfloat16>);\
    case DataType::kINT_32:\
        return applyOperation<int32_t>(a, b, OP<int32_t>);\
    case DataType::kUINT_32:\
        return applyOperation<uint32_t>(a, b, OP<uint32_t>);\
    case DataType::kFLOAT_32:\
        return applyOperation<float>(a, b, OP<float>);\
    case DataType::kFLOAT_64:\
        return applyOperation<double>(a, b, OP<double>);\
    case DataType::kINT_64:\
        return applyOperation<int64_t>(a, b, OP<int64_t>);\
    case DataType::kUINT_64:\
        return applyOperation<uint64_t>(a, b, OP<uint64_t>);\
    default:\
        std::cout << "Error: Unsupported data type for operation\n";\
        exit(0);\
    }\
}\

Tensor operator-(Tensor a, Tensor b)
{
    MakeOp(sub)
}

Tensor operator+(Tensor a, Tensor b)
{
    MakeOp(add)
}

Tensor operator*(Tensor a, Tensor b)
{
    MakeOp(mul)
}

Tensor operator/(Tensor a, Tensor b)
{
    MakeOp(div)
}

Tensor operator^(Tensor a, Tensor b)
{
    MakeOp(pow)
}

#endif