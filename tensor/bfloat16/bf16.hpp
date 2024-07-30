#ifndef BF16_HPP
#define BF16_HPP

#include <cstdint>

struct float16
{
    uint16_t fvalue;
    operator uint16_t() const { return fvalue; }
    float16(float value)
    {
        uint32_t x = *((uint32_t *)&value);
        fvalue = (uint16_t)((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    }
    float16() { fvalue = 0; }
    float16 operator=(float value)
    {
        uint32_t x = *((uint32_t *)&value);
        fvalue = (uint16_t)((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
        return *this;
    }
    float16 operator=(float16 value)
    {
        fvalue = value.fvalue;
        return *this;
    }

    template <typename T>
    float16 operator+(T value)
    {
        return float16(float(*this) + float(value));
    }

    template <typename T>
    float16 operator-(T value)
    {
        return float16(float(*this) - float(value));
    }

    template <typename T>
    float16 operator*(T value)
    {
        return float16(float(*this) * float(value));
    }

    template <typename T>
    float16 operator/(T value)
    {
        return float16(float(*this) / float(value));
    }

    operator float() const
    {
        uint32_t x = ((fvalue & 0x8000) << 16) | (((fvalue & 0x7c00) + 0x1C000) << 13) | ((fvalue & 0x03FF) << 13);
        return *((float *)&x);
    }
    
};

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

#define bfloat16 __bf16

static float bfloat16_to_float32(bfloat16 value)
{
    auto x = (uint32_t(*(uint16_t *)(&value)) << 16);
    return *((float *)&x);
}

#elif defined(__CUDACC__)

#include <cuda_bf16.h>
#include <ostream>
#define bfloat16 __nv_bfloat16
static float bfloat16_to_float32(bfloat16 value)
{

    return __bfloat162float(value);
}

#else

#define BF16FALLBACKS

struct bfloat16;
static float bfloat16_to_float32(bfloat16 value);
static bfloat16 float32_to_bfloat16(float value);
struct bfloat16
{
    uint16_t value;
    operator float() const { return bfloat16_to_float32(*this); }
    bfloat16(double valuein) { this->value = float32_to_bfloat16((float)valuein); }
    bfloat16(float valuein) { this->value = float32_to_bfloat16(valuein); }
    bfloat16(uint16_t valuein) { this->value = valuein; }
    bfloat16() { this->value = 0; }
    bfloat16 operator=(float valuein)
    {
        this->value = float32_to_bfloat16(valuein);
        return *this;
    }
    bfloat16 operator=(bfloat16 valuein)
    {
        this->value = valuein.value;
        return *this;
    }
    // bfloat16 operator = (uint16_t value) {this->value = value; return *this;}
    // bfloat16 operator = (double value) {this->value = float32_to_bfloat16((float)value); return *this;}
    bfloat16 operator+=(bfloat16 valuein)
    {
        *this = *this + valuein;
        return *this;
    }

    template <typename T>
    bfloat16 operator-(T valuein) { return bfloat16(float(*this) - float(valuein)); }

    template <typename T>
    bfloat16 operator+(T valuein) { return bfloat16(float(*this) + float(valuein)); }

    template <typename T>
    bfloat16 operator*(T valuein) { return bfloat16(float(*this) * float(valuein)); }

    template <typename T>
    bfloat16 operator/(T valuein) { return bfloat16(float(*this) / float(valuein)); }

    
};

static float bfloat16_to_float32(bfloat16 value)
{
    // cast as uint16_t, then cast as float32, then bitshift 16 bits to the left, then cast as float32
    uint32_t inter(uint32_t((uint16_t)value.value) << 16);
    return *((float *)&inter);
}

static bfloat16 float32_to_bfloat16(float value)
{
    // cast as uint32_t, then bitshift 16 bits to the right, then cast as uint16_t, then cast as bfloat16
    uint32_t inter(uint32_t(*((uint32_t *)&value)) >> 16);
    return {
        (uint16_t)inter};
}

#endif

static std::ostream &operator<<(std::ostream &os, const bfloat16 &value)
{
    return os << bfloat16_to_float32(value);
}

#endif