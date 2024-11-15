
#define pows(x) (pow(float(x), 2))
#include "immintrin.h"

struct float2 {
    float x;
    float y;

    float2(int x, int y)
    {
        this->x = x;
        this->y = y;
    };

    float2(float x, float y)
    {
        this->x = x;
        this->y = y;
    };

    float2()
    {
        this->x = 0;
        this->y = 0;
    };

    
};

struct float4
{
    float x;
    float y;
    float z;
    float w;

    float4(int x, int y, int z, int w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    float4(float x, float y, float z, float w)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    float4()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };

    float2& xy()
    {
        return *(float2 *)this;
    };

    float2& yz()
    {
        return *(float2 *)&y;
    };

    float2& zw()
    {
        return *(float2 *)&z;
    };

    float4 operator+(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_add_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator-(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_sub_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator/(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_div_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator*(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_mul_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator*(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_mul_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator/(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_div_ps(_mm_loadu_ps((float *)this),_mm_set1_ps(other)));
        return out;
    };

    float4 operator+(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_add_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator-(float other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_sub_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator+=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_add_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator-=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_sub_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator/=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_div_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator*=(float4 &other)
    {
        _mm_storeu_ps((float *)this, _mm_mul_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator*=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_mul_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator/=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_div_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator+=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_add_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };

    float4 operator-=(float other)
    {
        _mm_storeu_ps((float *)this, _mm_sub_ps(_mm_set1_ps(other), _mm_loadu_ps((float *)this)));
        return *this;
    };


    float4 copy()
    {
        float4 out;
        out.x = x;
        out.y = y;
        out.z = z;
        out.w = w;
        return out;
    };

    static float4 random()
    {
        float4 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        out.z = (rand() % 10000) / 10000.0 - 0.5;
        out.w = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    float4 operator=(float4 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };

    float4 operator=(float other)
    {
        x = other;
        y = other;
        z = other;
        w = other;
        return *this;
    };

    float4 operator=(float4 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    };

    Tensor toTensor()
    {
        Tensor out({4}, kFLOAT_32);
        out[0] = x;
        out[1] = y;
        out[2] = z;
        out[3] = w;
        return out;
    };

    float& operator [](int index)
    {
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        default:
            throw "Index out of range";
        }
    };
};
