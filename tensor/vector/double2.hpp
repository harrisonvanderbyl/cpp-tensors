
#define pows(x) (pow(float(x), 2))
#include "immintrin.h"
struct double2
{
    double x;
    double y;

    double2(int x, int y)
    {
        this->x = x;
        this->y = y;
    };

    double2()
    {
        this->x = 0;
        this->y = 0;
    };

    double2 operator+(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator-(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator/(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator*(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator*(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator/(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_loadu_pd((double *)this),_mm_set1_pd(other)));
        return out;
    };

    double2 operator+(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator-(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator+=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator-=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator/=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator*=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator*=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator/=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator+=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator-=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };


    double2 copy()
    {
        double2 out;
        out.x = x;
        out.y = y;
        return out;
    };

    static double2 random()
    {
        double2 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    double2 operator=(double2 &other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    double2 operator=(double other)
    {
        x = other;
        y = other;
        return *this;
    };

    double2 operator=(double2 other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    Tensor toTensor()
    {
        Tensor out({2}, kFLOAT_64);
        out[0] = x;
        out[1] = y;
        return out;
    };
};
