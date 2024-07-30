#include "tensor.hpp"
#include <iostream>
#include <chrono>  
#include <immintrin.h>

void matvec(float* A, float* x, float* y, int m, int n){
    for(int i = 0; i < m; i++){
        y[i] = 0;
        for(int j = 0; j < n; j++){
            y[i] += A[i*n + j] * x[j];
        }
    }
}

void matvecavx256(float* A, float* x, float* y, int m, int n){
    for(int i = 0; i < m; i++){
        auto sum = _mm256_setzero_ps();
        for(int j = 0; j < n; j+= 8){
                    auto a = _mm256_loadu_ps(A + i*n + j);
            auto b = _mm256_loadu_ps(x + j);
            sum = _mm256_fmadd_ps(a, b, sum);
        }
        y[i] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];        
    }
}

void matvecavx512(float* A, float* x, float* y, int m, int n){
    for(int i = 0; i < m; i++){
        auto sum = _mm512_setzero_ps();
        for(int j = 0; j < n; j+= 16){
            auto a = _mm512_loadu_ps(A + i*n + j);
            auto b = _mm512_loadu_ps(x + j);
            sum = _mm512_fmadd_ps(a, b, sum);
        }
        y[i] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] + sum[8] + sum[9] + sum[10] + sum[11] + sum[12] + sum[13] + sum[14] + sum[15];
    }
}

void matvecavx512side(float* A, float* x, float* y, int m, int n){
    for(int i = 0; i < m; i+= 16){
        auto sum = _mm512_setzero_ps();
        for(int j = 0; j < n; j+= 1){

            auto b = _mm512_set1_ps(x[j]);
            auto a = _mm512_loadu_ps(A + j*n + i);
            sum = _mm512_fmadd_ps(a, b, sum);
        }
        _mm512_storeu_ps(y + i, sum);
    }
}

void matvecavx512dot(float* A, float* x, float* y, int m, int n){
    for(int i = 0; i < m; i+= 1){
        auto sum = _mm512_setzero_ps();
        for(int j = 0; j < n; j+= 32){
            auto a = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(A + i*n + j), _mm512_loadu_ps(A + i*n + j + 16));
            auto b = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + j), _mm512_loadu_ps(x + j + 16));
            sum = _mm512_dpbf16_ps(sum,a,b);

        }
        y[i] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] + sum[8] + sum[9] + sum[10] + sum[11] + sum[12] + sum[13] + sum[14] + sum[15];
    }
}

void matvecavx512dot(bfloat16* A, bfloat16* x, bfloat16* y, int m, int n){
    for(int i = 0; i < m; i+= 1){
        auto sum = _mm512_setzero_ps();
        for(int j = 0; j < n; j+= 32){

            auto a = (__m512bh )_mm512_load_si512((__m512i*)(A + i*n + j));

            auto b = (__m512bh )_mm512_load_si512((__m512i*)(x + j));
            
            sum = _mm512_dpbf16_ps(sum,a,b);

        }
        y[i] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] + sum[8] + sum[9] + sum[10] + sum[11] + sum[12] + sum[13] + sum[14] + sum[15];
    }
}




int main(){
    Tensor t1({5120, 5120});
    Tensor t2(Shape{5120});
    Tensor t4(Shape{5120, 5120}, kBFLOAT_16);
    Tensor t3(Shape{5120});
    Tensor t5(Shape{5120}, kBFLOAT_16);
    Tensor t6(Shape{5120}, kBFLOAT_16);
    t1 = 1;
    t2 = 2;
    t3 = 2;
    t4 = 2;
    t5 = 3;
    t6 = 4;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvec((float*)t1.data, (float*)t2.data, (float*)t3.data, 5120, 5120);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Tops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvecavx256((float*)t1.data, (float*)t2.data, (float*)t3.data, 5120, 5120);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Tops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvecavx512((float*)t1.data, (float*)t2.data, (float*)t3.data, 5120, 5120);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Ops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvecavx512side((float*)t1.data, (float*)t2.data, (float*)t3.data, 5120, 5120);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Ops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvecavx512dot((float*)t1.data, (float*)t2.data, (float*)t3.data, 5120, 5120);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Ops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++){
        matvecavx512dot((bfloat16*)t4.data, (bfloat16*)t5.data, (bfloat16*)t6.data, 5120, 5120);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Ops per second: " << 10*5120*5120/elapsed_seconds.count()*1e-12 << "\n";

    return 0;

}