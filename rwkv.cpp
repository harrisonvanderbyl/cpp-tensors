#include "stdio.h"
#include "tensor.hpp"
#include "file_loaders/safetensors.hpp"
// #include "tokenizers/world.hpp"
#include "ops/ops.h"

int main(){
 

    Tensor b = {{1024,1024}};
    Tensor cc = 1.0f;
    // for (size_t i = 0; i < b.shape[0]; i++)
    // {
    //     for (size_t j = 0; j < b.shape[1]; j++)
    //     {
            // b[i][j] = i*b.shape[1] + j;
    b = 1.0f;
    //     }
    // }
    // std::cout << b << "\n";
    // std::cout << b[{0,2}] << "\n";
    // std::cout << b[{2,4}] << "\n";
    // std::cout << c;
    auto xx = b * -1.0f;
    std::cout <<xx<< "\n";

    
}