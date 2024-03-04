#include <vector>
#include <torch/extension.h>
#include "mkl.h"
#include "cpu_mm_mkl.cpp"


int main(int argc, char *argv[])
{
    int m = 2;
    int n = 2;
    int k = 2;

    torch::Tensor self = 127 + torch::zeros({m, k}, torch::kInt8);
    torch::Tensor mat2 = 127 + torch::zeros({k, n}, torch::kInt8);

    std::cout << self << std::endl;
    std::cout << mat2 << std::endl;5

    auto result = _int_mm_cpu(self, mat2);

    std::cout << result << std::endl;

}

