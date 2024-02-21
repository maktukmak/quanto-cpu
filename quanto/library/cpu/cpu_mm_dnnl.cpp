#include <vector>
#include <torch/extension.h>
#include "mkl.h"


torch::Tensor& _int_mm_out_cpu(const torch::Tensor& self, const torch::Tensor& mat2, torch::Tensor& result) {

    // TORCH_CHECK(self.dim() == 2, "Expected self to be of dimension 2 but got ", self.dim());
    // TORCH_CHECK(mat2.dim() == 2, "Expected mat2 to be of dimension 2 but got ", mat2.dim());
    // TORCH_CHECK(self.size(0) > 16, "self.size(0) needs to be greater than 16, but got ", self.size(0));
    // TORCH_CHECK(self.size(1) > 0 && self.size(1) % 8 == 0, "self.size(1) needs to be greater than 0 and a multiple of 8, but got ", self.size(1));
    // TORCH_CHECK(self.size(1) == mat2.size(0), "self.size(1) needs to match mat2.size(0) but got ", self.size(1), " and ", mat2.size(0));
    // TORCH_CHECK(mat2.size(1) > 0 && mat2.size(1) % 8 == 0, "mat2.size(1) needs to be greater than 0 and a multiple of 8, but got ", mat2.size(1));
    // TORCH_CHECK(result.dtype() == at::kInt, "Expected result dtype to be of type kInt but got ", result.dtype());
    // TORCH_CHECK(result.size(0) == self.size(0), "Expected result.size(0) to be ", self.size(0), " but got ", result.size(0));
    // TORCH_CHECK(result.size(1) == mat2.size(1), "Expected result.size(1) to be ", mat2.size(1), " but got ", result.size(1));
    // TORCH_CHECK(result.dim() == 2, "Expected result to be of dimension 2 but got ", result.dim());
    // TORCH_CHECK(result.is_contiguous(), "Expected result to be contiguous.");

    MKL_INT         m=self.size(0), n=mat2.size(0), k=self.size(1);
    MKL_INT         lda=self.size(1), ldb=mat2.size(1), ldc=result.size(1);
    float           alpha=1.0f, beta=0.0f;

    const MKL_INT8 ao = 0;
    const MKL_INT8 bo = 0;
    MKL_INT32 co = 0;
    CBLAS_OFFSET    offsetc=CblasFixOffset;

    cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, offsetc,
                      m, n, k, alpha, self.data_ptr<int8_t>(), lda, ao, mat2.data_ptr<uint8_t>(), ldb, bo, beta, result.data_ptr<int32_t>(), ldc, &co);

    auto asd = result.data_ptr<int32_t>();

  return result;
}


torch::Tensor _int_mm_cpu(const torch::Tensor& self, const torch::Tensor& mat2) {
  torch::Tensor result = torch::empty({self.size(0), mat2.size(1)}, self.options().dtype(torch::kInt32));
  return _int_mm_out_cpu(self, mat2, result);
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("_int_mm", &_int_mm_cpu, "_int_mm");
// }