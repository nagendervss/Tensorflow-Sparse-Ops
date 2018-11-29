#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

#ifndef SPARSE_RESHAPE_GPU_H_
#define SPARSE_RESHAPE_GPU_H_

using namespace tensorflow;

template <typename Device>
struct SparseReshapeGpuFunctor {
  void operator()(const Device& d, const Tensor& input_ind, const Tensor& input_sh, const Tensor& new_sh, Tensor* output_ind, const int64 nnz, const int64 inputRank, const int64 outputRank);
};

/*#if GOOGLE_CUDA
template <typename Eigen::GpuDevice>
struct SparseReshapeGpuFunctor {
  void operator()(const Eigen::GpuDevice& d, const Tensor& input_ind, const Tensor& input_sh, const Tensor& new_sh, Tensor* output_ind, const int64 nnz, const int64 inputRank, const int64 outputRank);
};
#endif*/

#endif SPARSE_RESHAPE_GPU_H_
