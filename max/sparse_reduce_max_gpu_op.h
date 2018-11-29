#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

//#ifndef SPARSE_REDUCE_MAX_GPU_H_
//#define SPARSE_REDUCE_MAX_GPU_H_

using namespace tensorflow;

template <typename Device, typename T>
struct SparseReduceMaxGpuFunctor {
    void operator()(const Device& d, const Tensor* input_indices, const Tensor* input_values, const Tensor* input_shape, Tensor* out_values, const int64 inputRank, const int32 reductionAxis, const int64 nnz);
};

//#endif SPARSE_REDUCE_MAX_GPU_H_
