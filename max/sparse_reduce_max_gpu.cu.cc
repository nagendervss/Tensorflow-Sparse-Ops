#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_max_gpu_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

using GPUDevice = Eigen::GpuDevice;

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template <typename T>
__global__ void SparseReduceMaxGpuCudaKernel(const int64* input_indices, const T* input_values, const int64* input_shape, T* output_values, const int64 inputRank, const int32 reductionAxis, const int64 nnz){
    int64 threadId = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadId < nnz){
        int64 inputId = threadId*inputRank;
        int64 outputId = 0;
        for(int j=0; j<inputRank; j++){
            if(j==reductionAxis) continue;
            int64 currentRankStride = 1;
            for(int k=j+1; k<inputRank; k++){
                if(k == reductionAxis) continue;
                currentRankStride *= input_shape[k];
            }
            outputId += input_indices[inputId+j] * currentRankStride;
        }
        T inputVal = input_values[threadId];
        T oldVal = atomicMax(output_values + outputId, inputVal);
    }
}

template <typename GPUDevice, typename T>
void SparseReduceMaxGpuFunctor<GPUDevice, T>::operator()(const GPUDevice& d, const Tensor* input_indices, const Tensor* input_values, const Tensor* input_shape, Tensor* out_values, const int64 inputRank, const int32 reductionAxis, const int64 nnz){

    int threads_per_block = 1024;
    int block_count = nnz/threads_per_block + 1;
    auto input_ind = input_indices->matrix<int64>();
    auto input_val = input_values->vec<T>();
    auto input_sh = input_shape->vec<int64>();
    auto out_flat = out_values->flat<T>();
    
    SparseReduceMaxGpuCudaKernel<T><<<block_count, threads_per_block, 0, d.stream()>>>(input_ind.data(), input_val.data(), input_sh.data(), out_flat.data(), inputRank, reductionAxis, nnz);
}

template struct SparseReduceMaxGpuFunctor<GPUDevice, float>;

#endif
