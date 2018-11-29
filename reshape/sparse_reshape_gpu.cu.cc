#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reshape_gpu_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

//template <typename T>
//__global__ void SparseReshapeGpuCudaKernel(const Tensor& input_indices, const Tensor& input_shape, const Tensor& new_shape, Tensor* output_indices, const int64 nnz, const int64 inputRank, const int64 outputRank){
__global__ void SparseReshapeGpuCudaKernel(const int64* input_indices, const int64* input_shape, const int64* new_shape, int64* output_indices, const int64 nnz, const int64 inputRank, const int64 outputRank){
    int threadId = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadId < nnz){
    /*    auto input_ind = input_indices.matrix<int64>();
        auto output_ind = output_indices->matrix<int64>();
        auto input_sh = input_shape.vec<int64>();
        auto new_sh = new_shape.vec<int64>();*/
        int64 inputId = threadId*inputRank;
        int64 outputId = threadId*outputRank;
        int64 id = 0;
        for (int j = 0; j < inputRank; ++j) {
            //id += input_ind(threadId, j) * input_sh(j);
            int64 currentRankStride = 1;
            for(int k=j+1; k<inputRank; k++){
                currentRankStride *= input_shape[k];
            }
            id += input_indices[inputId+j] * currentRankStride;
        }
        for (int j = 0; j < outputRank; ++j) {
            int64 currentRankStride = 1;
            for(int k=j+1; k<outputRank; k++){
                currentRankStride *= new_shape[k];
            }
            output_indices[outputId + j] = id / currentRankStride;
            id %= currentRankStride;
        }
    }
}

//template <typename T>
template <>
void SparseReshapeGpuFunctor<GPUDevice>::operator()(const GPUDevice& d, const Tensor& input_indices, const Tensor& input_shape, const Tensor& new_shape, Tensor* output_indices, const int64 nnz, const int64 inputRank, const int64 outputRank){
    int thread_per_block = 1024;
    auto input_ind = input_indices.matrix<int64>();
    auto output_ind = output_indices->matrix<int64>();
    auto input_sh = input_shape.vec<int64>();
    auto new_sh = new_shape.vec<int64>();

    int block_count = nnz/thread_per_block + 1;
    SparseReshapeGpuCudaKernel<<<block_count, thread_per_block, 0, d.stream()>>>(input_ind.data(), input_sh.data(), new_sh.data(), output_ind.data(), nnz, inputRank, outputRank);
}

/*template struct SparseReshapeGpuFunctor<GPUDevice, float>;
template struct SparseReshapeGpuFunctor<GPUDevice, int32>;
template struct SparseReshapeGpuFunctor<GPUDevice, int64>;*/

template struct SparseReshapeGpuFunctor<GPUDevice>;

#endif
