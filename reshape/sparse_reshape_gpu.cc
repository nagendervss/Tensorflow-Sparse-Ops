#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "sparse_reshape_gpu_op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using namespace std;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseReshapeGpu")
    .Input("input_indices: int64")
    .Input("input_shape: int64")
    .Input("new_shape: int64")
    .Output("output_indices: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle new_shape;
      
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &new_shape));

      c->set_output(0, c->Matrix(c->Dim(indices,0), c->Dim(new_shape,0)));
      return Status::OK();
    });

//template <typename T>
template <>
struct SparseReshapeGpuFunctor<CPUDevice> {
    void operator()(const CPUDevice& d, const Tensor& input_indices, const Tensor& input_shape, const Tensor& new_shape, Tensor* output_indices, const int64 nnz, const int64 inputRank, const int64 outputRank) {
        auto input_ind = input_indices.matrix<int64>();
        auto output_ind = output_indices->matrix<int64>();
        auto input_sh = input_shape.vec<int64>();
        auto new_sh = new_shape.vec<int64>();

        for (int i = 0; i < nnz; ++i) {
            int64 id = 0;
            for (int j = 0; j < inputRank; ++j) {
                int64 currentRankStride = 1;
                for(int k=j+1; k<inputRank; k++){
                    currentRankStride *= input_sh(k);
                }
                //id += input_ind(i, j) * input_sh(j);
                id += input_ind(i, j) * currentRankStride;
            }
            for (int j = 0; j < outputRank; ++j) {
                int64 currentRankStride = 1;
                for(int k=j+1; k<outputRank; k++){
                    currentRankStride *= new_sh(k);
                }
                output_ind(i, j) = id / currentRankStride;
                id %= currentRankStride;
            }
        }
    }
};

template<typename Device>
class SparseReshapeGpuOp : public OpKernel {
    public:
        explicit SparseReshapeGpuOp(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {
            const Tensor& input_indices = context->input(0);
            const Tensor& input_shape = context->input(1);
            const Tensor& new_shape = context->input(2);

            Tensor* output_indices = NULL;
            const int64 nnz = input_indices.shape().dim_size(0);
            const int64 outputRank = new_shape.NumElements();
            const int64 inputRank = input_shape.NumElements();
            
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nnz, outputRank}), &output_indices));
            
            SparseReshapeGpuFunctor<Device>()(context->eigen_device<Device>(), input_indices, input_shape, new_shape, output_indices, nnz, inputRank, outputRank);
        }
};

/*#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("SparseReshapeGpu").Device(DEVICE_CPU).TypeConstraint<T>("T"), SparseReshapeGpuOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);
REGISTER_CPU(int64);*/

REGISTER_KERNEL_BUILDER(Name("SparseReshapeGpu").Device(DEVICE_CPU), SparseReshapeGpuOp<CPUDevice>);

//#ifdef GOOGLE_CUDA
/*#define REGISTER_GPU(T) extern template SparseReshapeGPUFunctor<GPUDevice, T>; REGISTER_KERNEL_BUILDER(Name("SparseReshapeGpu").Device(DEVICE_GPU).TypeConstraint<T>("T"), SparseReshapeGpuOp<GPUDevice, T>);

REGISTER_GPU(float);
REGISTER_GPU(int32);
REGISTER_GPU(int64);*/

#define REGISTER_GPU() REGISTER_KERNEL_BUILDER(Name("SparseReshapeGpu").Device(DEVICE_GPU), SparseReshapeGpuOp<GPUDevice>);

REGISTER_GPU();
//#endif
