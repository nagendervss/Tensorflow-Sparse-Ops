#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "sparse_reduce_max_gpu_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseReduceMaxGpu")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Input("input_shape_cpu: int64")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnknownShape);

template <typename Device, typename T>
class SparseReduceMaxGpuOp : public OpKernel{
    public:
        explicit SparseReduceMaxGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
        void Compute(OpKernelContext* ctx) override {
            const Tensor *indices_t, *values_t, *shape_t, *shape_t_cpu, *reduction_axes_t;
            OP_REQUIRES_OK(ctx, ctx->input("input_indices", &indices_t));
            OP_REQUIRES_OK(ctx, ctx->input("input_values", &values_t));
            OP_REQUIRES_OK(ctx, ctx->input("input_shape", &shape_t));
            OP_REQUIRES_OK(ctx, ctx->input("reduction_axes", &reduction_axes_t));
            OP_REQUIRES_OK(ctx, ctx->input("input_shape_cpu", &shape_t_cpu));
            
            //Compute the output shape. Currently computing only for one input axis.
            
            TensorShape output_shape;
            const int64 inputRank = shape_t->NumElements();
            
            int32 reductionAxis = reduction_axes_t->flat<int32>().data()[0];
            
            auto inputShape = shape_t_cpu->vec<int64>().data();
            for(int i=0; i<inputRank; i++){
                if(i == reductionAxis){
                    continue;
                }
                else{
                    int64 currentDim = inputShape[i];
                    output_shape.AddDim(currentDim);
                }
            }

            Tensor *out_values;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out_values));
            
            const int64 nnz = indices_t->shape().dim_size(0);
            
            SparseReduceMaxGpuFunctor<Device, T>()(ctx->eigen_device<Device>(), indices_t, values_t, shape_t, out_values, inputRank, reductionAxis, nnz);
        }
};

#ifdef GOOGLE_CUDA
//#define REGISTER_GPU(T) extern template SparseReduceMaxGpuFunctor<GPUDevice, T>; REGISTER_KERNEL_BUILDER(Name("SparseReduceMaxGpu").Device(DEVICE_GPU).TypeConstraint<T>("T"));
#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(Name("SparseReduceMaxGpu").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("reduction_axes").HostMemory("input_shape_cpu"), SparseReduceMaxGpuOp<GPUDevice, T>);

REGISTER_GPU(float);
#endif
