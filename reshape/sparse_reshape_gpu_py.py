import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

sparse_reshape_gpu_module = tf.load_op_library('./sparse_reshape_gpu.so')

def sparse_reshape_gpu(sp_input, shape):
  shape = math_ops.cast(shape, dtype=dtypes.int64)
  reshaped_ind = sparse_reshape_gpu_module.sparse_reshape_gpu(math_ops.cast(sp_input.indices, dtype=dtypes.int64), math_ops.cast(sp_input.dense_shape, dtype=dtypes.int64), shape)
  return sparse_tensor.SparseTensor(reshaped_ind, array_ops.identity(sp_input.values), shape)
