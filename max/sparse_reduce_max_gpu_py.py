import tensorflow as tf
#from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

sparse_reduce_max_gpu_module = tf.load_op_library('./sparse_reduce_max_gpu.so')

def sparse_reduce_max_gpu(sp_input, axis):
  return sparse_reduce_max_gpu_module.sparse_reduce_max_gpu(sp_input.indices, sp_input.values, math_ops.cast(sp_input.dense_shape, dtype=dtypes.int64), axis, math_ops.cast(sp_input.dense_shape, dtype=dtypes.int64))
