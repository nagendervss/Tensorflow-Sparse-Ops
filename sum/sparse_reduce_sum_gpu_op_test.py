import sys
import tensorflow as tf
import sparse_reduce_sum_gpu_py as sparse_reduce
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import numpy as np

sys.settrace

indices = math_ops.cast([[0,0,0], [0,0,2], [0,1,1], [0,1,2], [0,2,2], [1,0,0], [1,0,1], [1,1,2], [1,2,0], [1,2,1], [2,0,1], [2,0,2], [2,1,0], [2,2,2]], dtype=dtypes.int64)

values = math_ops.cast([3,1,2,6,1,1.5,2.5,3.5,2.5,5.5,0.5,0.75,0.35,0.1], dtype=dtypes.float32)

sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=[3, 3, 3])

sp_reduced = sparse_reduce.sparse_reduce_sum_gpu(sp_input, axis=0, keepdims=True)
sp_reduced_cpu = tf.sparse_reduce_sum(sp_input, axis=0, keep_dims=True)

config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
  output = sess.run(sp_reduced)
  outputCPU = sess.run(sp_reduced_cpu)
  print "output GPU",output
  print "output CPU",outputCPU
