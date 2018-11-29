import tensorflow as tf
import sparse_reshape_gpu_py as sparse_reshape
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

indices = math_ops.cast([[0, 0], [1, 2]], dtype=dtypes.int64)
values = math_ops.cast([1, 2], dtype=dtypes.int64)

sp_input = tf.SparseTensor(indices=indices, values=values, dense_shape=[3, 4])
with tf.device('/device:GPU:0'):
  sp_reshaped = sparse_reshape.sparse_reshape_gpu(sp_input, [2,2,3])

config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
  sp_output = sess.run(sp_reshaped)
  print sp_output.indices
  print sp_output.values
  print sp_output.dense_shape
