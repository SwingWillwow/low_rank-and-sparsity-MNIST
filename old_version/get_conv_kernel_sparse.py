import tensorflow as tf
import numpy as np


def clean_s(var_list):
    ret_list = []
    for s in var_list:
        new_s = tf.reshape(s, [-1])
        size = np.prod(s.shape.as_list()) // 3
        values, indices = tf.nn.top_k(new_s, size)
        val, idx = tf.nn.top_k(indices, int(indices.shape[0]))
        values = tf.gather(values, idx)
        indices = tf.gather(indices, idx)
        values = tf.reverse(values, axis=[0])
        indices = tf.reverse(indices, axis=[0])
        indices = tf.cast(indices, tf.int32)
        add_one = tf.sparse_to_dense(sparse_indices=indices, output_shape=new_s.shape, sparse_values=values)
        add_one = tf.reshape(add_one, s.shape)
        s_zero = tf.zeros(s.shape)
        s_add = tf.add(s_zero, add_one)
        ret_list.append(tf.assign(s, s_add))
    return ret_list


# x1 = [[[[1, 2, 3],
#         [7, 62, 2],
#         [8, 49, 4]],
#        [[95, 44, 33],
#         [12, 55, 76],
#         [76, 43, 23]]]]
# x1 = tf.Variable(x1, dtype=tf.float32, name='x1')
# x2 = [[[[7, 9, 5],
#         [8, 55, 3],
#         [9, 7, 47]],
#        [[4, 6, 8],
#         [11, 56, 12],
#         [6, 7, 99]]]]
# x2 = tf.Variable(x2, dtype=tf.float32, name='x2')
x1 = [[1, 2, 3],
      [7, 6, 2],
      [8, 4, 4]]
x1 = tf.Variable(x1, dtype=tf.float32, name='x1')
x2 = [[7, 9, 5],
      [8, 5, 3],
      [9, 7, 4]]
x2 = tf.Variable(x2, dtype=tf.float32, name='x2')
list = [x1, x2]
ret_list = clean_s_conv(list)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(ret_list)
    for x in list:
        print(x.name)
        print(sess.run(x))