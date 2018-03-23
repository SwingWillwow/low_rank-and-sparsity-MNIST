import tensorflow as tf
import numpy as np
import time
import datetime


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name+'_weight')


def bias_variables(shape, name):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name+'_bias')


def ABS_fc_layer(input, shape, name, r):
    m, n = shape
    W_inner_fc1 = weight_variables([m, r], name=name+'_inner_fc1')
    tf.add_to_collection('train_phase', W_inner_fc1)
    W_inner_fc2 = weight_variables([r, n], name=name+'_inner_fc2')
    tf.add_to_collection('train_phase', W_inner_fc2)
    S_inner_layer = tf.get_variable(name=name+'_inner_sparse',
                                    shape=[m, n], dtype=tf.float32, initializer=tf.zeros_initializer)
    tf.add_to_collection('refine_phase', S_inner_layer)
    tf.add_to_collection('clean_phase', S_inner_layer)
    b_inner_layer = bias_variables([n], name=name+'_inner_layer')
    tf.add_to_collection('train_phase', b_inner_layer)
    tf.add_to_collection('refine_phase', b_inner_layer)
    h_inner1 = tf.matmul(input, W_inner_fc1)
    h_inner2 = tf.matmul(h_inner1, W_inner_fc2)
    h_sparse = tf.matmul(input, S_inner_layer)
    h_final = tf.nn.relu(tf.add(h_sparse, h_inner2)+b_inner_layer)
    return h_final


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1],
                        padding='SAME')


def low_rank_conv2d(x, shape, r, name):
    h, w, in_chanel, out_chanel = shape
    W_inner_conv1 = weight_variables([h, w, in_chanel, r], name=name+'_inner_conv1')
    tf.add_to_collection('train_phase', W_inner_conv1)
    W_inner_conv2 = weight_variables([1, 1, r, out_chanel], name=name+'_inner_conv2')
    tf.add_to_collection('train_phase', W_inner_conv2)
    b_inner = bias_variables([out_chanel], name=name+'_inner_bias')
    tf.add_to_collection('train_phase', b_inner)
    h_inner1 = tf.nn.conv2d(input=x, filter=W_inner_conv1, strides=[1, 1, 1, 1],
                            padding='SAME', name=name+'_h_inner1')
    h_inner2 = tf.nn.conv2d(input=h_inner1, filter=W_inner_conv2, strides=[1, 1, 1, 1],
                            padding='SAME', name=name+'_h_inner2')
    h_inner2 = tf.nn.relu(h_inner2+b_inner)
    return h_inner2


def clean_s(var_list):
    ret_list = []
    for s in var_list:
        m, n = s.shape
        m = int(m)
        n = int(n)
        new_s = tf.reshape(s, [-1])
        size = m*n // 5
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


def max_pooling_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def load_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data, train_labels, eval_data, eval_labels


# define computation graph
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
y_ = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='labels')
# first conv layer
x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])

W_conv1 = weight_variables([5, 5, 1, 32], name='conv1')
tf.add_to_collection('train_phase', W_conv1)
b_conv1 = bias_variables([32], name='conv1')
tf.add_to_collection('train_phase', b_conv1)
h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1)

# h_conv1 = low_rank_conv2d(x_reshape, [5, 5, 1, 32], 10, name='conv1')

# first pooling

h_pool1 = max_pooling_2x2(h_conv1)

# second conv layer

# W_conv2 = weight_variables([5, 5, 32, 64], name='conv2')
# b_conv2 = bias_variables([64], name='conv2')
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_conv2 = low_rank_conv2d(h_pool1, [5, 5, 32, 64], 10, name='conv2')

# second pooling layer

h_pool2 = max_pooling_2x2(h_conv2)

# flatten the pooling result for the dense layer

flatten_pooling = tf.reshape(h_pool2, [-1, 7*7*64])

# fully-connected layer 1
# W_fc1 = weight_variables([7*7*64, 1024], name='fc1')
# b_fc1 = bias_variables([1024], name='fc1')
#
# h_fc1 = tf.nn.relu(tf.matmul(flatten_pooling, W_fc1) + b_fc1)
h_fc1 = ABS_fc_layer(flatten_pooling, [7*7*64, 1024], name='fc1', r=5)

# dropout

keep_rate = tf.placeholder(tf.float32, shape=None, name='drop_rate')
h_fc1_after_drop = tf.nn.dropout(h_fc1, keep_rate)

# fully-connected layer 2 (output layer)

# W_fc2 = weight_variables([1024, 10], name='fc2')
# b_fc2 = bias_variables([10], name='fc2')
# h_fc2 = tf.matmul(h_fc1_after_drop, W_fc2) + b_fc2
h_fc2 = ABS_fc_layer(h_fc1_after_drop, [1024, 10], name='fc2', r=5)

# use soft-max to get probability
softmax_out = tf.nn.softmax(h_fc2)

# get loss(cross_entropy)
cross_entropy = -tf.reduce_sum(tf.cast(y_, tf.float32)*tf.log(softmax_out+1e-8))

# train
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_list = tf.get_collection('train_phase')
refine_list = tf.get_collection('refine_phase')
clean_list = tf.get_collection('clean_phase')
clean_step = clean_s(clean_list)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=train_list)
refine_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, var_list=refine_list)
correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(h_fc2, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)

#import data

train_data, train_labels, eval_data, eval_labels = load_data()
one_hot_train_labels = tf.one_hot(indices=train_labels, depth=10)
one_hot_eval_labels = tf.one_hot(indices=eval_labels, depth=10)
train_data_set = tf.data.Dataset().from_tensor_slices((train_data, one_hot_train_labels))
train_data_set = train_data_set.repeat().batch(100)
epoch_size = int(len(train_data)/100)
eval_data_set = tf.data.Dataset().from_tensors((eval_data, one_hot_eval_labels))
train_iterator = train_data_set.make_initializable_iterator()
next_train_data = train_iterator.get_next()
eval_iterator = eval_data_set.make_initializable_iterator()
next_eval_data = eval_iterator.get_next()
# initialize
with tf.Session() as sess:
    sess.run(train_iterator.initializer)
    sess.run(eval_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    start_time = time.time()
    # training phase
    for i in range(50 * epoch_size):
        train_batch = sess.run(next_train_data)
        sess.run([train_step], feed_dict={x: train_batch[0],
                                          y_: train_batch[1],
                                          keep_rate: 0.5})
        if i % 100 == 0:
            train_accuracy, current_loss = sess.run([accuracy, cross_entropy],
                                                    feed_dict={x: train_batch[0],
                                                               y_: train_batch[1],
                                                               keep_rate: 1.0})
            print("accuracy: ", train_accuracy, ", current_loss", current_loss, " ,step:", i)
    start_sparse_time = time.time()
    # refine phase
    for i in range(20 * epoch_size):
        train_batch = sess.run(next_train_data)
        sess.run([refine_step], feed_dict={x: train_batch[0],
                                           y_: train_batch[1],
                                           keep_rate: 0.5})
        sess.run(clean_step)
        if i % 100 == 0:
            end_sparse_time = time.time()
            train_accuracy, current_loss = sess.run([accuracy, cross_entropy],
                                                    feed_dict={x: train_batch[0],
                                                               y_: train_batch[1],
                                                               keep_rate: 1.0})
            print("this iteration cost:", str(datetime.timedelta(seconds=(end_sparse_time - start_sparse_time))))
            print("accuracy: ", train_accuracy, ", current_loss", current_loss, " ,step:", i)
            start_sparse_time = time.time()
    eval_batch = sess.run(next_eval_data)
    end_time = time.time()
    total_time = end_time - start_time
    total_time = str(datetime.timedelta(seconds=total_time))
    final_accuracy, final_loss = sess.run([accuracy, cross_entropy],
                                          feed_dict={x: eval_batch[0],
                                                     y_: eval_batch[1],
                                                     keep_rate: 1.0})
    print("training finshed!!!!!!!")
    print("accuracy in eval data: ", final_accuracy, ", loss in eval_data: ", final_loss)
    print("total cost ", total_time, " to train.")
    for v in tf.trainable_variables():
        print(v.name)
        print(v.shape)
    parameter_num = 0
    for v in tf.trainable_variables():
        if str.find(v.name, 'sparse') != -1:
            parameter_num += np.prod(v.get_shape().as_list())//5
        else:
            parameter_num += np.prod(v.get_shape().as_list())
    tmp_s = sess.run(clean_list[1])
    count = 0
    for i in range(1024):
        for j in range(10):
            if tmp_s[i][j] != 0:
                count += 1
    print(count)
    # parameter_num = [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
    # parameter_num = np.sum(parameter_num)
    print("total parameter number:", parameter_num)
    # saver = tf.train.Saver()
    # saver.save(sess, 'model/ABS/train_to_get_sparse.ckpt')


