import tensorflow as tf
import numpy as np
import csv
import os
import re
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def save_weights(saver, sess, epoch_i):
    path = "./params/"
    if not os.path.exists(path):
        os.makedirs(path)
    saver.save(sess, path + "epoch" + str(epoch_i))

def load_weights_with_confirm(saver, sess):
    path = "./params"
    if not os.path.exists(path):
        os.makedirs(path)

    params_files_name = [obj for obj in os.listdir(path) if os.path.isfile(path + "/" + obj)]
    ckpt_files = [file_name for file_name in params_files_name if ".data" in file_name]

    if ckpt_files:
        params_files = [[int(re.findall("\d+", file_name)[0]), file_name] for file_name in ckpt_files]
        params_files.sort()

        epochs = [i[0] for i in params_files]

        print("The parameter file  was found.")
        s = input("Do you wanna use it? (y/n) : ")
        if s == "y" or s == "Y":
            print("What epochs number do you want to start with?")
            for i in params_files:
                print("epoch ", i[0])
            epoch_input = int(input("Please enter the number : "))

            if epoch_input in epochs:
                params_file_name = params_files[epochs.index(epoch_input)][1]
                saver.restore(sess, path + "/epoch" + str(epoch_input))
                return epoch_input + 1
            else:
                print("The number you entered is invalid.")

    return 0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc(inputs, input_shape, output_shape, keep_prob, activation_f = tf.nn.relu):
    W_fc = weight_variable([input_shape, output_shape])
    b_fc = bias_variable([output_shape])

    if activation_f == None:
        h_fc = tf.matmul(inputs, W_fc) + b_fc
    else:
        h_fc = activation_f(tf.matmul(inputs, W_fc) + b_fc)

    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    return h_fc_drop

def conv2d(x, W, strides = [1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides, padding='SAME')

def max_pool_2(x, strides = [2, 2]):
    return tf.layers.max_pooling2d(x, (3, 3), strides)
    #return tf.layers.max_pooling1d(x, 3, 1)
    #return tf.nn.max_pool(x, ksize=[1, 2, 1], strides=[1, 2, 1], padding='SAME')

def bn_relu_weight(inputs, shape):
    bn = tf.layers.batch_normalization(inputs, True)
    activ = tf.nn.relu(bn)

    W_conv = weight_variable(shape)
    h_conv = conv2d(activ, W_conv)

    return h_conv

def block(inputs, input_channel, output_channel, bottleneck = True):
    if bottleneck:
        h1 = bn_relu_weight(inputs, (1, 1, input_channel, input_channel))
        h2 = bn_relu_weight(h1, (3, 3, input_channel, input_channel))
        h3 = bn_relu_weight(h2, (1, 1, input_channel, output_channel))

        if input_channel < output_channel:
            return h3 + tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, output_channel - input_channel]])

        return h3 + inputs
    else:
        bn1 = tf.layers.batch_normalization(inputs, True)

        W_conv1 = weight_variable((3, input_channel, input_channel))
        h_conv1 = conv2d(bn1, W_conv1)

        bn2 = tf.layers.batch_normalization(h_conv1, True)

        activ = tf.nn.relu(bn2)

        W_conv2 = weight_variable((3, input_channel, output_channel))
        h_conv2 = conv2d(activ, W_conv2)

        bn3 = tf.layers.batch_normalization(h_conv2, True)

        if input_channel < output_channel:
            return bn3 + tf.pad(inputs, [[0, 0], [0, 0], [0, output_channel - input_channel]])

        return bn3 + inputs

def blocks(inputs, depth, input_channel, output_channel):
    if depth <= 1:
        return block(inputs, input_channel, output_channel)

    h = block(inputs, input_channel, input_channel)

    for _ in range(depth - 2): 
        h = block(h, input_channel, input_channel)
    
    h = block(h, input_channel, output_channel)
    return h

def gap(inputs, channel_size):
    return tf.reduce_mean(inputs, [1, 2])
    #return tf.layers.average_pooling2d(inputs, [7, 7], 1)

width = 28
height = 28
channel = 1
output_shape = 10

batch_size = 128
n_epochs = 0

if __name__ == "__main__":
    sess = tf.InteractiveSession()

    # 50
    input_x = tf.placeholder(tf.float32, shape=[None, width, height, channel])
    input_y = tf.placeholder(tf.float32, shape=[None, output_shape])

    keep_prob = tf.placeholder("float")

    W_conv1 = weight_variable((7, 7, channel, 32))
    h1 = conv2d(input_x, W_conv1, strides = [1, 2, 2, 1])
    h2 = max_pool_2(h1)

    h3 = blocks(h2, 3, 32, 64)
    h4 = max_pool_2(h3)

    h5 = blocks(h4, 8, 64, 128)
    #h6 = max_pool_2(h5)

    h7 = blocks(h5, 36, 128, 256)
    #h8 = max_pool_2(h7)

    h9 = blocks(h7, 3, 256, 512)
    #h10 = max_pool_2(h9)

    h11 = gap(h9, 512)
    flatten = tf.reshape(h11, [-1, 512])

    h12 = fc(flatten, 512, output_shape, keep_prob, None)

    #y = tf.nn.softmax(h12)
    y  = h12

    #loss = -tf.reduce_sum(input_y * tf.log(adj(0.001, y)) + (1 - input_y) * tf.log(adj(0.001, 1 - y)))
    #loss = tf.reduce_mean(tf.abs(input_y - y))
    #loss = -tf.reduce_sum(input_y * tf.log(y))
    y_t = tf.argmax(input_y, axis=1)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                 labels=y_t,
                 logits=y
             )
    loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    start_epoch_i = load_weights_with_confirm(saver, sess)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for i in range(n_epochs):
        if i % 10 == 0:
            batch = mnist.train.next_batch(batch_size)

            train_loss.append(
                [
                    start_epoch_i + i,
                    loss.eval(feed_dict={
                        input_x: np.reshape(batch[0], [-1, 28, 28, 1]), input_y: batch[1], keep_prob: 1.0
                    })
                ]
            )
            train_acc.append(
                [
                    start_epoch_i + i,
                    accuracy.eval(feed_dict={
                        input_x: np.reshape(batch[0], [-1, 28, 28, 1]), input_y: batch[1], keep_prob: 1.0
                    })
                ]
            )

            test_batch = mnist.test.next_batch(batch_size)
            test_loss.append(
                [
                    start_epoch_i + i,
                    loss.eval(feed_dict={
                        input_x: np.reshape(test_batch[0], [-1, 28, 28, 1]), input_y: test_batch[1], keep_prob: 1.0
                    })
                ]
            )
            test_acc.append(
                [
                    start_epoch_i + i,
                    accuracy.eval(feed_dict={
                        input_x: np.reshape(test_batch[0], [-1, 28, 28, 1]), input_y: test_batch[1], keep_prob: 1.0
                    })
                ]
            )

            save_weights(saver, sess, start_epoch_i + i)

            print(str(start_epoch_i + i) + "epoch train loss : " + str(train_loss[-1][1]))
            print(str(start_epoch_i + i) + "epoch test loss : " + str(test_loss[-1][1]))
        else:
            print(i)

        for _ in range(60000 // batch_size):
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={
                input_x: np.reshape(batch[0], [-1, 28, 28, 1]), input_y: batch[1], keep_prob: 0.5
            })
        
    test_x, test_y = mnist.test.next_batch(10000)

    print("test accuracy : " + str(accuracy.eval(feed_dict={
            input_x: np.reshape(test_x, [-1, 28, 28, 1]), input_y: test_y, keep_prob: 1.0
        }))
    )

    def write_log(file_name, log):
        with open(file_name, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(log)

    write_log("train_loss.csv", train_loss)
    write_log("train_acc.csv", train_acc)
    write_log("test_loss.csv", test_loss)
    write_log("test_acc.csv", test_acc)
