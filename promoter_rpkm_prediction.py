#!/user/bin/env python
'''
promoter_rpkm_prediction.py: Using neural networks to predict expression level
                             of a gene based on it's promoter sequence
Authorship information:
    __author__ = "Mars Huang"
    __email__ = "marshuang80@gmai.com:
    __status__ = "complete"
'''
from data_generator import *
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import tensorflow as tf

# Hypterparameters
promoter_len = 500
num_filters = 32
filter_size = 8
epoche = 2000
alpha = 1e-3
classes = 1
accs,test_accs = [],[]

# Placeholders for neural network
x_input = tf.placeholder(tf.float32,[None, promoter_len,4])
y_input = tf.placeholder(tf.float32,[None, classes])


print("building convolutional neural network...")
# First convolutional neural network layer
with tf.name_scope("con%s"%(str(filter_size))):
    # Expand dimentions for x-input
    x_in = tf.expand_dims(x_input,-1)

    # Define weights and bais for filters
    w = tf.Variable(tf.truncated_normal([filter_size,4,1, num_filters],mean = 0,stddev = 0.1))
    b = tf.Variable(tf.constant(0.1,shape=[num_filters]))

    # Convolution layer
    conv = tf.nn.conv2d(x_in, w, strides = [1,1,1,1], padding = "SAME")

    # Activation:
    out = tf.nn.relu(tf.nn.bias_add(conv,b))

    #max pooling
    pool = tf.nn.max_pool(out,ksize = [1,8,4, 1], strides = [1,8,1,1], padding = 'VALID')


# Transpose pooling layer
pool = tf.transpose(pool,[0,1,3,2])


# Second convolutional neural network layer
with tf.name_scope("con%s"%(str(2))):
    # Defile weights for 2nd CNN
    w1 = tf.Variable(tf.truncated_normal([4,32,1, 32],mean = 0,stddev = 0.1))
    b1 = tf.Variable(tf.constant(0.1,shape=[32]))

    # Second CNN
    conv1 = tf.nn.conv2d(pool, w1, strides = [1,1,1,1], padding = "SAME")

    # Activation
    out1 = tf.nn.relu(tf.nn.bias_add(conv1,b1))

    #max pooling
    pool1 = tf.nn.max_pool(out1,ksize = [1,2,32,1], strides = [1,2,1,1], padding = 'VALID')


# Flatten tensor to feed in fully connected layer
pool_flatten = tf.reshape(pool1, [-1, 31*32])


# Dropout for regulation and prevent overfitting
with tf.name_scope("dropout"):
    drop_layer = tf.nn.dropout(pool_flatten,0.8)


# Fully connected layer
with tf.name_scope("fc1"):
    w2 = tf.Variable(tf.truncated_normal([31*32,256],stddev = 0.5))
    b2 = tf.Variable(tf.constant(0.0,shape = [256]))
    fc1 = tf.nn.relu(tf.matmul(drop_layer,w2)+b2)


# Second fully connected layer to output prediction
with tf.name_scope("output"):
    W = tf.get_variable("W",shape = [256,1],initializer=tf.contrib.layers.xavier_initializer())
    B = tf.Variable(tf.constant(0.0,shape=[1]), name = 'b')
    output = tf.matmul(fc1,W)+B


# Optimize neural network with Adam optimizer
with tf.name_scope("loss"):
    losses = tf.reduce_mean(tf.reduce_sum(tf.square(output-y_input)))
    train = tf.train.AdamOptimizer(alpha).minimize(losses)


# Define tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Start training
with sess.as_default():
    print("start training...")

    # Iterate training epoches
    for i in range(epoche):
        # Gernate training batch
        x_batch, y_batch = batch(l,m,h,1024)
        x_batch, y_batch = shuffle_data(x_batch,y_batch)

        # Generate validation batch
        x_valid,y_valid = batch(l,m,h,128)
        x_valid,y_valid = shuffle_data(x_valid,y_valid)

        # Input dictionaries
        feed_dict = {
                x_input: x_batch,
                y_input: y_batch,
            }

        feed_dict_valid = {
                x_input: x_valid,
                y_input: y_valid,
            }

        # Run training and get MSE
        _,acc = sess.run([train,losses],feed_dict = feed_dict)
        valid_acc = sess.run(losses,feed_dict = feed_dict_valid)
        print(acc/len(x_batch))
        print("valid loss: ", valid_acc/len(x_valid))

        # Check predictions vs labels
        if i % 20 == 0:
            out_valid = sess.run(output,feed_dict = feed_dict_valid)
            for a,b in zip(y_valid,out_valid):
                print(a,b)

        # Store accuracies
        accs.append(acc/len(x_batch))
        test_accs.append(test_acc/len(y_valid))

print("train max: ",min(accs))
print("valid max: ",min(test_accs))


# Graph training/validation accuracy for each epoche
plt.plot(accs,'r')
plt.plot(test_accs, 'b')
plt.ylabel('mse')
plt.xlabel('rounds')
plt.legend(['train_accuracy', 'valid_accuracy'], loc='upper right')
plt.savefig("sth4.png")

