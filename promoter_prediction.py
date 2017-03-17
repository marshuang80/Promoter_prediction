import numpy as np
import re
import itertools
import csv
import random
from collections import Counter
import tensorflow as tf
from tensorflow.contrib import learn
import pickle
def load_data():
    promoters = pickle.load(open("onehot500.pickle",'rb'))
    x =[a[-1].T for a in promoters[4:]]
    #x = [a for a in x if len(a) != 501]
    y= [int(a[1]) for a in promoters[4:] if int(a[1]) != -1]
    #y = [[1,0] if a[-2] >0 else [0,1] for a in promoters[5:]]
    return x, onehot(y)

def onehot(Y):
    labels = []
    for a in Y:
        one_hot = [0]*18
        one_hot[a-1] = 1
        labels.append(one_hot)
    return labels

#shuffle data
def shuffle_data(x,y):
    np.random.seed(7)
    idx = np.random.permutation(np.arange(len(y)))
    y = np.array(y)
    return x[idx], y[idx]

#Create batches of data with size "batch_size)
def batch(x,y,batch_size):
    idx = random.sample(range(len(x)),batch_size)
    return x[idx],y[idx]

#generate training and validation dataset
def generate_data():
    print("loading data...")
    x,y = load_data()
    x,y = np.array(x),np.array(y)
    print("shuffling data...")
    x, y = shuffle_data(x,y)
    x_test = x[10000:]
    x = x[:10000]
    y_test = y[10000:]
    y = y[:10000]
    return x,x_test,y,y_test

x,x_test,y,y_test = generate_data()

##### CNN Model #####
#hypterparameters
promoter_len = 501 
num_filters = 128
filter_size = 8
epoche = 200
alpha = 1e-3
classes = 18
accs,test_accs = [],[]
#model
sess = tf.Session()
with sess.as_default():
    print("building convolutional neural network...")
    #placeholders
    x_input = tf.placeholder(tf.float32,[None, promoter_len,4])
    y_input = tf.placeholder(tf.float32,[None, classes])
    x_in = tf.expand_dims(x_input,-1)
    #Conv layer
    outputs  = []
    #iteration through different filter size
    with tf.name_scope("con%s"%(str(filter_size))):
        w = tf.Variable(tf.truncated_normal([filter_size,4,1, num_filters],stddev = 0.1))
        b = tf.Variable(tf.constant(0.1,shape=[num_filters]))
        conv = tf.nn.conv2d(x_in, w, strides = [1,1,1,1], padding = "SAME")
        #activation:
        out = tf.nn.relu(tf.nn.bias_add(conv,b))
        #max pooling
        pool = tf.nn.max_pool(out,ksize = [1, promoter_len, 4, 1], strides = [1,1,1,1], padding = 'VALID')
        #pool = tf.nn.max_pool(out,ksize = [1,8,4, 1], strides = [1,8,1,1], padding = 'VALID')
    #combine pooled layer
    #concatanate on the 3rd dimention (filters)
    pool_flatten = tf.reshape(pool, [-1, num_filters])
    #dropout layer
    with tf.name_scope("dropout"):
        drop_layer = tf.nn.dropout(pool_flatten,0.5)
    with tf.name_scope("fc1"):
        w1 = tf.Variable(tf.truncated_normal([num_filters,180],stddev = 0.1))
        b1 = tf.Variable(tf.constant(0.1,shape = [180]))
        fc1 = tf.nn.softmax(tf.matmul(drop_layer,w1)+b1)
    #output predictions
    with tf.name_scope("output"):
        W = tf.get_variable("W",shape = [180,classes],initializer=tf.contrib.layers.xavier_initializer())
        B = tf.Variable(tf.constant(0.1,shape=[classes]), name = 'b')
        #scores = tf.nn.xw_plus_b(drop_layer,W,B)
        output = tf.nn.softmax(tf.matmul(fc1,W)+B)
        predictions = tf.argmax(output,1)
        #predictions = tf.argmax(scores,1) 
    #cross-entropy loss
    with tf.name_scope("loss"):
        #logits: sum of scores!= 1/ scores are not probablistic 
        losses = tf.reduce_mean(-tf.reduce_sum(y_input*tf.log(output),reduction_indices=[1])) 
        #losses = tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = y_input)
    with tf.name_scope("accuracy"):
        pred = tf.equal(predictions,tf.argmax(y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(pred,"float"))
    train = tf.train.AdamOptimizer(alpha).minimize(losses)
    #grad = optimizer.compute_gradients(losses)
    #train = optimizer.apply_gradients(grad)
    sess.run(tf.global_variables_initializer())
    print("start training...")
    x_valid, y_valid = batch(x_test,y_test,128)
    for i in range(epoche):
        x_batch, y_batch = batch(x,y,int((len(x)/4)))
        #x_batch, y_batch = batch(x,y,512)
        feed_dict = {
                x_input: x_batch, 
                y_input: y_batch,
            } 
        sess.run(train,feed_dict = feed_dict) 
        #acc = sess.run(accuracy,feed_dict = feed_dict) 
        feed_dict_valid = {
                x_input: x_valid, 
                y_input: y_valid,
            }  
        acc = sess.run(accuracy,feed_dict = feed_dict) 
        print(acc)
        test_acc = sess.run(accuracy,feed_dict = feed_dict_valid)
        print("test: ",test_acc)
        accs.append(acc)
        test_accs.append(test_acc)

print("train max: ",max(accs))
print("valid max: ",max(test_accs))

import matplotlib.pyplot as plt
plt.plot(accs,'r')
plt.plot(test_accs, 'b')
plt.ylabel('% accuracy')
plt.xlabel('rounds')
plt.legend(['train_accuracy', 'valid_accuracy'], loc='lower right')
plt.show()
