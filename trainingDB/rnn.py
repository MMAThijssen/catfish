#!/usr/bin/env python3
"""
Recurrent neural network

@author: thijs030
"""
from helper_functions import load_db
from math import ceil
import numpy as np
import reader
from sys import argv
import tensorflow as tf

## hyperparameters
#batch_size = [32, 64]    # ook nog 128?
#learning_rate = range(0.0001, 1)     # choose log scale
#optimizer = ["Adam", "RMSprop"] #adjust
#layer_size = [16, 32, 64, 128, 256] 
#n_layers = range(1, 6)
#dropout = range(0, 1)

# retrieve training set
db_dir = argv[1]
n_train = 10

db, npz_files = load_db(db_dir)
#for npz_file in npz_files:
#    x, y = reader.load_npz(npz_file)    # x and y are both np.ndarrays
#print(x.shape)  #1039195,
#print(y.shape)  #1039195,


train_x, train_y = db.get_training_set(n_train) # x is a tuple of np.ndarrays; y is a tuple of lists
train_y = np.asarray(train_y)

test_data, test_labels = db.get_training_set(n_train, sets="test")

len_x = len(train_x[0]) # number of points per training examples
n_x = len(train_x)      # number of training examples

# network
## hyperparameters for initial testing
layer_size = 16
n_classes = 2
learning_rate = 0.1
batch_size = 32
lambda_loss = 0.001

def accuracy(y_predicted, y):
    return (100.0 * np.sum(np.argmax(y_predicted, 1) == np.argmax(y, 1)) / y_predicted.shape[0])


def rnn_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)
    
    cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)

    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]
    
    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit


# 1. tf.Graph contains computational steps required for NN
## initialize placeholders
graph = tf.Graph()
with graph.as_default():
    data = tf.placeholder(tf.float32, shape=(None, len_x))
    labels = tf.placeholder(tf.float32, shape=(None, len_x))

## define model and caclualte output values (logits)
    logits = rnn_model(data, layer_size, n_classes)
    
## calculate loss using logits
    l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + l2
 
## use optimizer for weights
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    prediction= tf.nn.softmax(logits)

# 2. tf.Session executes tf.Graph
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized global variables.")
    n_batches = ceil(n_x / batch_size)
    for n in range(n_batches):
        offset = n * batch_size
        batch_data = data[offset : offset + batch_size]
        batch_labels = labels[offset : offset + batch_size]
        
        feed_dict = {data :  batch_data, labels :  batch_labels}
        _, l, train_predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
        train_accuracy = accuracy(train_predictions, batch_labels)

        if n % (n_batches // 2) == 0:
            feed_dict = {data :  test_data, labels :  test_labels}
            _, test_predictions = session.run([loss, prediction], feed_dict=feed_dict)
            test_accuracy = accuracy(test_predictions, test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {} %, accuracy on test set {:02.2f} %".format(n, l, train_accuracy, test_accuracy)
            print(message)

# softmax layer
#tf.layers.dense(final_state, params.num_classes)

#tf.contrib.rnn.GRUCell
# or tf.contrib.cudnn_rnn.CudnnGRU      for getter performance on GPU
# or tf.contrib.rnn.GRUBlockCellV2      for CPU
#
# argument names have NOTHING to do with hyperparameters for now
#def gru_cell():
#    return(tf.contrib.rnn.GRUCell(gru_size))
#    
#stacked_gru = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(n_layers)])
#state = stacked_gru.zero_state(batch_size, tf.float32)
#
# from tensorflow on quickdraw
#We pass the output from the convolutions into bidirectional LSTM layers for which we use a helper function from contrib.
#
#outputs, _, _ = contrib_rnn.stack_bidirectional_dynamic_rnn(
#    cells_fw=[cell(params.num_nodes) for _ in range(params.num_layers)],
#    cells_bw=[cell(params.num_nodes) for _ in range(params.num_layers)],
#    inputs=convolved,
#    sequence_length=lengths,
#    dtype=tf.float32,
#    scope="rnn_classification")