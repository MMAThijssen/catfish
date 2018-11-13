#!/usr/bin/env python3
from resnet_class import ResNetRNN
from rnn_class import RNN
import train

## parameters for now ##
batch_size = 64
max_seq_length = 5250
window = 35
n_outputs = 1
n_classes = 1
n_inputs = 1
learning_rate = 0.01
n_epochs = 10
loss_history = []
n_layers = 5
layer_size = 32
keep_prob = 0.8     
optimizer_choice = "Adam"

#~ training_type = "trainingreads"

## additional for ResNet:
layer_size_res = 32
depth = 3

## for training ##
#~ db_dir = "/mnt/nexenta/thijs030/data/trainingDB/test3/"
db_dir = "/mnt/nexenta/thijs030/data/trainingDB/training4000w34/"
db_dir_ts = "/mnt/nexenta/thijs030/data/trainingDB/val857w34/"


training_nr = 2 * 30131712 // window * window
test_nr = 5250

print("Number of training reads: {}\nNumber of test reads: {}".format(
        training_nr, test_nr))

#~ network = RNN(batch_size=batch_size, max_seq_length=max_seq_length, layer_size=layer_size, 
                #~ n_layers=n_layers, optimizer_choice=optimizer_choice, n_epochs=n_epochs, 
                #~ learning_rate=learning_rate, keep_prob=keep_prob, n_train=training_nr)
               
network = ResNetRNN(batch_size=batch_size, max_seq_length=max_seq_length, layer_size=layer_size, 
                n_layers=n_layers, optimizer_choice=optimizer_choice, n_epochs=n_epochs, 
                learning_rate=learning_rate, keep_prob=keep_prob, n_train=training_nr,
                n_layers_res=depth, layer_size_res=layer_size_res)
               

train_x, train_y = train.retrieve_set(db_dir, training_nr, training_type="trainingreads")
test_x, test_y = train.retrieve_set(db_dir_ts, test_nr, training_type="squiggles")

network.train_network(train_x, train_y, training_type="trainingreads")
network.test_network(test_x, test_y)                # eg. "/mnt/nexenta/thijs030/networks/GRUtestclass/checkpoints-100"

print("\n\n\n\nWindows for testing now:")
test_nr = 2 * 6336390 // window * window
testu_x, testu_y = train.retrieve_set(db_dir_ts, test_nr, training_type="trainingreads")
network.test_network(testu_x, testu_y)
