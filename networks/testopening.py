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


training_nr = 2 * 30131712          # // window * window    - shouldn't matter because all windows are 35
test_nr = 5250

print("Number of training reads: {}".format(training_nr))

model_id = 0

#~ network = RNN(model_id, batch_size=batch_size, layer_size=layer_size, 
                #~ n_layers=n_layers, optimizer_choice=optimizer_choice,  
                #~ learning_rate=learning_rate, keep_prob=keep_prob)
               
network = ResNetRNN(model_id, batch_size=batch_size, layer_size=layer_size, 
                n_layers=n_layers, optimizer_choice=optimizer_choice, 
                learning_rate=learning_rate, keep_prob=keep_prob, 
                n_layers_res=depth, layer_size_res=layer_size_res)
               

train_x, train_y = train.retrieve_set(db_dir, training_nr, training_type="trainingreads")
test_x, test_y = train.retrieve_set(db_dir_ts, test_nr, training_type="squiggles")

network.train_network(train_x, train_y, training_type="trainingreads")
network.test_network(test_x, test_y)                # eg. "/mnt/nexenta/thijs030/networks/GRUtestclass/checkpoints-100"

print("\n\n\n\nWindows for testing now:")
test_nr = 2 * 6336390 // window * window
testu_x, testu_y = train.retrieve_set(db_dir_ts, test_nr, training_type="trainingreads")
network.test_network(testu_x, testu_y)
