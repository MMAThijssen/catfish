#!/usr/bin/env python3
import os.path
from rnn_class import RNN
import tensorflow as tf

#### RESNET ######

# how many blocks? 3, 5, 8, 32 etc

#~ STRIDE = 1              # you want to check every signal - or maybe 7       # is 1 by default
#~ PADDING = "same"       # or valid - what diff?

#TODO: input of arguments cleaner

class ResNetRNN(RNN):
    
    def __init__(self, model_id, **kwargs):
        self.n_layers_res = kwargs["n_layers_res"]
        self.layer_size_res = kwargs["layer_size_res"]
        self.model_type = "ResNet-biGRU-RNN"
        RNN.__init__(self, model_id, **kwargs)
    
    
    def network_layer(self, x_input):
        layer_output = x_input
        for d in range(self.n_layers_res):
            with tf.name_scope("ResNet_layer{}".format(d)):
                layer_output = self.residual_block(layer_output, self.layer_size_res)
        
        network_output = RNN.network_layer(self, layer_output)          # comment to have just ResNet instead of Resnet-RNN
        
        return network_output
            

    def residual_block(self, x_input, layer_size, kernel_in=1, kernel_mid=3, kernel_out=1, pad="same"):
        """
        Args:
            x -- tensor, input
        """
        with tf.name_scope("residual_block"):           # uitgeschreven voor duidelijkheid
            # shortcut 
            sc_output = tf.layers.conv1d(x_input, layer_size, kernel_in, padding=pad)
            sc_output = tf.layers.batch_normalization(sc_output)
            
            # first conv layer
            output1 = tf.layers.conv1d(x_input, layer_size, kernel_in, padding=pad)
            output1 = tf.layers.batch_normalization(output1)
            output1 = tf.nn.relu(output1)
            
            # second conv layer
            output2 = tf.layers.conv1d(output1, layer_size, kernel_mid, padding=pad)
            output2 = tf.layers.batch_normalization(output2)
            output2 = tf.nn.relu(output2)
            
            # third conv layer              - or just two?
            output3 = tf.layers.conv1d(output2, layer_size, kernel_out, padding=pad)
            output3 = tf.layers.batch_normalization(output3)
            output3 = tf.nn.relu(output3)
            
            # add shortcut
            rb_output = tf.add(output3, sc_output)
            rb_output = tf.nn.relu(rb_output)
        
            return rb_output
        
        
    def set_model_path(self):
        cur_dir = "/mnt/nexenta/thijs030/networks"                      # change to take abs path - good for now
        model_path = cur_dir + "/" + self.model_type
        
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        print("\nSaving to ", model_path, "\n")
        
        return model_path  
    
    
    def save_info(self):
        RNN.save_info(self)
        #~ with open(self.model_type + ".txt", "a") as dest:
        print("Layer size ResNet: {}\nNumber of layers ResNet: {}".format(
                  self.layer_size_res, self.n_layers_res))           # use this one if args in are changed

#~ Again, inputs specifies the input tensor, with a shape of [batch_size, image_height, image_width, channels]inputs specifies the input tensor, with a shape of [batch_size, image_height, image_width, channels]
