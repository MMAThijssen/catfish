#!/usr/bin/env python3
import os.path
from models.rnn_class import RNN
import tensorflow as tf


class ResNetRNN(RNN):
    
    def __init__(self, **kwargs):
        self.n_layers_res = kwargs["n_layers_res"]
        self.layer_size_res = kwargs["layer_size_res"]
        self.network_type = "ResNet-RNN"
        self.model_type = self.network_type
        RNN.__init__(self, **kwargs)
        
    
    def network_layer(self, x_input):
        layer_output = x_input
        for d in range(self.n_layers_res):
            with tf.name_scope("ResNet_layer_{}".format(d)):
                layer_output = residual_block(layer_output, self.layer_size_res)
        
        network_output = RNN.network_layer(self, layer_output)                  # comment to have just ResNet instead of Resnet-RNN
        
        return network_output
       
    
    def save_info(self):
        RNN.save_info(self)
        with open(self.model_path + ".txt", "a") as dest:
            dest.write("layer_size_res: {}\nn_layers_res: {}\n\n".format(
                  self.layer_size_res, self.n_layers_res))        
    
    @property
    def model_type(self):
        return self._model_type  
                    
                  
    @model_type.setter
    def model_type(self, network_type):
        self._model_type = self.network_type
            

def residual_block(x_input, layer_size, kernel_in=1, kernel_mid=3, kernel_out=1, pad="same"):
    """
    Residual block with bottleneck architecture and skip connection. 
    
    Args:
        x_input -- tensor, input [batch_size, window, number of features]
        layer_size -- int, number of hidden units in each layer
        kernel_in -- int
        kernel_mid -- int
        kernel_out -- int
        pad -- str, padding of data, either "same" or "valid" [default: "same"]
        
    Returns: tensor
    """
    with tf.name_scope("residual_block"):                                       # uitgeschreven voor duidelijkheid
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
        
        # third conv layer                                                      # or just two? - this arch is bottleneck now
        output3 = tf.layers.conv1d(output2, layer_size, kernel_out, padding=pad)
        output3 = tf.layers.batch_normalization(output3)
        output3 = tf.nn.relu(output3)
        
        # add shortcut
        rb_output = tf.add(output3, sc_output)
        rb_output = tf.nn.relu(rb_output)
    
        return rb_output
