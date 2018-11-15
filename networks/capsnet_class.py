#!/usr/bin/env python3
import numpy as np
import os
from rnn_class import RNN
import tensorflow as tf

class CapsNet(RNN):
    def __init__(self, model_id, **kwargs):
        self.n_first_caps = kwargs["n_first_caps"]
        self.caps_block_size = kwargs["caps_block_size"]
        self.model_type = "ResNet-biGRU-RNN"
        RNN.__init__(self, model_id, **kwargs)


    def network_layer(self, x_input, conv_layer_size, kernel_in, pad="same"):
        # conv layer to create feature maps
        layer_output = tf.layers.conv1d(x_input, conv_layer_size, kernel_in, padding=pad)
        layer_output = tf.nn.relu(layer_output)

        # first capslayer
        for d in range(self.n_first_caps):
            with tf.name_scope("CapsBlock_{}".format(d)):
                layer_output = self.caps_block(layer_output, self.caps_block_size)
        
        # second caps layer

        # RNN
        network_output = RNN.network_layer(self, layer_output)    
        
        return network_output

    def caps_block():
        pass

    def routing():
        pass

    def squash(vector):
        '''
        Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return vec_squashed

    def save_info(self):
        RNN.save_info(self)
        print("Number of CapsBlocks: {}\nSize of CapsBlocks: {}".format(
                  self.n_first_caps, self.caps_block_size))           # use this one if args in are changed
