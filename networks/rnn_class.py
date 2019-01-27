#!/usr/bin/env python3

import trainingDB.metrics
import numpy as np
import os
import os.path
import tensorflow as tf

class RNN(object):
    def __init__(self, save=False, **kwargs):
        
        # adjustable parameters
        self.batch_size = kwargs["batch_size"]
        self.optimizer_choice = kwargs["optimizer_choice"]
        self.learning_rate = kwargs["learning_rate"]
        self.layer_size = kwargs["layer_size"]        
        self.n_layers = kwargs["n_layers"]
        self.keep_prob = kwargs["keep_prob"]
        self.keep_prob_test = 1.0                                               # no dropout wanted in testing
                
        # set parameters
        #~ self.model_id = model_id                                             # for pbt
        #~ self.name_scope = tf.get_default_graph().get_name_scope()            # for pbt
        
        self.n_inputs = 1        
        self.n_outputs = 1
        self.window = 35
        self.max_seq_length = 5250  
        
        self.layer_sizes = [self.layer_size,] * self.n_layers                   # does this work when extended or shortened? maybe move to the network layer
        self.saving_step = 10000
        
        self.cell_type = "GRU"
        self.model_type = self.cell_type
        self.sess = tf.Session()    
             
        
        # build network and additionals
        self.input_layer()
        self.layer_output = self.network_layer(self.x)
        self.logits = self.output_layer(self.layer_output)
        self.loss = self.compute_loss()
        self.accuracy = self.compute_accuracy()
        self.optimizer = self.optimizer_choice
        if save:
            self.model_path = self.model_type
            self.save_info()
            self.summary = self.activate_tensorboard()
        self.saver = tf.train.Saver(max_to_keep=1000, save_relative_paths=True)
        
        # saving test performance
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

              
    @property
    def optimizer(self):
        return self._optimizer


    @optimizer.setter
    def optimizer(self, optimizer_choice):
        with tf.name_scope("optimizer"):
            if self.optimizer_choice == "Adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer_choice == "RMSProp":
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            else:
                raise ValueError("Given optimizer choice is not known. Choose 'Adam' or 'RMSProp'.")
            self._optimizer = optimizer.minimize(self.loss)

    
    def compute_loss(self):    
        with tf.name_scope("loss"):
            loss_ce = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(loss_ce)
        
        return loss
            
    
    def compute_accuracy(self):
        with tf.name_scope("accuracy"):
            self.predictions = tf.nn.sigmoid(self.logits)
            correct = tf.equal(tf.round(self.predictions), self.y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        return accuracy
    
    
    @property
    def model_type(self):
        return self._model_type
        
    @property
    def model_path(self):
        return self._model_path
    
    
    @model_path.setter
    def model_path(self, model_type):
        cur_dir = "/mnt/scratch/thijs030/actualnetworks"
        #~ cur_dir = os.getcwd()               
            
        check_for_dir = True
        number = 0
        model_path = cur_dir + "/" + model_type + "_" + str(number)
        while check_for_dir:
            if os.path.isdir(model_path):
                number += 1
                model_path = model_path.rsplit("_")[0] + "_" + str(number)
            else:
                os.mkdir(model_path)
                check_for_dir = False
                
        print("\nSaving network checkpoints to", model_path, "\n")
        
        self._model_path = model_path
        
        
    @model_type.setter
    def model_type(self, cell_type):
        self._model_type = "bi" + self.cell_type + "-RNN"
                
        
    def activate_tensorboard(self):
        with tf.name_scope("TensorBoard"):            
            print("\nCurrent directory: {}\nUse 'tensorboard --logdir=[cur_dir]'\n".format(self.model_path))
            
            self.writer = tf.summary.FileWriter(self.model_path + "/tensorboard")
            self.writer.add_graph(tf.get_default_graph())
            
            # summaries
            loss_summ = tf.summary.scalar("loss", self.loss)
            acc_summ = tf.summary.scalar("accuracy", self.accuracy)
        
            all_summs = tf.summary.merge_all()
                
            return all_summs
        

    def set_cell(self, layer_size):
        if self.cell_type == "basic":
            cell = tf.contrib.rnn.BasicRNNCell(num_units=layer_size)          # activation=tf.nn.relu
        elif self.cell_type == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=layer_size)
        
        return cell
        

    def dropout(self, cell):
        dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.p_dropout)
        
        return dropout_cell
        

    def input_layer(self):
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.window, self.n_inputs])
            self.y = tf.placeholder(tf.float32, shape=[None, self.window, self.n_outputs])
        
        self.p_dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

            
    def network_layer(self, x_input):
        with tf.name_scope("recurrent_layer"):
            stacked_cells_fw = [self.dropout(self.set_cell(size)) for size in self.layer_sizes]
            stacked_cells_bw = [self.dropout(self.set_cell(size)) for size in self.layer_sizes]
                
            self.rnn_output, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                    stacked_cells_fw, stacked_cells_bw, x_input, dtype=tf.float32)
        
        stacked_rnn_output = tf.reshape(self.rnn_output, [-1, self.layer_size * 2])           # 2 cause bidirectional    
        
        return stacked_rnn_output
        
    
    def output_layer(self, stacked_rnn_output):
        stacked_outputs = tf.layers.dense(stacked_rnn_output, self.n_outputs, name="final_fully_connected")
        
        final_output = tf.reshape(stacked_outputs, [-1, self.window, self.n_outputs]) 
        
        return final_output
        
    
    def initialize_network(self):
        self.sess.run(tf.global_variables_initializer())
        print("\nNot yet initialized: ", self.sess.run(tf.report_uninitialized_variables()), "\n")


    def restore_network(self, path, ckpnt="latest", meta=None):
        self.sess.run(tf.global_variables_initializer())
        if ckpnt == "latest":
            self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
        else:
            self.saver.restore(self.sess, path + "/" + ckpnt)

        print("Model {} restored\n".format(path.split("/")[-2]))

    
    def train_network(self, train_x, train_y, step):        
        feed_dict = {self.x: train_x, 
                     self.y: train_y, 
                     self.p_dropout: self.keep_prob}

        self.sess.run(self.optimizer, feed_dict=feed_dict)
        
        summary = self.sess.run(self.summary, feed_dict=feed_dict)

        self.writer.add_summary(summary, step)  
    
    
    def infer(self, input_x):
        feed_dict_pred = {self.x: input_x, self.p_dropout: self.keep_prob_test}
        
        confidences = self.sess.run(self.predictions, feed_dict=feed_dict_pred) 
        confidences = np.reshape(confidences, (-1)).astype(float) 
        
        return(confidences)        
        

    def test_network(self, test_x, test_y, read_name, file_path, threshold=0.5):
        # get predicted values:
        feed_dict_pred = {self.x: test_x, self.p_dropout: self.keep_prob_test}

        confidences = self.sess.run(self.predictions, feed_dict=feed_dict_pred) 
        confidences = np.reshape(confidences, (-1)).astype(float)               # is necessary! 150 > 5250
        
        pred_vals = [1 if c >= threshold else 0 for c in confidences]
        
        # get testing accuracy:
        feed_dict_test = {self.x: test_x, self.y: test_y, self.p_dropout: self.keep_prob_test}
        test_acc, test_loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict_test)
    
        # evaluate performance:
        test_labels = test_y.reshape(-1)
        true_pos, false_pos, true_neg, false_neg = metrics.confusion_matrix(test_labels, pred_vals)
        self.tp += true_pos
        self.fp += false_pos
        self.tn += true_neg
        self.fn += false_neg

        #~ with open(file_path + "_checkpoint20000_c.txt", "a+") as dest:
            #~ dest.write(read_name)
            #~ dest.write("\n")
            #~ dest.write("* {}".format(list(test_labels)))
            #~ dest.write("\n")
            #~ dest.write("# {}".format(list(pred_vals)))
            #~ dest.write("\n")
            #~ dest.write("@ {}".format(list(confidences)))
            #~ dest.write("\n")
                    
        return test_acc, test_loss

    
    def save_info(self):
        with open(self.model_path + ".txt", "w") as dest:
            dest.write("MODEL TYPE: {}\n\n".format(self.model_type))
            dest.write("batch_size: {}\noptimizer_choice: {}\nlearning_rate: {}\n".format(
                      self.batch_size, self.optimizer_choice, self.learning_rate))
            dest.write("layer_size: {}\nn_layers: {}\nkeep_prob: {}\n".format(
                      self.layer_size, self.n_layers, self.keep_prob))
