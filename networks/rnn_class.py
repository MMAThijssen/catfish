#!/usr/bin/env python3
#~ from functools import lru_cache
import metrics
import numpy as np
import os
import tensorflow as tf

#TODO:
# * implement check for given max_seq_length: must be divisible
# * fix set batches (self.n_train now..)
# * make properties of n_batches, cell_type (set_cell), saver, model_path
# * change cur dir to take abspath
# * ASK: What is the difference between a class def returning something or setting self.variable?


class RNN(object):
#    def __init__(self):
    #~ def __init__(self, batch_size, layer_size, n_layers, 
                   #~ optimizer_choice, n_epochs, learning_rate, keep_prob):
    def __init__(self, **kwargs):

        #~ tf.set_random_seed(16)
        
        # adjustable parameters
        self.batch_size = kwargs["batch_size"]
        self.optimizer_choice = kwargs["optimizer_choice"]
        self.learning_rate = kwargs["learning_rate"]
        self.layer_size = kwargs["layer_size"]        
        self.n_layers = kwargs["n_layers"]
        self.keep_prob = kwargs["keep_prob"]
        self.keep_prob_test = 1.0                                               # no dropout wanted in testing
                
        # set parameters
        #~ self.model_id = model_id                                                # for pbt
        #~ self.name_scope = tf.get_default_graph().get_name_scope()            # for pbt
        
        self.n_inputs = 1        
        self.n_outputs = 1
        self.window = 35
        self.max_seq_length = 5250  
        
        self.layer_sizes = [self.layer_size,] * self.n_layers                   # does this work when extended or shortened? maybe move to the network layer
        self.saving_step = 100
        
        self.cell_type = "GRU"
        self.model_type = "bi" + self.cell_type + "-RNN"
        #~ self.model_type = "NN"
        self.sess = tf.Session()          
        
        # build network and additionals
        self.input_layer()
        self.layer_output = self.network_layer(self.x)
        self.logits = self.output_layer(self.layer_output)
        self.loss = self.compute_loss()
        self.accuracy = self.compute_accuracy()
        self.optimizer = self.optimizer_choice
        self.model_path = self.model_type
        self.save_info()
        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)
        self.summary = self.activate_tensorboard()
        #~ self.initialize_network()
        
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
    def model_path(self):
        return self._model_path
    
    
    @model_path.setter
    def model_path(self, model_type):
        cur_dir = "/mnt/nexenta/thijs030/networks"
        #~ cur_dir = "/lustre/scratch/WUR/BIOINF/thijs030/networks"                # "/mnt/nexenta/thijs030/networks"              
            
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
        
    #~ @property.setter
    #~ def saver(self):  
        #~ with tf.name_scope("saver"):
            #~ self._saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)
    
    #~ @property
    #~ def saver(self):
        #~ return _saver
        
        
    def activate_tensorboard(self):
        with tf.name_scope("TensorBoard"):            
            print("\nCurrent directory: {}\nUse 'tensorboard --logdir=[cur_dir]'\n".format(self.model_path))
            
            self.writer = tf.summary.FileWriter(self.model_path + "/tensorboard")
            self.writer.add_graph(tf.get_default_graph())
            
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
        
    
    def initialize_network(self, mode="initial"):
        #~ self.sess = tf.Session()
        if mode=="initial":
            self.sess.run(tf.global_variables_initializer())
            print("\nNot yet initialized: ", self.sess.run(tf.report_uninitialized_variables()), "\n")

    def restore_network(self, path=None, ckpnt="latest", meta=None):
            self.sess.run(tf.global_variables_initializer())
            #~ new_saver = tf.train.import_meta_graph(path + "/" + meta)
            if ckpnt == "latest":
                #~ tf.train.Saver(tf.global_variables()).restore(self.sess, tf.train.latest_checkpoint(path))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
            else:
                #~ saver.restore(self.sess, ckpnt)
                new_saver.restore(self.sess, path + "/" + ckpnt)

            print("Model {} restored\n".format(os.path.basename(path)))
            

#~ def restore_network(sess, path, ckpnt="latest"):
    #~ # restore model:
    #~ self.sess.run(tf.global_variables_initializer())
    #~ saver = tf.train.import_meta_graph("{}/checkpoints-1.meta".format(path))


            
    
    def train_network(self, train_x, train_y, step):        
        feed_dict = {self.x: train_x, 
                     self.y: train_y, 
                     self.p_dropout: self.keep_prob}

        _, summary = self.sess.run([self.optimizer, self.summary], feed_dict=feed_dict)

        self.writer.add_summary(summary, step)  
        
        

    def test_network(self, test_x, test_y, read_nr, ckpnt="latest"):
        # get predicted values:
        feed_dict_pred = {self.x: test_x, self.p_dropout: self.keep_prob_test}
        pred_vals = self.sess.run(tf.round(self.predictions), feed_dict=feed_dict_pred)                  
        pred_vals = np.reshape(pred_vals, (-1)).astype(int)   
        
        confidences = self.sess.run(self.predictions, feed_dict=feed_dict_pred) 
        confidences = np.reshape(confidences, (-1)).astype(float)       # is necessary! 150 > 5250
        
        # get testing accuracy:
        feed_dict_test = {self.x: test_x, self.y: test_y, self.p_dropout: self.keep_prob_test}
        test_acc = self.sess.run(self.accuracy, feed_dict=feed_dict_test)
        
    
    #~ def evaluate_performance(self, test_y):
        test_labels = test_y.reshape(-1)
        true_pos, false_pos, true_neg, false_neg = metrics.confusion_matrix(test_labels, pred_vals)
        self.tp += true_pos
        self.fp += false_pos
        self.tn += true_neg
        self.fn += false_neg

        #~ test_precision, test_recall = metrics.precision_recall(true_pos, false_pos, false_neg)
        #~ test_f1 = metrics.weighted_f1(test_precision, test_recall, true_pos + false_neg, len(test_labels))
        #~ roc_auc = metrics.calculate_auc(test_labels, confidences)
        with open(self.model_path + "_labels.txt", "a+") as dest:
            dest.write("\n")
            dest.write("{}".format(list(test_labels)))
        with open(self.model_path + "_predictions.txt", "a+") as dest:
            dest.write("\n")
            dest.write("{}".format(list(confidences)))
        
        if read_nr % 100 == 0:
            metrics.generate_heatmap([pred_vals, confidences, test_labels], 
                                     ["homopolymer", "confidence", "truth"], "Comparison_{}_{}".format(os.path.basename(self.model_path), read_nr))
        #~ return test_acc, roc_auc, test_precision, test_recall, test_f1
        return test_acc

        
    
    def save_info(self):
        with open(self.model_path + ".txt", "w") as dest:
            #~ dest.write("Model type: {}\n".format(self.model_type))
            dest.write("batch_size: {}\noptimizer_choice: {}\nlearning_rate: {}\n".format(
                      self.batch_size, self.optimizer_choice, self.learning_rate))
            dest.write("layer_size: {}\nn_layers: {}\nkeep_prob: {}\n".format(
                      self.layer_size, self.n_layers, self.keep_prob))
    
    

        

#~ ############# NOT USED YET: PBT ################
    #~ @lru_cache(maxsize=None)                                            
    #~ def copy_from(self, other_model):
        #~ # This method is used for exploitation. We copy all weights and hyper-parameters
        #~ # from other_model to this model
        #~ my_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope + '/')
        #~ their_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, other_model.name_scope + '/')
        #~ assign_ops = [mine.assign(theirs).op for mine, theirs in zip(my_weights, their_weights)]
        #~ return tf.group(*assign_ops)

        
    #~ def _restore(self, checkpoint_path):
        #~ reader = tf.train.NewCheckpointReader(checkpoint_path)
        #~ for var in self.saver._var_list:
            #~ tensor_name = var.name.split(':')[0]
            #~ if not reader.has_tensor(tensor_name):
                #~ continue
            #~ saved_value = reader.get_tensor(tensor_name)
            #~ resized_value = fit_to_shape(saved_value, var.shape.as_list())
            #~ var.load(resized_value, self.sess)


    #~ def fit_to_shape(array, target_shape):
        #~ source_shape = np.array(array.shape)
        #~ target_shape = np.array(target_shape)

        #~ if len(target_shape) != len(source_shape):
            #~ raise ValueError('Axes must match')

        #~ size_diff = target_shape - source_shape

        #~ if np.all(size_diff == 0):
            #~ return array

        #~ if np.any(size_diff > 0):
            #~ paddings = np.zeros((len(target_shape), 2), dtype=np.int32)
            #~ paddings[:, 1] = np.maximum(size_diff, 0)
            #~ array = np.pad(array, paddings, mode='constant')

        #~ if np.any(size_diff < 0):
            #~ slice_desc = [slice(d) for d in target_shape]
            #~ array = array[slice_desc]

        #~ return array
