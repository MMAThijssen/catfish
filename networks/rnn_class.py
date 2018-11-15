#!/usr/bin/env python3
from functools import lru_cache
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
    def __init__(self, model_id, **kwargs):

        #~ tf.set_random_seed(16)
        
        # adjustable parameters
        self.batch_size = kwargs["batch_size"]
#        self.max_seq_length = kwargs["max_seq_length"]
        self.optimizer_choice = kwargs["optimizer_choice"]
        self.learning_rate = kwargs["learning_rate"]
        self.layer_size = kwargs["layer_size"]        
        self.n_layers = kwargs["n_layers"]
        self.keep_prob = kwargs["keep_prob"]
        self.keep_prob_test = 1.0               # no dropout wanted in testing
                
        # set parameters
        self.model_id = model_id                                                # for pbt
        #~ self.name_scope = tf.get_default_graph().get_name_scope()       # for pbt
        
        self.n_inputs = 1        
        self.n_outputs = 1
        self.window = 35
        self.max_seq_length = 5250  
        
        self.loss_history = []
        self.layer_sizes = [self.layer_size,] * self.n_layers       # does this work when extended or shortened? maybe move to the network layer
        self.saving_step = 10000
        
        self.extra = False
        #~ self.save = True
        self.cell_type = "GRU"
        
        self.model_type = "bi" + self.cell_type + "-RNN"
        self.save_info()
        self.sess = tf.Session()            # is this right and change everywhere

        # build network and additionals
        #~ self.build_network(self.rnn_layer())
        self.input_layer()
        self.layer_output = self.network_layer(self.x)
        self.logits = self.output_layer(self.layer_output)
        self.loss = self.compute_loss()
        self.accuracy = self.compute_accuracy()
        self.optimizer = self.optimizer_choice
        self.model_path = self.set_model_path()
        self.saver = self.set_saver()
        self.summary = self.activate_tensorboard()
        self.initialize_network()

              
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
        
    
    def set_model_path(self):
        cur_dir = "/mnt/nexenta/thijs030/networks"                      # change to take abs path - good for now
        model_path = cur_dir + "/" + self.model_type + "_" + str(self.model_id)
        check_for_dir = True
        
        #~ if not os.path.isdir(model_path):
            #~ os.mkdir(model_path)
        
        number = 0
        while check_for_dir:
            if os.path.isdir(model_path):
                number += 1
                model_path = model_path.rsplit("_")[0] + "_" + str(number)
            else:
                os.mkdir(model_path)
                check_for_dir = False
                
        print("\nSaving network checkpoints to", model_path, "\n")
        
        return model_path    
        
    
    def set_saver(self):  
        with tf.name_scope("saver"):
            saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)
        
        return saver
        
        
    def activate_tensorboard(self):
        with tf.name_scope("TensorBoard"):            
            print("\nCurrent directory: {}\nUse 'tensorboard --logdir=[cur_dir]'\n".format(self.model_path))
            
            self.writer = tf.summary.FileWriter(self.model_path + "/tensorboard")
            self.writer.add_graph(tf.get_default_graph())
            
            loss_summ = tf.summary.scalar("loss", self.loss)
            acc_summ = tf.summary.scalar("accuracy", self.accuracy)
        
            all_summs = tf.summary.merge_all()
            
            return all_summs
            
            
    def set_nbatches(self, n_train, training_type="trainingreads"):
        if training_type == "squiggles":
            n_batches = self.max_seq_length // self.batch_size // self.window      
        if training_type == "trainingreads":
            n_batches = n_train // self.batch_size
        
        return n_batches
        

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
            
        self.p_dropout = tf.placeholder(dtype=tf.float32, name="keep_prob")

            
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
        
        if self.extra:
            print("\n", "RNN output get shape: ", self.rnn_output.get_shape())
            print("Stacked RNN get shape: ", stacked_rnn_output.get_shape())           # changed from stacked_rnn_output to main_output to make work for different network architectures
            print("Stacked outputs shape ", stacked_outputs.get_shape())                
            print("Logits get shape: ", final_output.get_shape(), "\n")
            
        return final_output
        
    
    def initialize_network(self):
        self.sess.run(tf.global_variables_initializer())
        print("\nNot yet initialized: ", self.sess.run(tf.report_uninitialized_variables()), "\n")
        
        self.saver.save(self.sess, self.model_path + "/checkpoints/meta", write_meta_graph=True)
        print("\nSaved meta graph\n")
            
    
    def train_network(self, train_x, train_y, step):        
        feed_dict = {self.x: train_x, 
                     self.y: train_y, 
                     self.p_dropout: self.keep_prob}

        _, summary = self.sess.run([self.optimizer, self.summary], feed_dict=feed_dict)

        self.writer.add_summary(summary, step)  
        
        # removed step += 1


    def test_network(self, test_x, test_y, ckpnt="latest"):
        # restore model:
        self.sess.run(tf.global_variables_initializer())
        if ckpnt == "latest":
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
        else:
            self.saver.restore(self.sess, ckpnt)
        print("Model {} restored\n".format(os.path.basename(self.model_path)))
                      
        # get predicted values:
        pred_vals, confidences = self.predict(test_x)
#            pred_list = [pred_vals, confidences]       # limit to one squiggle: [:5250]
        #~ metrics.generate_heatmap(pred_list, ["homopolymer", "confidence"], "Predictions_{}".format(self.model_id))

        # get testing accuracy:
        feed_dict_test = {self.x: test_x, self.y: test_y, self.p_dropout: self.keep_prob_test}
        test_acc = self.sess.run(self.accuracy, feed_dict=feed_dict_test)
                
        # output performance:
        test_labels = test_y.reshape(-1)
        nonhp = np.count_nonzero(pred_vals == 0)
        hp = np.count_nonzero(pred_vals == 1)
        hp_true = np.count_nonzero(test_labels == 1)
        test_precision, test_recall = metrics.precision_recall(test_labels, pred_vals)
        test_f1 = metrics.weighted_f1(test_precision, test_recall, hp_true, hp + nonhp)
        tpr, fpr, roc_auc = metrics.calculate_auc(test_labels, confidences)
        metrics.draw_roc(tpr, fpr, roc_auc, "ROC_{}".format(self.model_id))
        print("Predicted percentage HPs: {:.2%}".format(hp / (nonhp + hp)))
        
        #~ metrics.generate_heatmap([pred_vals, confidences, test_labels], 
                                #~ ["homopolymer", "confidence", "truth"], "Comparison_{}".format(self.model_id))
        
        return test_acc, test_precision, test_recall, test_f1, roc_auc
    
    #TODO: if sequence is smaller than needed. Add nonsense values that can be cut later ("dead" value 0?)
    # predict assumes reshaped sequence         
    def predict(self, sequence):        
        # get predicted values:
        feed_dict_pred = {self.x: sequence, self.p_dropout: self.keep_prob_test}
        pred_vals = self.sess.run(tf.round(self.predictions), feed_dict=feed_dict_pred)                  
        pred_vals = np.reshape(pred_vals, (-1)).astype(int)    
        
        confidence = self.sess.run(self.predictions, feed_dict=feed_dict_pred) 
        confidence = np.reshape(confidence, (-1)).astype(float)
        
        return pred_vals, confidence
        
    
    def save_info(self):
        #~ print("Saving information to file: ", self.model_type + ".txt")
        #~ with open(self.model_type + ".txt", "w") as dest:
        print("Model type: {}".format(self.model_type))
        print("Batch size: {}\nOptimizer: {}\nLearning rate: {}".format(
                  self.batch_size, self.optimizer_choice, self.learning_rate))
        print("Layer size RNN: {}\nNumber of layers RNN: {}\nKeep probability: {}".format(
                  self.layer_size, self.n_layers, self.keep_prob))
    



    @lru_cache(maxsize=None)                                            # for PBT!
    def copy_from(self, other_model):
        # This method is used for exploitation. We copy all weights and hyper-parameters
        # from other_model to this model
        my_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope + '/')
        their_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, other_model.name_scope + '/')
        assign_ops = [mine.assign(theirs).op for mine, theirs in zip(my_weights, their_weights)]
        return tf.group(*assign_ops)

        
    def _restore(self, checkpoint_path):
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        for var in self.saver._var_list:
            tensor_name = var.name.split(':')[0]
            if not reader.has_tensor(tensor_name):
                continue
            saved_value = reader.get_tensor(tensor_name)
            resized_value = fit_to_shape(saved_value, var.shape.as_list())
            var.load(resized_value, self.sess)


    def fit_to_shape(array, target_shape):
        source_shape = np.array(array.shape)
        target_shape = np.array(target_shape)

        if len(target_shape) != len(source_shape):
            raise ValueError('Axes must match')

        size_diff = target_shape - source_shape

        if np.all(size_diff == 0):
            return array

        if np.any(size_diff > 0):
            paddings = np.zeros((len(target_shape), 2), dtype=np.int32)
            paddings[:, 1] = np.maximum(size_diff, 0)
            array = np.pad(array, paddings, mode='constant')

        if np.any(size_diff < 0):
            slice_desc = [slice(d) for d in target_shape]
            array = array[slice_desc]

        return array
