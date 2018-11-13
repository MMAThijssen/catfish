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

        tf.set_random_seed(16)
        
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
        self.layer_sizes = [self.layer_size,] * self.n_layers
        self.saving_step = 100
        # self.session = None
        
        self.extra = False
        self.save = True
        self.cell_type = "GRU"
        
        self.model_type = "bi" + self.cell_type + "-RNN-full"
        self.save_info()

        # build network and additionals
        #~ self.build_network(self.rnn_layer())
        self.input_layer()
        self.layer_output = self.network_layer(self.x)
        self.logits = self.output_layer(self.layer_output)
        self.loss = self.compute_loss()
        self.accuracy = self.compute_accuracy()
        self.optimizer = self.optimizer_choice
        if self.save:
            self.model_path = self.set_model_path()
            self.saver = self.set_saver()

              
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
        
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        
        #~ number = 0
        #~ while check_for_dir:
            #~ if os.path.isdir(model_path):
                #~ number += 1
                #~ model_path = model_path.rsplit("_")[0] + "_" + str(number)
            #~ else:
                #~ os.mkdir(model_path)
                #~ check_for_dir = False
        print("\nSaving network checkpoints to", model_path, "\n")
        
        return model_path    
        
    
    def set_saver(self):  
        with tf.name_scope("saver"):
            saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)
                
        return saver
                
        #~ with tf.name_scope("TensorBoard"):            
            #~ print("\nCurrent directory: {}\nUse 'tensorboard --logdir=[cur_dir]'\n".format(cur_dir))
            
            #~ writer = tf.summary.FileWriter(model_path + "/tensorboard")
            #~ writer.add_graph(tf.get_default_graph())
            
            #~ loss_summ = tf.summary.scalar("loss", loss)
            #~ acc_summ = tf.summary.scalar("accuracy", accuracy)
        
            #~ all_summs = tf.summary.merge_all()
            
            
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
        
            
    
    def train_network(self, train_x, train_y, n_epochs):
        with tf.Session() as sess:
            # initialize model variables:
            sess.run(tf.global_variables_initializer())
            print("\nNot yet initialized: ", sess.run(tf.report_uninitialized_variables()), "\n")
            
            # feed training data:
            step = 0
            n_batches = self.set_nbatches(len(train_x))
            
            if self.save:
                saver = self.set_saver()
                
                self.saver.save(sess, self.model_path + "/checkpoints/meta", write_meta_graph=True)
                print("\nSaved meta graph\n")
            
            for epoch in range(1, n_epochs + 1):
                epoch_loss = 0
                epoch_acc = 0          
                #~ random.shuffle(list_batches)                                # is it bad that the full reads are not sequentially fed?
                for batch in range(n_batches):
                    feed_dict = {self.x: train_x[batch * self.batch_size : (batch + 1) * self.batch_size], 
                                 self.y: train_y[batch * self.batch_size: (batch + 1) * self.batch_size], 
                                 self.p_dropout: self.keep_prob}

                    batch_loss, _, batch_acc = sess.run([self.loss, self.optimizer, self.accuracy], feed_dict=feed_dict)
                    epoch_loss += batch_loss
                    epoch_acc += batch_acc      
                    
                    if self.save and step % self.saving_step == 0:      
                        saver.save(sess, self.model_path + "/checkpoints", global_step=step, write_meta_graph=False)            
                    
                    step += 1
                    
                self.loss_history.append(epoch_loss)
                
                #~ dest = open(self.model_type + ".txt", "a")
                print("Epoch {}\t\tLoss: {:.4f}\t\tAccuracy: {:.2%}".format(epoch, epoch_loss / n_batches, epoch_acc / n_batches))
                #~ dest.close()

            if self.save:
                saver.save(sess, self.model_path + "/checkpoints", global_step=step)
                print("\nSaved final checkpoint\n")


    def test_network(self, test_x, test_y, ckpnt="latest"):
        with tf.Session() as sess:
            # initialize model variables:
            sess.run(tf.global_variables_initializer())
            #~ saver = tf.train.import_meta_graph(meta_file)
            saver = self.set_saver()
            if ckpnt == "latest":
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            else:
                saver.restore(sess, ckpnt)
            print("Model {} restored".format(os.path.basename(self.model_path)))
                    
            #~ # get predicted values:
            #~ pred_vals = sess.run(tf.round(self.predictions), feed_dict=feed_dict_test)                 
            #~ pred_vals = np.concatenate(pred_vals).ravel().astype(int)       
        
            pred_vals, confidences = self.predict(sess, test_x)
#            pred_list = [pred_vals, confidences]       # limit to one squiggle: [:5250]
            #~ metrics.generate_heatmap(pred_list, ["homopolymer", "confidence"], "Predictions_{}".format(self.model_id))

            # get testing accuracy:
            feed_dict_test = {self.x: test_x, self.y: test_y, self.p_dropout: self.keep_prob_test}
            test_acc = sess.run(self.accuracy, feed_dict=feed_dict_test)
            print("\nTest accuracy: {:.2%}".format(test_acc))
                    
            # output performance:
            test_labels = test_y.reshape(-1)
            nonhp = np.count_nonzero(pred_vals == 0)
            hp = np.count_nonzero(pred_vals == 1)
            hp_true = np.count_nonzero(test_labels == 1)
            test_precision, test_recall = metrics.precision_recall(test_labels, pred_vals)
            test_f1 = metrics.weighted_f1(test_precision, test_recall, hp_true, hp + nonhp)
            tpr, fpr, roc_auc = metrics.calculate_auc(test_labels, confidences)
            metrics.draw_roc(tpr, fpr, roc_auc, "ROC_{}".format(self.model_id))
            print("Predicted percentage HPs in test set: {:.2%}".format(hp / (nonhp + hp)))
            print("Test precision: {:.2%}\nTest recall: {:.2%}".format(test_precision, test_recall))
            print("Test weighted F1 measure: {0:.4f}".format(float(test_f1)))
            print("AUC: {0:.4f}".format(roc_auc))
            
            #~ metrics.generate_heatmap([pred_vals, confidences, test_labels], 
                                    #~ ["homopolymer", "confidence", "truth"], "Comparison_{}".format(self.model_id))
            
    
    #TODO: if sequence is smaller than needed. Add nonsense values that can be cut later ("dead" value 0?)
    # predict assumes reshaped sequence         
    def predict(self, sess, sequence):        
        # get predicted values:
        feed_dict_pred = {self.x: sequence, self.p_dropout: self.keep_prob_test}
        pred_vals = sess.run(tf.round(self.predictions), feed_dict=feed_dict_pred)                 
        pred_vals = np.concatenate(pred_vals).ravel().astype(int)       # ravel needed?
        
        confidence = sess.run(self.predictions, feed_dict=feed_dict_pred) 
        confidence = np.concatenate(confidence).ravel().astype(float)
        
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
        print(my_weights)
        their_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, other_model.name_scope + '/')
        assign_ops = [mine.assign(theirs).op for mine, theirs in zip(my_weights, their_weights)]
        return tf.group(*assign_ops)
