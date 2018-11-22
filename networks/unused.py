    #~ #TODO: if sequence is smaller than needed. Add nonsense values that can be cut later ("dead" value 0?)
    #~ # predict assumes reshaped sequence         
    #~ def predict(self, sequence):        
        #~ # get predicted values:
        #~ feed_dict_pred = {self.x: sequence, self.p_dropout: self.keep_prob_test}
        #~ pred_vals = self.sess.run(tf.round(self.predictions), feed_dict=feed_dict_pred)                  
        #~ pred_vals = np.reshape(pred_vals, (-1)).astype(int)    
        
        #~ confidence = self.sess.run(self.predictions, feed_dict=feed_dict_pred) 
        #~ confidence = np.reshape(confidence, (-1)).astype(float)
        
        #~ return pred_vals, confidence
        
# train_validate
    #~ predicted = []
    #~ probabilities = []
    #~ tp = 0
    #~ fp = 0
    #~ tn = 0
    #~ fn = 0
            
            #~ sgl_predicted, slg_probabilities = network.test_network(set_x)
            #~ sgl_auc, sgl_tp, sgl_fp, sgl_tn, sgl_fn, true_count = network.test_network(set_x, set_y)
            #~ sgl_acc, sgl_auc, sgl_tp, sgl_fp, sgl_tn, sgl_fn, true_count = network.test_network(set_x, set_y)
            #~ accuracy += sgl_acc
            #~ roc_auc += sgl_auc
            #~ predicted.append(sgl_predicted)
            #~ probabilities.append(sgl_probabilities)
            #~ tp += sgl_tp
            #~ fp += sgl_fp
            #~ tn += sgl_tn
            #~ fn += sgl_fn
            
                #~ precision, recall = metrics.precision_recall(tp, fp, fn)
    #~ f1 = metrics.weighted_f1(precision, recall, true_count, valid_reads * max_seq_length)
    #~ accuracy = metrics.calculate_accuracy(tp, fp, tn, fn)
    #~ tpr, fpr, roc_auc = metrics.calculate_auc(test_labels, confidences)
    #~ metrics.draw_roc(tpr, fpr, roc_auc, "ROC_{}".format(self.model_id))

#~ BACKUP
                #~ sess.run([m.optimizer for m in models],
                            #~ feed_dict=[{m.x: train_x[batch * m.batch_size : (batch + 1) * m.batch_size], 
                             #~ m.y: train_y[batch * m.batch_size: (batch + 1) * m.batch_size], 
                            #~ m.p_dropout: m.keep_prob} for m in models])
                #~ batch_loss = sess.run({m: m.loss for m in models}, 
                            #~ feed_dict=[{m.x: train_x[batch * self.batch_size : (batch + 1) * m.batch_size], 
                             #~ m.y: train_y[batch * m.batch_size: (batch + 1) * m.batch_size], 
                             #~ m.p_dropout: m.keep_prob} for m in models])
                #~ batch_acc = sess.run({m: m.accuracy for m in models}, 
                            #~ feed_dict=[{m.x: train_x[batch * m.batch_size : (batch + 1) * m.batch_size], 
                             #~ m.y: train_y[batch * m.batch_size: (batch + 1) * m.batch_size], 
                             #~ m.p_dropout: m.keep_prob} for m in models])
                #~ epoch_loss = {m: batch_loss[m] + epoch_loss[m] for m in epoch_loss.keys()}
                #~ epoch_acc = {m: batch_acc[m] + epoch_loss[m] for m in epoch_acc.keys()}
                
                # got type error: must be dict, feed_dict

# from MetaFlow
def save(self):
    # This function is usually common to all your models, Here is an example:
    global_step_t = tf.train.get_global_step(self.graph)
    global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
    if self.config['debug']:
        print('Saving to %s with global_step %d' % (self.result_dir, global_step))
    self.saver.save(self.sess, self.result_dir + '/agent-ep_' + str(episode_id), global_step)

    # I always keep the configuration that
    if not os.path.isfile(self.result_dir + '/config.json'):
        config = self.config
        if 'phi' in config:
            del config['phi']
        with open(self.result_dir + '/config.json', 'w') as f:
            json.dump(self.config, f)
            

def add_running():
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary, i)
    print('Adding run metadata for', i)



## On reshaping
print("Before reshaping: ", data.shape)
data = data.reshape(batch_size, seq_length, window)
labels = labels.reshape(batch_size, seq_length, n_outputs)
test_data = test_data.reshape(test_batch_size, seq_length, window)
test_labels = test_labels.reshape(test_batch_size, seq_length, n_outputs)
print("After reshaping: ", data.shape)

#~ def write_to_tensorboard(log_dir, folder="tensorboard"):
    #~ writer = tf.summary.FileWriter(log_dir + "/" + folder)
    #~ writer.add_graph(tf.get_default_graph())
    
#~ def _to_tensorboard():
    #~ loss_summ = tf.summary.scalar("loss", loss)
    #~ acc_summ = tf.summary.scalar("accuracy", accuracy)

    #~ all_summs = tf.summary.merge_all()
    #~ return all_summs

    #~ writer.add_summary(summary, global_step=global_step)
# TODO: save summaries is different subfolders eg. graph/lr1.0 and graph/lr0.5 > write python script for this
    #~ with tf.name_scope("summaries"):
        #~ tf.summary.scalar("loss", loss)
        #~ tf.summary.scalar("accuracy", accuracy)
        #~ tf.summary.histogram("loss histogram", loss_history)
        #~ summary_op = tf.summary.merge_all()
        #~ return summary_op
