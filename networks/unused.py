def infer_class_from_signal2(fast5_file, model, hdf_path="Analyses/Basecall_1D_000", window_size=35):
    """
    Infers classes from raw MinION signal in FAST5.
    
    Args:
        fast5_file -- str, path to FAST5 file
        model -- RNN object, network model
        hdf_path -- str, path in FAST5 leading to signal
        network_type -- str, type of network [default: ResNetRNN]
        window_size -- int, size of window used to train network
    
    Returns: list of classes
    """           
    # open FAST5
    if not os.path.exists(fast5_file):
        raise ValueError("path to FAST5 is not correct.")
    with h5py.File(fast5_file, "r") as fast5:
        # process signal
        raw = process_signal(fast5)
        #~ print("Length of raw signal: ", len(raw))
        #~ raw = raw[: 36]                                                         # VERANDEREN!                                              
        #~ print(len(raw))
        
    # pad if needed
    if not (len(raw) / window_size).is_integer():
        # pad
        padding_size = window_size - (len(raw) - (len(raw) // window_size * window_size))
        padding = np.array(padding_size * [0])
        raw = np.hstack((raw, padding))
    
    # take per 35 from raw and predict scores
    n_input = 1
    n_batches = len(raw) // window_size
    raw_in = reshape_input(raw, window_size, n_input)
    scores = model.infer(raw_in)
    
    # cut padding
    scores = scores[: -padding_size]
    
    return scores

def infer2(fast5_dir, output_file, threshold=0.5, network_path="/mnt/scratch/thijs030/actualnetworks/ResNet-RNN_14", network_type="ResNetRNN"):
    """
    Infers scores for every raw signal in directory.
    Can be adjusted easily to infer classes for given threshold.
    
    Args:
        fast5_dir -- str, path to directory containing FAST5 files
        output_file -- str, name of output file to write predictions to
        threshold -- float, threshold for assigning classes
        network_path -- str, path to network to be loaded
        network_type -- str, type of network to be loaded
        
    Returns: None
    """
    # load model
    model = load_network(network_type, network_path)
    
    # infer for every file in dir
    abs_fast5dir = os.path.abspath(fast5_dir)
    input_files = os.listdir(abs_fast5dir)
    with open(output_file, "w") as dest:
        #~ scores_all = [infer_class_from_signal("{}/{}".format(abs_fast5dir, fast5_file), model) for fast5_file in input_files]
        #~ print(len(scores_all))
        #~ print(len(input_files) * 36)
        #~ [dest.write("{}, ".format(s)) for scores in scores_all for s in scores]
        #~ dest.write("\n")
        for fast5_file in input_files:
            scores = infer_class_from_signal("{}/{}".format(abs_fast5dir, fast5_file), model)
            #~ classes = class_from_threshold(scores, threshold)
            dest.write("{}\n".format(fast5_file))
            [dest.write("{}, ".format(s)) for s in scores]
            dest.write("\n")
    
    print("Finished inference.")

    #~ scores = np.zeros(shape=(n_batches, window_size))
    #~ for n in range(n_batches):
        #~ raw_in = raw[n * window_size : (n + 1) * window_size]
        

    bases, new = get_base_new_signal(read)
    bases = bases[start: start + length]
    new = new[start: start + length]
    print(len(bases))
    print(len(new))
    
    # 1. Do majority voting per event
    for n in range(len(new)):
        if new[n] == "n":
            first_n = n
        elif new[n] == "n":
            second_n = n

                 #~ if not found_start and summed_len <= start:
                    #~ start_len = summed_len
                #~ summed_len += event_lengths[n]                   
                #~ if not found_start and summed_len >= start:
                    #~ found_start = True
                    #~ start_event = n                
                #~ if found_start and summed_len >= (start + length):
                    #~ final_event = n 
                    #~ break    
            
            #~ event_lengths = event_lengths[start_event: final_event + 1]
            #~ event_bases = event_bases[start_event: final_event + 1]
# correctiondef get_bases_events(fast5, start=30000, length=4970, use_tombo=True):
    """
    Returns events and bases measured in certain stretch.
    
    Args:
        fast5 - str, path to FAST5
        start -- int, start point in read
        length -- int, length to correct output / length of read from start
        use_tombo -- bool, use of corrected or uncorrected reads [default: True]
                
    Returns: bases, events
    """
    with h5py.File(fast5, "r") as hdf:
        hdf_path = "Analyses/RawGenomeCorrected_000/"
        hdf_events_path = '{}BaseCalled_template/Events'.format(hdf_path)
        # get list of event lengths:
        event_lengths = hdf[hdf_events_path]["length"]                         # indicates number of measurements belong to base/event
        if use_tombo:
            # get list of base sequence:
            event_bases = hdf[hdf_events_path]["base"].astype(str)
            
            # start at correct point:
            summed_len = 0
            found_start = False
            for n in range(len(event_lengths)):
                summed_len += event_lengths[n]
                if not found_start and summed_len >= start:
                    found_start = True
                    start_event = n
                if found_start and summed_len >= (start + length):
                    final_event = n 
                    break    
            
            event_lengths = event_lengths[start_event: final_event + 1]
            event_bases = event_bases[start_event: final_event + 1]
            
        else:
            raise Exception("Not yet implemented for uncorrected reads")
    
    return event_bases, event_lengths

## TOOL ##
    #~ scores = np.zeros(shape=(n_batches, window_size))
    #~ for n in range(n_batches):
        #~ raw_in = raw[n * window_size : (n + 1) * window_size]

# process_tp
                    #~ for hp in true_hp.values():
                        #~ state, estimate, inter = check_true_hp(hp, predicted_labels)  
                        #~ states.append(state)
                        #~ estimates.append(estimates)
                        #~ inters.append(inters) 

                #~ print(base_dict)
                # using basenew file     # SLOWER.. 0m19.006s
                #~ base = list_basenew(base_file, read_name, "bases")
                #~ new = list_basenew(base_file, read_name,  "new")
                #~ base_dict = {k: get_bases(base, new, predicted_dict[k][0], predicted_dict[k][1]) for k in predicted_dict}
                #~ print(base[15:64])
                #~ print(base_dict)

# for validation
        # DO NOT DO THIS NOW; CAN DO THIS LATER BY RESTORING (every 100 000 steps)
        #~ # After each half epoch validate     if steps % (n_batches / 2) == 0:
        #~ tv1 = datetime.datetime.now()
        #~ if steps % 100000 == 0:
            #~ validate(squiggles, max_seq_length)
            #~ tv2 = datetime.datetime.now()
            #~ print("Validated in {}".format(tv2 - tv1)) 

# FOR PBT:
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


    #~ batch_size = 128
    #~ learning_rate = 0.01
    #~ n_layers = 3
    #~ layer_size = 64
    #~ keep_prob = 0.8     
    #~ optimizer_choice = "Adam"
    #~ layer_size_res = 32
    #~ n_layers_res = 2

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
