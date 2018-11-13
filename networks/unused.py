#   temp unused

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
