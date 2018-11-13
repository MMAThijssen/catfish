#!/usr/bin/env python3
import helper_functions
import matplotlib.pyplot as plt
import numpy as np
import os
import reader
import tensorflow as tf

tf.set_random_seed(16)

def metrics(true_labels, predicted_labels):
    """
    Returns precision and recall
    """

    if len(true_labels) != len(predicted_labels):
        print("Len true labels: ", len(true_labels))
        print("Len pred labels: ", len(predicted_labels))
        print("True labels: ", true_labels)
        print("Pred labels: ", predicted_labels)
        raise ValueError("Length of labels to compare is not equal.")
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    
    for i in range(len(true_labels)):
        if predicted_labels[i] == 1:
            #~ print("Predicted label pos at pos {}: ".format(i), predicted_labels[i])
            if true_labels[i] == 1:
                true_pos += 1
            else:
                false_pos += 1
        elif predicted_labels[i] == 0:
            #~ print("Predicted label neg at pos {} ".format(i), predicted_labels[i])
            if true_labels[i] == 0:
                true_neg += 1
            else:
                false_neg += 1

    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0
        print("No true positives detected")
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0
    return(precision, recall)
    

### Hyperparameters ###
batch_size = 64
test_batch_size = 1
max_seq_length = 5250
window = 35
n_outputs = 1
n_classes = 1
n_inputs = 1
learning_rate = 0.01
n_epochs = 10
loss_history = []
layer_size = 32

n_train = 10000
n_test = 1000 #1000 but changed for testing

cat = "trainingreads"
#~ cat = "squiggles"

def set_nbatches(category):
    if category == "squiggles":
        n_batches = max_seq_length // batch_size // window      # 9 (9.375)
    if category == "trainingreads":
        n_batches = n_train // batch_size
        
    return n_batches
    

### Get input ###
def get_input(category):
    #~ db_dir = "/mnt/nexenta/thijs030/data/trainingDB/test3/"
    #~ db_dir = "/mnt/nexenta/thijs030/data/trainingDB/examples2w34/"
    #~ db, squiggles = helper_functions.load_db(db_dir)
    db_dir = "/mnt/nexenta/thijs030/data/trainingDB/training4000w34/"
    db_dir_ts = "/mnt/nexenta/thijs030/data/trainingDB/test858w34/"
    db, squiggles = helper_functions.load_db(db_dir)
    db_ts, squiggles_ts = helper_functions.load_db(db_dir_ts)
    
    # data from squiggles:
    if category == "squiggles":
        data = []
        labels = []
        test_data = []
        test_labels = []
        
        for squig in squiggles:
            data_sq, labels_sq = reader.load_npz(squig)
            
            test_data.append(data_sq[max_seq_length : max_seq_length + batch_size * window])
            test_labels.append(labels_sq[max_seq_length : max_seq_length + batch_size * window])
            data.append(data_sq[: max_seq_length])
            labels.append(labels_sq[: max_seq_length])
            

    # data from TrainingReads:
    if category == "trainingreads":
        print("\nRetrieving balanced training set")
        data, labels = db.get_training_set(n_train)
        print("\nRetrieving balanced test set")
        test_data, test_labels = db_ts.get_training_set(n_test)
        
    data = np.concatenate(data).ravel()
    labels = np.concatenate(labels).ravel()
    test_data = np.concatenate(test_data).ravel()
    test_labels = np.concatenate(test_labels).ravel()
    
    return(data, labels, test_data, test_labels)
    

def reshape_input(data, labels, test_data, test_labels):
    #~ print("Before reshaping data: ", data.shape)
    #~ print("Before reshaping labels: ", labels.shape)
    data = data.reshape(-1, window, n_inputs)
    labels = labels.reshape(-1, window, n_outputs)
    test_data = test_data.reshape(-1, window, n_inputs)
    test_labels = test_labels.reshape(-1, window, n_outputs)
    #~ print("After reshaping data : ", data.shape)
    #~ print("After reshaping labels: ", labels.shape)
    
    return(data, labels, test_data, test_labels)
    
#~ def write_to_tensorboard(log_dir, folder="tensorboard"):
    #~ writer = tf.summary.FileWriter(log_dir + "/" + folder)
    #~ writer.add_graph(tf.get_default_graph())
    
def _to_tensorboard():
    loss_summ = tf.summary.scalar("loss", loss)
    acc_summ = tf.summary.scalar("accuracy", accuracy)

    all_summs = tf.summary.merge_all()
    return all_summs

    #~ writer.add_summary(summary, global_step=global_step)
# TODO: save summaries is different subfolders eg. graph/lr1.0 and graph/lr0.5 > write python script for this
    #~ with tf.name_scope("summaries"):
        #~ tf.summary.scalar("loss", loss)
        #~ tf.summary.scalar("accuracy", accuracy)
        #~ tf.summary.histogram("loss histogram", loss_history)
        #~ summary_op = tf.summary.merge_all()
        #~ return summary_op

n_batches = set_nbatches(category=cat)
print("Number of batches: ", n_batches)
data, labels, test_data, test_labels = get_input(category=cat)    
train_x, train_y, test_x, test_y = reshape_input(data, labels, test_data, test_labels)


### Assemble graph ###
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

with tf.name_scope("data"):
    x = tf.placeholder(tf.float32, shape=[None, window, n_inputs])
    y = tf.placeholder(tf.float32, shape=[None, window, n_outputs])

with tf.name_scope("logistic_regression"):
    w = tf.get_variable("w", shape=[window, n_outputs])
    b = tf.get_variable("b", shape=[n_outputs], initializer=tf.zeros_initializer)
    logits = tf.add(tf.multiply(x, w), b) 

print("Logits get shape: ", logits.get_shape(), "\n")

with tf.name_scope("accuracy"):
    predictions = tf.nn.sigmoid(logits)
    correct = tf.equal(tf.round(predictions), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("loss"):
    loss_ce = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
    loss = tf.reduce_mean(loss_ce)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope("saver"):
    saver = tf.train.Saver()


with tf.name_scope("TensorBoard"):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print("\nCurrent directory: {}\nUse 'tensorboard --logdir=[cur_dir]'\n".format(cur_dir))
    
    writer = tf.summary.FileWriter(cur_dir + "/tensorboard")
    writer.add_graph(tf.get_default_graph())
    
    loss_summ = tf.summary.scalar("loss", loss)
    acc_summ = tf.summary.scalar("accuracy", accuracy)

    all_summs = tf.summary.merge_all()
    
    
### Execute computation ###
with tf.Session() as sess:
    # initialize model variables:
    sess.run(tf.global_variables_initializer())
    print("\nNot yet initialized: ", sess.run(tf.report_uninitialized_variables()), "\n")
    
    #~ summary_op = write_to_tensorboard(cur_dir)
    
    # feed training data:
    step = 0
    
    saver.save(sess, cur_dir + "/logistic_regression/checkpoints", write_meta_graph=True)  # saves meta graph
    print("Saved meta graph")
    
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        for batch in range(n_batches):
            step += 1
            feed_dict = {x: train_x[batch * batch_size : (batch + 1) * batch_size], y: train_y[batch * batch_size: (batch + 1) * batch_size]}
            batch_loss, _, batch_acc, summ_all = sess.run([loss, train_op, accuracy, all_summs], feed_dict=feed_dict)
            epoch_loss += batch_loss
            epoch_acc += batch_acc      
            
            writer.add_summary(summ_all, step)
            
            if step % 10 == 0:      # change later! - is for testing (in ex. they use 1000)
                saver.save(sess, cur_dir + "/logistic_regression/checkpoints", global_step=epoch*1000000+step, write_meta_graph=False)
            
        loss_history.append(epoch_loss)
        
        print("Epoch {}\t\tLoss: {:.4f}\t\tAccuracy: {:.2%}".format(epoch, epoch_loss / n_batches, epoch_acc / n_batches))
        #~ print("Epoch {}\t\tLoss: {:.4f}\t\tAccuracy: {:.2%}\t\tLAST TRAINING EXAMPLE".format(epoch, batch_loss, batch_acc))
        
    # final save
    saver.save(sess, cur_dir + "/logistic_regression/checkpoints", global_step=step)
    print("Saved final checkpoint")

    feed_dict_test = {x: test_x, y: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Test accuracy: {:.2%}".format(test_acc))

    # get predicted values:
    pred_vals = sess.run(tf.round(predictions), feed_dict=feed_dict_test)       
    #~ pred_vals = pred_vals[0].astype(int)            
    pred_vals = np.concatenate(pred_vals).ravel().astype(int)       # are all 0
    #~ print(len(pred_vals))
    nonhp = np.count_nonzero(pred_vals == 0)
    hp = np.count_nonzero(pred_vals == 1)
    hp_true = np.count_nonzero(test_labels == 1)
    print("\nPredicted percentage HPs in test set: {:.2%}".format(hp / (nonhp + hp)))
    test_precision, test_recall = metrics(test_labels, pred_vals)
    print("Test precision: {:.2%}\nTest recall: {:.2%}".format(test_precision, test_recall))

## inspo: https://www.kaggle.com/autuanliuyc/logistic-regression-with-tensorflow
