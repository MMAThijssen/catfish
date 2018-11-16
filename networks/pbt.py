#!/usr/bin/env python3
from functools import lru_cache
import numpy as np
#~ from resnet_class import ResNetRNN
import ray
from rnn_class import RNN
from sys import argv
import tensorflow as tf
import train


# from: http://louiskirsch.com/ai/population-based-training
#~ n_batches = n_train // batch_size

POPULATION_SIZE = 6
BEST_THRES = 3
WORST_THRES = 3
POPULATION_STEPS = 5             # was 500 - I use n_epochs now
ITERATIONS = 5           # was 100 - I use random number now
#~ accuracy_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
#~ l1_scale_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
#~ best_accuracy_hist = np.zeros((POPULATION_STEPS,))
#~ best_l1_scale_hist = np.zeros((POPULATION_STEPS,))

def initialize_ray(gpu=False):
    if not gpu:
        ray.init()
    else:
        ray.init(redis_address="HEAD_HOSTNAME:6379")        # example - pick machine to be head node 


def pbt(db_dir, training_nr, training_type="trainingreads"):
    POPULATION_SIZE = 6
    BEST_THRES = 3
    WORST_THRES = 3
    POPULATION_STEPS = 5             # was 500 - I use n_epochs now
    ITERATIONS = 5  

    # 2. Train                                  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("\nNot yet initialized: ", sess.run(tf.report_uninitialized_variables()), "\n")
        
        step = 0

        for i in range(1, POPULATION_STEPS + 1):
        
            # Training                                  # optimize is using the optimizer (Adam / RMSProp)
            #~ for _ in range(ITERATIONS):
            n_batches = training_nr // 64           # 64 is batch size
            for batch in range(n_batches):
                initial_to_zero = [0,] * POPULATION_SIZE
                model_ids = range(POPULATION_SIZE)
                epoch_loss = dict(zip(model_ids, initial_to_zero))
                epoch_acc = dict(zip(model_ids, initial_to_zero))

                for m in models:
                    feed_dict = {m.x: train_x[batch * m.batch_size : (batch + 1) * m.batch_size], 
                                m.y: train_y[batch * m.batch_size: (batch + 1) * m.batch_size], 
                                m.p_dropout: m.keep_prob}
                    sess.run(m.optimizer, feed_dict=feed_dict)
                    batch_loss = sess.run(m.loss, feed_dict=feed_dict)
                    batch_acc = sess.run(m.accuracy, feed_dict=feed_dict)
                    epoch_loss[m.model_id] = batch_loss + epoch_loss[m.model_id]
                    epoch_acc[m.model_id] = batch_acc + epoch_acc[m.model_id]
                
                step += 1
                if step % ITERATIONS == 0:
                    check_acc = {key: value / n_batches for (key, value) in epoch_acc.items()}
                    models.sort(key=lambda m: check_acc[m.model_id], reverse=True)
                    print(models)
                    
                    # Copy best
                    sess.run([m.copy_from(models[0]) for m in models[-WORST_THRES:]])
            
                    # Perturb others                            # check how to implement - random change one value
                    explore_values = explore()      # works for now because explore has more value than best_thres size
                    for m in models[BEST_THRES:]:
                        choice = random.choice(explore_values)
                        print("Model ID: {}\tExplored: {} {}".format(m.model_id, choice[0], choice[1]))
                        sess.run(m.setattr(choice[0], choice[1]))
                        #~ sess.run([m.random.choice(explore_values) for m in models[BEST_THRES:]])
            
            print("\n\n\nEPOCH: {}\n".format(i))
            for m in models:
                print("Model ID: {}\t\tLoss: {:.4f}\t\tAccuracy: {:.2%}".format(m.model_id, epoch_loss[m.model_id] / n_batches, epoch_acc[m.model_id] / n_batches))
             


def explore():
    hps = ["learning_rate", "optimizer_choice", "layer_size", "n_layers", "batch_size", "keep_prob"]
    random_hps = generate_random_hyperparameters()
    #~ hp_int = random.choice(range(len(random_hps)))
    return hps, random_hps
    
    

def create_model(model_id):                        # werkt
    with tf.variable_scope(None, 'model'):
        lr, opt, l_size, n_layers, batch_size, dropout = generate_random_hyperparameters()
            #~ return ResNetRNN(**kwargs)       # change model to desired model
        return RNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=64, keep_prob=dropout)    # batch_size is 64 for now
                    
  

def generate_random_hyperparameters():          # werkt
    """
    Returns: learning_rate, optimizer, layer_size, n_layers, batch_size, dropout
    """
    # HYPERPARAMETERS #
    learning_rate_min = -4      # 10 ^ -4
    learning_rate_max = 0

    optimizer_list = ["Adam", "RMSProp"]

    layer_size_list = [16, 32, 64, 128, 256]

    n_layers_min = 1
    n_layers_max = 5

    #~ batch_size_list = [32, 64, 128, 256]      # cannot vary batch_size

    dropout_min = 0.2
    dropout_max = 0.8
    
    # pick random hyperparameter:
    learning_rate = 10 ** np.random.randint(learning_rate_min, learning_rate_max)
    optimizer = np.random.choice(optimizer_list)
    layer_size = np.random.choice(layer_size_list)
    n_layers = np.random.randint(n_layers_min, n_layers_max)
    #~ batch_size = np.random.choice(batch_size_list)
    dropout = round(np.random.uniform(dropout_min, dropout_max), 1)
    
    #~ return [("learning_rate", learning_rate), ("optimizer", optimizer), 
            #~ layer_size, n_layers, batch_size, ("keep_prob", dropout)]
    return learning_rate, optimizer, layer_size, n_layers, dropout
    

if __name__ == "__main__":
    #~ batch_size = 64
    # 0. Get input
    if not len(argv) == 8:
        raise ValueError("The following arguments should be provided:\n\t-number of models\n" +
                         "\t-networktype\t-trainingdb\n\t-nr trainingreads\n\t-nr epochs\n" +
                         "\t-valdb\n\t-nr validationreads")
    
    POPULATION_SIZE = int(argv[1])
    network_type = argv[2]
    
    db_dir_train = argv[3]
    training_nr = int(argv[4])
    
    n_epochs = argv[5]
    
    db_dir_val = argv[6]
    max_seq_length = int(argv[7])
    
    
    # 1. Create different models
    models = [create_model(i, network_type) for i in range(POPULATION_SIZE)]    

    # 2. Train models
    print("Loading training database..")
    db_train = helper_functions.load_db(db_dir_train)
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        train(m, db_train, training_nr, n_epochs)
    
    
    #3. Assess performance on validation set
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    for m in models:
        print("------------------------------MODEL {}------------------------------".format(m.model_id))
        network = m.restore()
        validate(network, squiggles, max_seq_length)
