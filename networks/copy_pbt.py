#!/usr/bin/env python3
from functools import lru_cache
import numpy as np
#~ from resnet_class import ResNetRNN
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
        lr, opt, l_size, n_layers, dropout = generate_random_hyperparameters()
            #~ return ResNetRNN(**kwargs)       # change model to desired model
        return RNN(model_id, learning_rate=lr, optimizer_choice=opt, n_layers=n_layers,
                    layer_size=l_size, batch_size=64, keep_prob=dropout)    # batch_size is 64 for now

                    
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


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
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided:\n\t-number of models\n" +
                         "\t-trainingdb\n\t-nr trainingreads\n\t-nr epochs\n\t-valdb\n\t-nr validationreads")
    
    # 1. Create different models
    THRES = 1
    POPULATION_SIZE = int(argv[1])
    models = [create_model(i) for i in range(POPULATION_SIZE)]
    
    db_dir = argv[2]
    training_nr = int(argv[3])
    
    db_dir_val = argv[5]
    val_nr = 5250                   # int(argv[6])
    
    train_x, train_y = train.retrieve_set(db_dir, training_nr, "trainingreads")
    val_x, val_y = train.retrieve_set(db_dir_val, val_nr, "squiggles")

    print("Training on : {} windows".format(training_nr))
    print("Validating on: {} squiggles".format(val_nr))

    
    POPULATION_STEPS = int(argv[4])
    for p in range(POPULATION_STEPS):
        # 2. Train models
        n_epochs = 1        # copy and explore per epoch
        training = [m.train_network(train_x, train_y, n_epochs) for m in models]
        
        #3. Assess performance on validation set (squiggles)
        # pick a certain number of NEW squiggles per time ?
        
        f1_dict = {m.model_id: m.test_network(val_x, val_y) for m in models}
        print(models)
        models.sort(key=lambda m: f1_dict[m.model_id], reverse=True)
        print(models)
            
        #4. Copy best models to worst models
        for t in range(THRES):
            threshold = THRES
            while threshold > 0:
                # get model id
                current_model = models[-threshold].model_id # -3, -2, -1
                better_model = models[threshold - 1] # 2, 1, 0
                # restore better model
                current_model.saver.restore(current_model.sess, tf.train.latest_checkpoint(better_model.model_path)) # does this work or copies over?
                # set model id
                
                threshold += 1
        
                #5. Explore a new parameter
                is_substituted = False
                choice = random.choice(explore())
                # Check that it is not the same as the current one (or previously existing one)
                while not is_substituted:
                    if not current_model.getattr(choice[0]) == choice[1]:
                        current_model.setattr(choice[0], choice[1])
                        print("Model ID: {}\tExplored: {} {}".format(current_model.model_id, choice[0], choice[1]))
                        is_substituted = True
                    # TODO: initialize of not yet initialized values are present
                        initialize_uninitialized(current_model.sess)



        # # restore other model as own model
        # # only thing to do so is keeping the model id
            
        # sess.run([m.copy_from(models[0]) for m in models[-WORST_THRES:]])
        
        # #5. Explore a new parameter
        # for m in models[BEST_THRES:]:
        #     is_substituted = False
        #     choice = random.choice(explore())
        #     # Check that it is not the same as the current one (or previously existing one)
        #     while not is_substituted:
        #         if not m.getattr(choice[0]) == choice[1]:
        #             m.setattr(choice[0], choice[1])
        #             print("Model ID: {}\tExplored: {} {}".format(m.model_id, choice[0], choice[1]))
        #             is_substituted = True
        #             # TODO: initialize of not yet initialized values are present