import numpy as np
import ray.tune as tune
from resnet_class import ResNetRNN
from rnn_class import RNN
import tensorflow as tf

tune.register_trainable('MyTrainable', MyTrainable)
train_spec = {
    'run': 'MyTrainable',
    # Specify the number of CPU cores and GPUs each trial requires
    'trial_resources': {'cpu': 1, 'gpu': 1},
    'stop': {'timesteps_total': 20000},
    # All your hyperparameters (variable and static ones)
    'config': {
        'batch_size': 20,
        'units': 100,
        'l1_scale': lambda cfg: return np.random.uniform(1e-3, 1e-5),
        'learning_rate': tune.random_search([1e-3, 1e-4])
        ...
        },
  # Number of trials
  'repeat': 4
}

pbt = PopulationBasedTraining(
    time_attr='training_iteration',
    reward_attr='mean_loss',
    perturbation_interval=1,
    hyperparam_mutations={
        'l1_scale': lambda: return np.random.uniform(1e-3, 1e-5),
        'learning_rate': [1e-2, 1e-3, 1e-4]
    }
)
tune.run_experiments({'population_based_training': train_spec}, scheduler=pbt)


class MyTrainable(Trainable):
    def _setup(self):
        # Load your data
        self.data = ...
        # Setup your tensorflow model
        # Hyperparameters for this trial can be accessed in dictionary self.config
        self.model = Model(self.data, hyperparameters=self.config)
        # To save and restore your model
        self.saver = tf.train.Saver()
        # Start a tensorflow session
        self.sess = tf.Session()

    def _train(self):
        # Run your training op for n iterations
        for _ in range(n):
          self.sess.run(self.model.training_op)

        # Report a performance metric to be used in your hyperparameter search
        validation_loss = self.sess.run(self.model.validation_loss)
        return tune.TrainingResult(timesteps_this_iter=n, mean_loss=validation_loss)

    def _stop(self):
        self.sess.close()

  # This function will be called if a population member
  # is good enough to be exploited
    def _save(self, checkpoint_dir):
        path = checkpoint_dir + '/save'
        return self.saver.save(self.sess, path, global_step=self._timesteps_total)

  # Population members that perform very well will be
  # exploited (restored) from their checkpoint
    def _restore(self, checkpoint_path):
        return self.saver.restore(self.sess, checkpoint_path)

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
