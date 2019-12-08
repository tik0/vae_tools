#!/usr/bin/python

from tensorflow import keras
from tensorflow.keras import backend as K
import numpy.random
import tensorflow
import random


def set_seed(seed = 0):
    # Set random seeds (https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)
    numpy.random.seed(seed)
    try:
        tensorflow.compat.v1.set_random_seed(seed)
    except: # backwards compat
        tensorflow.set_random_seed(seed)
    random.seed(seed)

class Sampling():
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def randn(self, args):
        # Backwards compatibility
        if hasattr(self, 'latent_dim') and hasattr(self, 'z_dim'):
            if self.latent_dim != self.z_dim:
                raise Exception("Redundant definition of latent_dim and z_dim")
        elif hasattr(self, 'z_dim'):
            pass
        elif hasattr(self, 'latent_dim'):
            self.z_dim = self.latent_dim
        else:
            raise Exception("No latent dimensionality given")
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.z_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon
