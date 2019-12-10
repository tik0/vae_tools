#!/usr/bin/python

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda
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
    '''
    Holds various sampling techniques
    '''
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

class RandN(Sampling):
    """
    Standard Gaussian sampling with linear mean and logvar layer as described by Kingma et al.

    Attributes
    ----------
    z_dim : int
        latent dimension
    sampling_layer : tensorflow.keras.layers
        some arbitrary layer (e.g. Lambda) that is used for sampling
    """
    def __init__(self, z_dim):
        """The init function

        :param z_dim: latent dimension
        :type z_dim: int
        """
        self.z_dim = z_dim
        self.sampling_layer = Lambda(self.randn, output_shape=(self.z_dim,), name='sample')

    def get_sampling(self, encoder_outputs_powerset: list) -> (list, list):
        """Returns the sampling and statistical parameter and sampling layers

        :param encoder_outputs_powerset: encoder powerset which will be extended
        :type encoder_outputs_powerset: list
        :return: tuple (Z_powerset, Z_layers_powerset)
            WHERE
            list Z_powerset list of sets of sampling layers
            list Z_layers_powerset list of sets of statistical parameter layers
        """
        Z_powerset = []
        Z_layers_powerset = []
        for encoder_output, idx_set in zip(encoder_outputs_powerset, range(len(encoder_outputs_powerset))):
            Z_layers_powerset.append({})  # add an empty dict which can be filled with statistic layers
            Z_layers_powerset[-1]["mean"] = Dense(self.z_dim, name="mean_" + str(idx_set))(encoder_output)
            Z_layers_powerset[-1]["logvar"] = Dense(self.z_dim, name="logvar_" + str(idx_set))(encoder_output)
            Z_powerset.append(self.sampling_layer([Z_layers_powerset[-1]["mean"], Z_layers_powerset[-1]["logvar"]]))
        return Z_powerset, Z_layers_powerset
