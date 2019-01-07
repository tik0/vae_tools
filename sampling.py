#!/usr/bin/python

from keras import backend as K

class Sampling():

    def randn(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.z_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon
