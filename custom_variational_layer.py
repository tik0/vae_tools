#!/usr/bin/python

import numpy as np
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
import keras

# Custom loss layer
class CameraRG(Layer):
    def __init__(self, Dx, beta = 1.0):
        self.Dx = Dx
        self.beta = beta
        self.is_placeholder = True
        super(CameraRG, self).__init__()

    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = self.Dx * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + self.beta * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

