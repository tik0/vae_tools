#!/usr/bin/python

import tensorflow as tf
# tf.enable_eager_execution()
#import keras
#from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
# import keras

# KL-divergence between two Gaussian
def kl_loss(mean1, mean2, log_var1, log_var2):
    '''kl_loss 
    Calculates D_KL(N(mean1, sigma1)||N(mean2, sigma2))
    mean*: is the mean value of each Gaussian
    log_var*: is the logarithm of variance (log(sigma^2)) of each gaussian
    '''
    return - .5 * (1 + log_var1 - log_var2 - ((K.exp(log_var1) + K.square(mean1 - mean2)) / K.exp(log_var2)))


# KL-divergence between an abitrary Gaussian and the normal distribution
def kl_loss_n(mean, log_var):
    '''kl_loss 
    Calculates D_KL(N(mean, sigma)||N(0, 1))
    mean: is the mean value
    log_var: is the logarithm of variance (log(sigma^2))
    '''
    return - .5 * (1 + log_var - K.square(mean) - K.exp(log_var))


# Custom loss layer for vanilla VAE
class VaeLoss(Layer):
    def __init__(self, original_dim, beta = 1.0, reconstruction_mse = True):
        self.original_dim = original_dim
        self.beta = beta
        self.reconstruction_mse = reconstruction_mse
        self.counter = np.int(1)
        self.is_placeholder = True
        super(VaeLoss, self).__init__()

    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        if self.reconstruction_mse:
            xent_loss = self.original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        else:
            xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = K.sum(kl_loss_n(z_mean, z_log_var), axis=-1)
        return K.mean(xent_loss + self.beta * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

# Custom loss layer for multi-modal VAE
class MmVaeLoss(Layer):
    def __init__(self, original_dim_x, original_dim_w, alpha = 1.0, beta_shared = 1.0, beta_uni = 1.0, gamma = 1.0, reconstruction_mse = True):
        self.original_dim_x = original_dim_x
        self.original_dim_w = original_dim_w
        self.alpha = alpha
        self.beta_shared = beta_shared
        self.beta_uni = beta_uni
        self.gamma = gamma
        self.reconstruction_mse = reconstruction_mse
        self.is_placeholder = True
        super(MmVaeLoss, self).__init__()
    
    def vae_loss(self, x, w, xx, ww, x_decoded_mean, w_decoded_mean, z_mean_shared, z_log_var_shared, z_mean_x, z_mean_w, z_log_var_x, z_log_var_w, x_decoded_mean_x, w_decoded_mean_w):
        kl_loss_x_xw = K.sum(kl_loss(z_mean_shared, z_mean_x, z_log_var_shared, z_log_var_x), axis=-1);
        kl_loss_w_xw = K.sum(kl_loss(z_mean_shared, z_mean_w, z_log_var_shared, z_log_var_w), axis=-1);
        if self.reconstruction_mse:
            xent_loss_x_shared = self.original_dim_x * metrics.mean_squared_error(x, x_decoded_mean)
            xent_loss_w_shared = self.original_dim_w * metrics.mean_squared_error(w, w_decoded_mean)
            xent_loss_x = self.original_dim_x * metrics.mean_squared_error(xx, x_decoded_mean_x)
            xent_loss_w = self.original_dim_w * metrics.mean_squared_error(ww, w_decoded_mean_w)
        else:
            xent_loss_x_shared = self.original_dim_x * metrics.binary_crossentropy(x, x_decoded_mean)
            xent_loss_w_shared = self.original_dim_w * metrics.binary_crossentropy(w, w_decoded_mean)
            xent_loss_x = self.original_dim_x * metrics.binary_crossentropy(xx, x_decoded_mean_x)
            xent_loss_w = self.original_dim_w * metrics.binary_crossentropy(ww, w_decoded_mean_w)
        kl_loss_x_prior = K.sum(kl_loss_n(z_mean_x, z_log_var_x), axis=-1)
        kl_loss_w_prior = K.sum(kl_loss_n(z_mean_w, z_log_var_w), axis=-1)
        kl_loss_xw_prior = K.sum(kl_loss_n(z_mean_shared, z_log_var_shared), axis=-1)
        return 0.5 * K.mean(xent_loss_x_shared + xent_loss_w_shared + 
                            self.gamma * (xent_loss_x + xent_loss_w) + 
                            self.beta_shared * kl_loss_xw_prior +
                            self.alpha * (kl_loss_w_xw + kl_loss_x_xw) +
                            self.beta_uni * (kl_loss_x_prior + kl_loss_w_prior))

    def call(self, inputs):
        x = inputs[0]
        w = inputs[1]
        xx = inputs[2]
        ww = inputs[3]
        x_decoded_mean = inputs[4]
        w_decoded_mean = inputs[5]
        z_mean_shared = inputs[6]
        z_log_var_shared = inputs[7]
        z_mean_x = inputs[8]
        z_mean_w = inputs[9]
        z_log_var_x = inputs[10]
        z_log_var_w = inputs[11]
        x_decoded_mean_x = inputs[12]
        w_decoded_mean_w = inputs[13]
        loss = self.vae_loss(x, w, xx, ww, x_decoded_mean, w_decoded_mean, z_mean_shared, z_log_var_shared,z_mean_x, z_mean_w, z_log_var_x, z_log_var_w, x_decoded_mean_x, w_decoded_mean_w)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
    
# Custom loss layer for vanilla multi-modal VAE
class MmVaeLossZero(Layer):
    def __init__(self, original_dim_x, original_dim_w, beta = 1.0):
        self.original_dim_x = original_dim_x
        self.original_dim_w = original_dim_w
        print("original_dim_w: ", original_dim_w)
        self.beta = beta
        self.is_placeholder = True
        super(MmVaeLossZero, self).__init__()
    
    def vae_loss(self, x, w, xx, ww, x_decoded_mean, w_decoded_mean, z_mean_shared, z_log_var_shared):
        # Get the error between the decoding and the true labels
        xent_loss_x_shared = self.original_dim_x * keras.metrics.mean_squared_error(xx, x_decoded_mean)
        xent_loss_w_shared = self.original_dim_w * keras.metrics.mean_squared_error(ww, w_decoded_mean)
        kl_loss_xw_prior = K.sum(kl_loss_n(z_mean_shared, z_log_var_shared), axis=-1)
        return K.mean(xent_loss_x_shared + xent_loss_w_shared + self.beta * kl_loss_xw_prior)
    
    def call(self, inputs):
        x = inputs[0]
        w = inputs[1]
        xx = inputs[2]
        ww = inputs[3]
        x_decoded_mean = inputs[4]
        w_decoded_mean = inputs[5]
        z_mean_shared = inputs[6]
        z_log_var_shared = inputs[7]
        loss = self.vae_loss(x, w, xx, ww, x_decoded_mean, w_decoded_mean, z_mean_shared, z_log_var_shared)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

# Custom loss layer for multi-modal VAE with three modalities
class MmmVaeLoss(Layer):
    def __init__(self, original_dim, alpha = 1.0, beta_shared = 1.0, beta_uni = 1.0, gamma = 1.0):
        self.original_dim = original_dim
        self.alpha = alpha
        self.beta_shared = beta_shared
        self.beta_uni = beta_uni
        self.gamma = gamma
        self.is_placeholder = True
        super(MmmVaeLoss, self).__init__()
        
    def counter_increment(self):
        self.counter = self.counter + np.int(1)
        return self.counter
    def counter_set(self, value):
        self.counter = np.int(value)
    def counter_reset(self):
        self.counter = np.int(0)
    
    def vae_loss(self, inputs):
        # Deflate
        self.counter_set(-1)
        x, w, v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        x_xw, w_xw = inputs[self.counter_increment()], inputs[self.counter_increment()]
        x_xv, v_xv = inputs[self.counter_increment()], inputs[self.counter_increment()]
        w_wv, v_wv = inputs[self.counter_increment()], inputs[self.counter_increment()]
        x_xwv, w_xwv, v_xwv = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        x_decoded_mean_x, x_decoded_mean_w, x_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        w_decoded_mean_x, w_decoded_mean_w, w_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        v_decoded_mean_x, v_decoded_mean_w, v_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        xw_decoded_mean_x, xw_decoded_mean_w, xw_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        xv_decoded_mean_x, wv_decoded_mean_w, wv_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        wv_decoded_mean_x, xv_decoded_mean_w, xv_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        xwv_decoded_mean_x, xwv_decoded_mean_w, xwv_decoded_mean_v = inputs[self.counter_increment()], inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_x, z_log_var_x = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_w, z_log_var_w = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_v, z_log_var_v = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_xw, z_log_var_xw = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_xv, z_log_var_xv = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_wv, z_log_var_wv = inputs[self.counter_increment()], inputs[self.counter_increment()]
        z_mean_xwv, z_log_var_xwv = inputs[self.counter_increment()], inputs[self.counter_increment()]
        
        
        
        # XWV
        xent_loss_x_xwv = self.original_dim * metrics.mean_squared_error(x_xwv, xwv_decoded_mean_x)
        xent_loss_w_xwv = self.original_dim * metrics.mean_squared_error(w_xwv, xwv_decoded_mean_w)
        xent_loss_v_xwv = self.original_dim * metrics.mean_squared_error(v_xwv, xwv_decoded_mean_v)
        kl_loss_xwv_prior = K.sum(kl_loss_n(z_mean_xwv, z_log_var_xwv), axis=-1)
        # XW
        xent_loss_x_xw = self.original_dim * metrics.mean_squared_error(x_xw, xw_decoded_mean_x)
        xent_loss_w_xw = self.original_dim * metrics.mean_squared_error(w_xw, xw_decoded_mean_w)
        kl_loss_xw_prior = K.sum(kl_loss_n(z_mean_xw, z_log_var_xw), axis=-1)
        # XV
        xent_loss_x_xv = self.original_dim * metrics.mean_squared_error(x_xv, xv_decoded_mean_x)
        xent_loss_v_xv = self.original_dim * metrics.mean_squared_error(v_xv, xv_decoded_mean_v)
        kl_loss_xv_prior = K.sum(kl_loss_n(z_mean_xv, z_log_var_xv), axis=-1)
        # WV
        xent_loss_w_wv = self.original_dim * metrics.mean_squared_error(w_wv, wv_decoded_mean_w)
        xent_loss_v_wv = self.original_dim * metrics.mean_squared_error(v_wv, wv_decoded_mean_v)
        kl_loss_wv_prior = K.sum(kl_loss_n(z_mean_wv, z_log_var_wv), axis=-1)
        # X
        xent_loss_x = self.original_dim * metrics.mean_squared_error(x, x_decoded_mean_x)
        kl_loss_x_prior = K.sum(kl_loss_n(z_mean_x, z_log_var_x), axis=-1)
        # W
        xent_loss_w = self.original_dim * metrics.mean_squared_error(w, w_decoded_mean_w)
        kl_loss_w_prior = K.sum(kl_loss_n(z_mean_w, z_log_var_w), axis=-1)
        # V
        xent_loss_v = self.original_dim * metrics.mean_squared_error(v, v_decoded_mean_v)
        kl_loss_v_prior = K.sum(kl_loss_n(z_mean_v, z_log_var_v), axis=-1)
        
        # Mutual XW
        kl_loss_x_xw = K.sum(kl_loss(z_mean_xw, z_mean_x, z_log_var_xw, z_log_var_x), axis=-1);
        kl_loss_w_xw = K.sum(kl_loss(z_mean_xw, z_mean_w, z_log_var_xw, z_log_var_w), axis=-1);
        # Mutual XV
        kl_loss_x_xv = K.sum(kl_loss(z_mean_xv, z_mean_x, z_log_var_xv, z_log_var_x), axis=-1);
        kl_loss_v_xv = K.sum(kl_loss(z_mean_xv, z_mean_v, z_log_var_xv, z_log_var_v), axis=-1);
        # Mutual WV
        kl_loss_w_wv = K.sum(kl_loss(z_mean_wv, z_mean_w, z_log_var_wv, z_log_var_w), axis=-1);
        kl_loss_v_wv = K.sum(kl_loss(z_mean_wv, z_mean_v, z_log_var_wv, z_log_var_v), axis=-1);
        # Mutual XWV
        kl_loss_xw_xwv = K.sum(kl_loss(z_mean_xwv, z_mean_xw, z_log_var_xwv, z_log_var_xw), axis=-1);
        kl_loss_xv_xwv = K.sum(kl_loss(z_mean_xwv, z_mean_xv, z_log_var_xwv, z_log_var_xv), axis=-1);
        kl_loss_wv_xwv = K.sum(kl_loss(z_mean_xwv, z_mean_wv, z_log_var_xwv, z_log_var_wv), axis=-1);        
        
        return K.mean(
        1/3 * (xent_loss_x_xwv + xent_loss_w_xwv + xent_loss_v_xwv +
               self.beta_shared * kl_loss_xwv_prior +
               self.alpha * (kl_loss_xw_xwv + kl_loss_xv_xwv + kl_loss_wv_xwv)) + 
        1/3 * (self.gamma * (xent_loss_x + xent_loss_w + xent_loss_v) +
               self.beta_uni * (kl_loss_x_prior + kl_loss_w_prior + kl_loss_v_prior)) +
        1/6 * (xent_loss_x_xw + xent_loss_w_xw + 
               self.beta_shared * kl_loss_xw_prior +
               self.alpha * (kl_loss_x_xw + kl_loss_w_xw)) +
        1/6 * (xent_loss_x_xv + xent_loss_v_xv + 
               self.beta_shared * kl_loss_xv_prior +
               self.alpha * (kl_loss_x_xv + kl_loss_v_xv)) +
        1/6 * (xent_loss_w_wv + xent_loss_v_wv + 
               self.beta_shared * kl_loss_wv_prior +
               self.alpha * (kl_loss_w_wv + kl_loss_v_wv))
        )

    def call(self, inputs):
        loss = self.vae_loss(inputs)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return inputs[0]