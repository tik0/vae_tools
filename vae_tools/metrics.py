#!/usr/bin/python

import numpy as np

import keras

try:
    from keras.utils.vis_utils import model_to_dot
except:
    from tensorflow.python.keras.utils.vis_utils import model_to_dot

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
import keras
import tensorflow as tf

def kl_loss(mean1, mean2, log_var1, log_var2):
    '''KL-divergence between two Gaussian'''
    return - .5 * (1 + log_var1 - log_var2 - ((np.exp(log_var1) + np.square(mean1 - mean2)) / np.exp(log_var2)))


def kl_loss_n(mean, log_var):
    '''KL-divergence between an abitrary Gaussian and the normal distribution'''
    return kl_loss(mean, 0., log_var, 0)

def mean_squared_error(A, B):
    '''Returns the mean squared error between the datum A and B
    A & B   (np.array): One datum each
    
    returns the mean squared error
    '''
    return ((A - B)**2).flatten().mean()


def binary_cross_entropy(A, B):
    '''Returns the binary cross entropy between the datum A and B
    A & B   (np.array): One datum each
    
    returns the binary cross entropy
    '''
    return (- (A * np.log(B) + (1-A) * np.log(1-B) )).flatten().sum()

# The elbo check
def _elbo_check(decoder, encoder_mean, encoder_logvar, data, data_expected_output, sample_from_enc_dec = False, batch_size = 128, entropy = "mse"):
    #data = np.array(data)
    #data_expected_output = np.array(data_expected_output)
    # Latent mean => (N,2)
    encoded_mean    = encoder_mean.predict(data, batch_size=batch_size)
    # Latent var => (N,2)
    encoded_log_var = encoder_logvar.predict(data, batch_size=batch_size)
    # Sum KL over latent dimensions => (N,)
    kl = np.sum(kl_loss_n(encoded_mean, encoded_log_var) , axis=-1)
    # Encoding gives us original data: (2,N,2), and wie sum over observable dimension => (2,N), i.e. (modalities, data, observable dimensions)
    if not sample_from_enc_dec:
        diff = np.sum(np.square(np.array(decoder.predict(encoded_mean, batch_size=batch_size)) - data_expected_output),axis=-1)
    else:
        diff = np.zeros((len(data_expected_output),len(data_expected_output[0])))
        for idx in np.arange(len(encoded_mean)):
            # Sample some latent space from the input signal
            num_samples = batch_size
            # Check if we have only one modality data.shape = (N,2) vs (2,N,2)
            if len(np.array(data).shape) == 2:
                _data = np.expand_dims(np.array(data), axis=0)
            else:
                _data = np.array(data)
            # Create one input multiple time and sample through the encoder_decoder pipelinge
            samples = np.repeat(_data[:,[idx],:],num_samples,axis=1)
            samples = list(samples)
            with tf.device('/cpu:0'):
                _encoded_mean = np.array(decoder.predict(samples))
            _data_expected_output = np.array(data_expected_output)
            if entropy == "mse":
                diff[:,idx] = np.sum(np.square(np.mean(_encoded_mean - _data_expected_output[:,[idx],:], axis=1)),axis=-1)
            elif entropy == "bin_xent":
                # todo
                pass
    # return the reconstruction loss of first and second modality, and kl loss
    return diff[0,:], diff[1,:], kl

def elbo_check(model_obj, x_set, w_set, _elbo_xw, _elbo_x, _elbo_w, use_ngiam_training_set = False):
    
    decoder = model_obj.get_decoder()

    xw_rec_x, xw_rec_w, xw_kl = _elbo_check(decoder, model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [x_set, w_set], [x_set, w_set])
    if use_ngiam_training_set:
        x_rec_x, x_rec_w, x_kl = _elbo_check(decoder, model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [x_set, np.zeros(w_set.shape)], [x_set, w_set])
        w_rec_x, w_rec_w, w_kl = _elbo_check(decoder, model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [np.zeros(x_set.shape), w_set], [x_set, w_set])
    else:
        x_rec_x, x_rec_w, x_kl = _elbo_check(decoder, model_obj.get_encoder_mean_x(), model_obj.get_encoder_logvar_x(), x_set, [x_set, w_set])
        w_rec_x, w_rec_w, w_kl = _elbo_check(decoder, model_obj.get_encoder_mean_w(), model_obj.get_encoder_logvar_w(), w_set, [x_set, w_set])
        
    elbo_xw = np.sum( xw_rec_x + xw_rec_w - xw_kl) / len(x_set)
    elbo_x = np.sum( x_rec_x + x_rec_w - x_kl) / len(x_set)
    elbo_w = np.sum(  w_rec_x + w_rec_w - w_kl) / len(x_set)
    
    return np.append(_elbo_xw, elbo_xw), np.append(_elbo_x, elbo_x), np.append(_elbo_w, elbo_w)