#!/usr/bin/python

import numpy as np
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
import keras
from . import custom_variational_layer

class CameraRGConv():

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon

    def __init__(self, image_rows_cols_chns, batch_size, filters, num_conv, intermediate_dim, latent_dim, beta):
        self.latent_dim = latent_dim
        rows = image_rows_cols_chns[0]
        rows_2 = int(rows/2)
        cols = image_rows_cols_chns[1]
        cols_2 = int(cols/2)
        img_chns = image_rows_cols_chns[2]
        if K.image_data_format() == 'channels_first':
            original_img_size = (image_rows_cols_chns[2], image_rows_cols_chns[0], image_rows_cols_chns[1])
            output_shape_reshape = (batch_size, filters, rows_2, cols_2)
            output_shape_upsamp = (batch_size, filters, rows+1, cols+1)
        else:
            original_img_size = image_rows_cols_chns
            output_shape_reshape = (batch_size, rows_2, cols_2, filters)
            output_shape_upsamp = (batch_size, rows+1, cols+1, filters)

        # build the net
        self.x = Input(shape=original_img_size)
        self.conv_1 = Conv2D(img_chns,
                        kernel_size=(2, 2),
                        padding='same', activation='relu')(self.x)
        self.conv_2 = Conv2D(filters,
                        kernel_size=(2, 2),
                        padding='same', activation='relu',
                        strides=(2, 2))(self.conv_1)
        self.conv_3 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(self.conv_2)
        self.conv_4 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(self.conv_3)
        self.flat = Flatten()(self.conv_4)
        self.hidden = Dense(intermediate_dim, activation='relu')(self.flat)

        self.z_mean = Dense(latent_dim)(self.hidden)
        self.z_log_var = Dense(latent_dim)(self.hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        self.z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_hid = Dense(intermediate_dim, activation='relu')
        self.decoder_upsample = Dense(filters * rows_2 * cols_2, activation='relu')
        output_shape = output_shape_reshape
        self.decoder_reshape = Reshape(output_shape[1:])
        self.decoder_deconv_1 = Conv2DTranspose(filters,
                                           kernel_size=num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        self.decoder_deconv_2 = Conv2DTranspose(filters,
                                           kernel_size=num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        output_shape = output_shape_upsamp
        self.decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                  kernel_size=(3, 3),
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        self.decoder_mean_squash = Conv2D(img_chns,
                                     kernel_size=2,
                                     padding='valid',
                                     activation='sigmoid')

        self.hid_decoded = self.decoder_hid(self.z)
        self.up_decoded = self.decoder_upsample(self.hid_decoded)
        self.reshape_decoded = self.decoder_reshape(self.up_decoded)
        self.deconv_1_decoded = self.decoder_deconv_1(self.reshape_decoded)
        self.deconv_2_decoded = self.decoder_deconv_2(self.deconv_1_decoded)
        self.x_decoded_relu = self.decoder_deconv_3_upsamp(self.deconv_2_decoded)
        self.x_decoded_mean_squash = self.decoder_mean_squash(self.x_decoded_relu)

        self.y = custom_variational_layer.CameraRG(Dx = np.prod(image_rows_cols_chns), beta = beta)([self.x, self.x_decoded_mean_squash, self.z_mean, self.z_log_var])

    def get_model(self):
        return Model(self.x, self.y)

