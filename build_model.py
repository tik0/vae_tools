#!/usr/bin/python

import numpy as np
import os.path
from keras.utils.vis_utils import model_to_dot
from keras.layers.merge import concatenate as concat
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv1D, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.models import model_from_json
import keras
from . import custom_variational_layer, sampling, layers

class GenericVae():
    def get_model(self, get_new_model = False):
        if 'model' not in dir(self) or get_new_model:
            self.model = Model(self.x, self.y)
        return self.model

    def get_encoder_mean(self):
        return Model(self.x, self.z_mean)

    def get_encoder_logvar(self):
        return Model(self.x, self.z_log_var)
    
    def store_model(self, filename, model, overwrite = False):
        filename_json = filename + ".json"
        filename_h5 = filename + ".h5"
        # serialize model to JSON
        if not os.path.isfile(filename_json) or overwrite:
            model_json = model.to_json()
            with open(filename_json, "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")
        if not os.path.isfile(filename_h5) or overwrite:
            # serialize weights to HDF5
            model.save_weights(filename_h5)
            print("Saved weights to disk")

    def load_model(self, filename):
        filename_json = filename + ".json"
        filename_h5 = filename + ".h5"
        # load json and create model
        json_file = open(filename_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(filename_h5)
        print("Loaded model from disk")
        return loaded_model

class Conv1DTranspose():
    def __init__(self, filters, kernel_size, strides=2, activation='relu', padding='same'):
        self.input  = x = Lambda(lambda x: K.expand_dims(x, axis=2))
        self.conv   = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)
        self.output = Lambda(lambda x: K.squeeze(x, axis=2))
    def __call__(self, input_tensor):
        return self.output(self.conv(self.input(input_tensor)))

class VaeRgConv(sampling.Sampling, GenericVae):

    def __init__(self):
        pass

    def configure(self, image_rows_cols_chns, batch_size, filters, num_conv, intermediate_dim, latent_dim, beta):
        self.use_conv = True
        self.latent_dim = latent_dim
        self.original_dim = np.prod(image_rows_cols_chns)
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
        self.z = Lambda(self.randn, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

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

        self.y = custom_variational_layer.VaeLoss(self.original_dim, beta = beta)([self.x, self.x_decoded_mean_squash, self.z_mean, self.z_log_var])

    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self._hid_decoded = self.decoder_hid(self.decoder_input)
        self._up_decoded = self.decoder_upsample(self._hid_decoded)
        self._reshape_decoded = self.decoder_reshape(self._up_decoded)
        self._deconv_1_decoded = self.decoder_deconv_1(self._reshape_decoded)
        self._deconv_2_decoded = self.decoder_deconv_2(self._deconv_1_decoded)
        self._x_decoded_relu = self.decoder_deconv_3_upsamp(self._deconv_2_decoded)
        self._x_decoded_mean_squash = self.decoder_mean_squash(self._x_decoded_relu)
        return Model(self.decoder_input, self._x_decoded_mean_squash)


class Vae(sampling.Sampling, GenericVae):

    def __init__(self):
        pass
        
    def configure(self, original_dim, intermediate_dim, latent_dim, beta):
        self.use_conv = False
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.intermediate_dim_2 = np.int(intermediate_dim / 2)
        self.original_dim = original_dim
        self.x = Input(shape=(self.original_dim,))
        self.h1 = Dense(self.intermediate_dim, activation='relu')(self.x)
        self.h2 = Dense(self.intermediate_dim_2, activation='relu')(self.h1)
        self.z_mean = Dense(self.latent_dim)(self.h2)
        self.z_log_var = Dense(self.latent_dim)(self.h2)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(self.randn, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h2 = Dense(self.intermediate_dim_2, activation='relu')
        self.decoder_h1 = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        self.h2_decoded = self.decoder_h2(self.z)
        self.h1_decoded = self.decoder_h1(self.h2_decoded)
        self.x_decoded_mean = self.decoder_mean(self.h1_decoded)

        self.y = custom_variational_layer.VaeLoss(self.original_dim, beta = beta)([self.x, self.x_decoded_mean, self.z_mean, self.z_log_var])

    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self._h2_decoded = self.decoder_h2(self.decoder_input)
        self._h1_decoded = self.decoder_h1(self._h2_decoded)
        self._x_decoded_mean = self.decoder_mean(self._h1_decoded)
        return Model(self.decoder_input, self._x_decoded_mean)

class Vae1dConv(sampling.Sampling, GenericVae):

    def __init__(self):
        pass
    
    def configure(self, original_dim, batch_size, filters, num_conv, intermediate_dim, latent_dim, beta):
        self.use_conv = True
        self.latent_dim = latent_dim
        self.original_dim = original_dim
        self.original_dim_2 = int(original_dim / 2)
        self.intermediate_dim = intermediate_dim

        # build the net
        
        self.x = Input(shape=(self.original_dim,1))
        self.conv_1 = Conv1D(1,
                        kernel_size=2,
                        padding='same', activation='relu')(self.x)
        self.conv_2 = Conv1D(filters,
                        kernel_size=2,
                        padding='same', activation='relu',
                        strides=2)(self.conv_1)
        self.conv_3 = Conv1D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(self.conv_2)
        self.conv_4 = Conv1D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(self.conv_3)
        self.flat = Flatten()(self.conv_4)
        self.hidden = Dense(self.intermediate_dim, activation='relu')(self.flat)

        self.z_mean = Dense(latent_dim)(self.hidden)
        self.z_log_var = Dense(latent_dim)(self.hidden)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        self.z = Lambda(self.randn, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_hid = Dense(self.intermediate_dim, activation='relu')
        self.decoder_upsample = Dense(filters * self.original_dim_2, activation='relu')

        self.output_shape = (batch_size, self.original_dim_2, filters)
        self.decoder_reshape = Reshape(self.output_shape[1:])

        self.hid_decoded = self.decoder_hid(self.z)
        self.up_decoded = self.decoder_upsample(self.hid_decoded)
        self.reshape_decoded = self.decoder_reshape(self.up_decoded)

        self.decoder_deconv_1 = Conv1DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        self.decoder_deconv_2 = Conv1DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        self.output_shape = (batch_size, self.original_dim + 1, filters)
        self.decoder_deconv_relu = Conv1DTranspose(filters, kernel_size=3, strides=2, padding='valid', activation='relu')
        self.decoder_mean_squash = Conv1D(1, kernel_size=2, padding='valid', activation='sigmoid')

        self.deconv_1_decoded = self.decoder_deconv_1(self.reshape_decoded)
        self.deconv_2_decoded = self.decoder_deconv_2(self.deconv_1_decoded)
        self.x_decoded_relu = self.decoder_deconv_relu(self.deconv_2_decoded) 
        self.x_decoded_mean_squash = self.decoder_mean_squash(self.x_decoded_relu)

        self.y = custom_variational_layer.VaeLoss(self.original_dim, beta = beta)([self.x, self.x_decoded_mean_squash, self.z_mean, self.z_log_var])

    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self._hid_decoded = self.decoder_hid(self.decoder_input)
        self._up_decoded = self.decoder_upsample(self._hid_decoded)
        self._reshape_decoded = self.decoder_reshape(self._up_decoded)
        self._deconv_1_decoded = self.decoder_deconv_1(self._reshape_decoded)
        self._deconv_2_decoded = self.decoder_deconv_2(self._deconv_1_decoded)
        self._x_decoded_relu = self.decoder_deconv_relu(self._deconv_2_decoded) 
        self._x_decoded_mean_squash = self.decoder_mean_squash(self._x_decoded_relu)
        return Model(self.decoder_input, self._x_decoded_mean_squash)

class Vae2dConv(sampling.Sampling, GenericVae):
    ''' Vanilla VAE with 2d convolutions
    '''
    def __init__(self):
        pass
    
    def configure(self, img_rows, img_cols, img_chns, batch_size, filters, num_conv, intermediate_dim, latent_dim, beta):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        # build the net
        if K.image_data_format() == 'channels_first':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)
        self.original_dim = np.int(np.prod(original_img_size))

        img_rows_2 = np.int(img_rows / 2)
        img_cols_2 = np.int(img_cols / 2)

        self.x = Input(shape=original_img_size)
        self.conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(self.x)
        self.conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(self.conv_1)
        self.conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(self.conv_2)
        self.conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(self.conv_3)
        self.flat = Flatten()(self.conv_4)
        self.hidden = Dense(intermediate_dim, activation='relu')(self.flat)

        self.z_mean = Dense(latent_dim)(self.hidden)
        self.z_log_var = Dense(latent_dim)(self.hidden)

        self.z = Lambda(self.randn, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_hid = Dense(intermediate_dim, activation='relu')
        self.decoder_upsample = Dense(filters * 14 * 14, activation='relu')

        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, img_rows_2, img_cols_2)
        else:
            output_shape = (batch_size, img_rows_2, img_cols_2, filters)

        self.decoder_reshape = Reshape(output_shape[1:])
        self.decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        self.decoder_deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, np.int(img_rows + 1), np.int(img_cols + 1))
        else:
            output_shape = (batch_size, np.int(img_rows + 1), np.int(img_cols + 1), filters)
        self.decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        self.decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

        self.hid_decoded = self.decoder_hid(self.z)
        self.up_decoded = self.decoder_upsample(self.hid_decoded)
        self.reshape_decoded = self.decoder_reshape(self.up_decoded)
        self.deconv_1_decoded = self.decoder_deconv_1(self.reshape_decoded)
        self.deconv_2_decoded = self.decoder_deconv_2(self.deconv_1_decoded)
        self.x_decoded_relu = self.decoder_deconv_3_upsamp(self.deconv_2_decoded)
        self.x_decoded_mean_squash = self.decoder_mean_squash(self.x_decoded_relu)

        self.y = custom_variational_layer.VaeLoss(self.original_dim, beta = beta, reconstruction_mse = False)([self.x, self.x_decoded_mean_squash, self.z_mean, self.z_log_var])

    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self._hid_decoded = self.decoder_hid(self.decoder_input)
        self._up_decoded = self.decoder_upsample(self._hid_decoded)
        self._reshape_decoded = self.decoder_reshape(self._up_decoded)
        self._deconv_1_decoded = self.decoder_deconv_1(self._reshape_decoded)
        self._deconv_2_decoded = self.decoder_deconv_2(self._deconv_1_decoded)
        self._x_decoded_relu = self.decoder_deconv_3_upsamp(self._deconv_2_decoded)
        self._x_decoded_mean_squash = self.decoder_mean_squash(self._x_decoded_relu)
        return Model(self.decoder_input, self._x_decoded_mean_squash)


    
class Cvae2dConv(sampling.Sampling, GenericVae):
    ''' Vanilla CVAE with 2d convolutions
    '''
    def __init__(self):
        pass
    
    def configure(self, img_rows, img_cols, img_chns, batch_size, label_dim, filters, num_conv, intermediate_dim, latent_dim, beta, deep = False):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.intermediate_dim_2 = np.int(intermediate_dim / 2)
        self.deep = deep
        self.label_dim = label_dim

        # build the net
        if K.image_data_format() == 'channels_first':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)
        self.original_dim = np.int(np.prod(original_img_size))

        img_rows_2 = np.int(img_rows / 2)
        img_cols_2 = np.int(img_cols / 2)

        self.x = Input(shape=original_img_size)
        self.label = Input(shape=(label_dim,))
        self.conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(self.x)
        self.conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(self.conv_1)
        self.conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(self.conv_2)
        self.conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(self.conv_3)
        self.flat = Flatten()(self.conv_4)
        self.flat_concat = concat([self.flat, self.label])
        self.hidden = Dense(intermediate_dim, activation='relu')(self.flat_concat)
        if deep == True:
            self.hidden_2 = Dense(self.intermediate_dim_2, activation='relu')(self.hidden)
        else:
            self.hidden_2 = self.hidden
        self.z_mean = Dense(latent_dim)(self.hidden_2)
        self.z_log_var = Dense(latent_dim)(self.hidden_2)

        self.z = Lambda(self.randn, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_hid_2 = Dense(self.intermediate_dim_2, activation='relu')
        self.decoder_hid = Dense(intermediate_dim, activation='relu')
        self.decoder_upsample = Dense(filters * 14 * 14, activation='relu')

        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, img_rows_2, img_cols_2)
        else:
            output_shape = (batch_size, img_rows_2, img_cols_2, filters)

        self.decoder_reshape = Reshape(output_shape[1:])
        self.decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        self.decoder_deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, np.int(img_rows + 1), np.int(img_cols + 1))
        else:
            output_shape = (batch_size, np.int(img_rows + 1), np.int(img_cols + 1), filters)
        self.decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        self.decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

        self.z_label_concat = concat([self.z, self.label])
        if deep == True:
            self.hid_decoded_2 = self.decoder_hid_2(self.z_label_concat)
            self.hid_decoded = self.decoder_hid(self.hid_decoded_2)
        else:
            self.hid_decoded = self.decoder_hid(self.z_label_concat)
        self.up_decoded = self.decoder_upsample(self.hid_decoded)
        self.reshape_decoded = self.decoder_reshape(self.up_decoded)
        self.deconv_1_decoded = self.decoder_deconv_1(self.reshape_decoded)
        self.deconv_2_decoded = self.decoder_deconv_2(self.deconv_1_decoded)
        self.x_decoded_relu = self.decoder_deconv_3_upsamp(self.deconv_2_decoded)
        self.x_decoded_mean_squash = self.decoder_mean_squash(self.x_decoded_relu)

        self.y = custom_variational_layer.VaeLoss(self.original_dim, beta = beta, reconstruction_mse = False)([self.x, self.x_decoded_mean_squash, self.z_mean, self.z_log_var])

    def get_model(self, get_new_model = False):
        if 'model' not in dir(self) or get_new_model:
            self.model = Model([self.x, self.label], self.y)
        return self.model

    def get_encoder_mean(self):
        return Model([self.x, self.label], self.z_mean)

    def get_encoder_logvar(self):
        return Model([self.x, self.label], self.z_log_var)
    
    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self.label_input = Input(shape=(self.label_dim,))
        self.decoder_label_concat_input = concat([self.decoder_input, self.label_input])
        if self.deep == True:
            self._hid_decoded_2 = self.decoder_hid_2(self.decoder_label_concat_input)
            self._hid_decoded = self.decoder_hid(self._hid_decoded_2)
        else:
            self._hid_decoded = self.decoder_hid(self.decoder_label_concat_input)
        self._up_decoded = self.decoder_upsample(self._hid_decoded)
        self._reshape_decoded = self.decoder_reshape(self._up_decoded)
        self._deconv_1_decoded = self.decoder_deconv_1(self._reshape_decoded)
        self._deconv_2_decoded = self.decoder_deconv_2(self._deconv_1_decoded)
        self._x_decoded_relu = self.decoder_deconv_3_upsamp(self._deconv_2_decoded)
        self._x_decoded_mean_squash = self.decoder_mean_squash(self._x_decoded_relu)
        return Model([self.decoder_input, self.label_input], self._x_decoded_mean_squash)

    
class MmVae(sampling.Sampling, GenericVae):

    def __init__(self):
        pass
        
    def configure(self, original_dim_x, original_dim_w, intermediate_dim_x, intermediate_dim_w, intermediate_dim_shared, latent_dim, deep = True, alpha = .1, beta_shared = .1, beta_uni = 1., gamma = 1., reconstruction_mse = True):
        self.latent_dim = latent_dim
        intermediate_dim_x_2 = np.int(intermediate_dim_x / 2)
        intermediate_dim_w_2 = np.int(intermediate_dim_w / 2)
        self.deep = deep
        self.alpha = alpha
        self.beta_shared = beta_shared
        self.beta_uni = beta_uni
        self.gamma = gamma
        self.original_dim_x = original_dim_x
        self.original_dim_w = original_dim_w
        self.reconstruction_mse = reconstruction_mse
        # X Encoder
        self.xx = Input(shape=(original_dim_x,), name='input_img')
        self.h_x = Dense(intermediate_dim_x, activation='relu', name='enc_img_x')(self.xx)
        if deep:
            self.h_x_2 = Dense(intermediate_dim_x_2, activation='relu', name='enc_img_x_2')(self.h_x)
        else:
            self.h_x_2 = self.h_x
        self.z_mean_x = Dense(latent_dim, name='mean_img')(self.h_x_2)
        self.z_log_var_x = Dense(latent_dim, name='var_img')(self.h_x_2)
        # W Encoder
        self.ww = Input(shape=(original_dim_w,), name='input_lab')
        self.h_w = Dense(intermediate_dim_w, activation='relu', name='enc_lab_w')(self.ww)
        if deep:
            self.h_w_2 = Dense(intermediate_dim_w_2, activation='relu', name='enc_img_w_2')(self.h_w)
        else:
            self.h_w_2 = self.h_w
        self.z_mean_w = Dense(latent_dim, name='mean_lab')(self.h_w_2)
        self.z_log_var_w = Dense(latent_dim, name='var_lab')(self.h_w_2)
        # Shared XW Encoder
        self.x = Input(shape=(original_dim_x,), name='input_img_shared')
        self.w = Input(shape=(original_dim_w,), name='input_lab_shared')
        self.h_x_shared = Dense(intermediate_dim_x, activation='relu', name='enc_img')(self.x)
        self.h_w_shared = Dense(intermediate_dim_w, activation='relu', name='enc_lab')(self.w)
        if deep:
            self.h_x_shared_2 = Dense(intermediate_dim_x_2, activation='relu', name='enc_img_2')(self.h_x_shared)
            self.h_w_shared_2 = Dense(intermediate_dim_w_2, activation='relu', name='enc_lab_2')(self.h_w_shared)
        else:
            self.h_x_shared_2 = self.h_x_shared
            self.h_w_shared_2 = self.h_w_shared
        self.h_concat = keras.layers.concatenate([self.h_x_shared_2, self.h_w_shared_2], name='concat')
        self.h_shared = Dense(intermediate_dim_shared, activation='relu', name='enc_shared')(self.h_concat)
        self.z_mean_shared = Dense(latent_dim, name='mean_shared')(self.h_shared)
        self.z_log_var_shared = Dense(latent_dim, name='var_shared')(self.h_shared)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z_x = Lambda(self.randn, output_shape=(latent_dim,), name='sample_img')([self.z_mean_x, self.z_log_var_x])
        self.z_w = Lambda(self.randn, output_shape=(latent_dim,), name='sample_lab')([self.z_mean_w, self.z_log_var_w])
        self.z_shared = Lambda(self.randn, output_shape=(latent_dim,), name='sample_shared')([self.z_mean_shared, self.z_log_var_shared])

        # we instantiate these layers separately so as to reuse them later
        # X and W decoder template
        self.decoder_h_x_2 = Dense(intermediate_dim_x_2, activation='relu', name='dec_img_2')
        self.decoder_h_w_2 = Dense(intermediate_dim_w_2, activation='relu', name='dec_lab_2')
        self.decoder_h_x = Dense(intermediate_dim_x, activation='relu', name='dec_img')
        self.decoder_h_w = Dense(intermediate_dim_w, activation='relu', name='dec_lab')
        self.decoder_mean_x = Dense(original_dim_w, activation='sigmoid', name='output_img')
        self.decoder_mean_w = Dense(original_dim_w, activation='sigmoid', name='output_lab')

        # X Decoder
        if deep:
            self.h_x_decoded_x_2 = self.decoder_h_x_2(self.z_x)
            self.h_w_decoded_x_2 = self.decoder_h_w_2(self.z_x)
            self.h_x_decoded_x = self.decoder_h_x(self.h_x_decoded_x_2)
            self.h_w_decoded_x = self.decoder_h_w(self.h_w_decoded_x_2)
        else:
            self.h_x_decoded_x = self.decoder_h_x(self.z_x)
            self.h_w_decoded_x = self.decoder_h_w(self.z_x)
        self.x_decoded_mean_x = self.decoder_mean_x(self.h_x_decoded_x)
        self.w_decoded_mean_x = self.decoder_mean_w(self.h_w_decoded_x)
        # W Decoder
        if deep:
            self.h_x_decoded_w_2 = self.decoder_h_x_2(self.z_w)
            self.h_w_decoded_w_2 = self.decoder_h_w_2(self.z_w)
            self.h_x_decoded_w = self.decoder_h_x(self.h_x_decoded_w_2)
            self.h_w_decoded_w = self.decoder_h_w(self.h_w_decoded_w_2)
        else:
            self.h_x_decoded_w = self.decoder_h_x(self.z_w)
            self.h_w_decoded_w = self.decoder_h_w(self.z_w)
        self.x_decoded_mean_w = self.decoder_mean_x(self.h_x_decoded_w)
        self.w_decoded_mean_w = self.decoder_mean_w(self.h_w_decoded_w)
        # Shared XW Decoder
        # for the shared decoder, we use the same decoder_* layers, we just rewire them
        if deep:
            self.h_x_decoded_shared_2 = self.decoder_h_x_2(self.z_shared)
            self.h_w_decoded_shared_2 = self.decoder_h_w_2(self.z_shared)
            self.h_x_decoded_shared = self.decoder_h_x(self.h_x_decoded_shared_2)
            self.h_w_decoded_shared = self.decoder_h_w(self.h_w_decoded_shared_2)
        else:
            self.h_x_decoded_shared = self.decoder_h_x(self.z_shared)
            self.h_w_decoded_shared = self.decoder_h_w(self.z_shared)
        self.x_decoded_mean_shared = self.decoder_mean_x(self.h_x_decoded_shared)
        self.w_decoded_mean_shared = self.decoder_mean_w(self.h_w_decoded_shared)

        self._set_custom_variational_layer()
        
    def reconfigure(self, alpha = None, beta_shared = None, beta_uni = None, gamma = None, reconstruction_mse = None):
        if alpha:
            self.alpha = alpha
        if beta_shared:
            self.beta_shared = beta_shared
        if beta_uni:
            self.beta_uni = beta_uni
        if gamma:
            self.gamma = gamma
        if reconstruction_mse:
            self.reconstruction_mse = reconstruction_mse
        
        self._set_custom_variational_layer()

    def _set_custom_variational_layer(self):
        self.y = custom_variational_layer.MmVaeLoss(self.original_dim_x,
                                                    self.original_dim_w,
                                                    alpha = self.alpha,
                                                    beta_shared = self.beta_shared,
                                                    beta_uni = self.beta_uni,
                                                    gamma = self.gamma,
                                                    reconstruction_mse = self.reconstruction_mse)(
            [self.x, self.w, self.xx, self.ww, self.x_decoded_mean_shared,
             self.w_decoded_mean_shared, self.z_mean_shared, self.z_log_var_shared,
             self.z_mean_x, self.z_mean_w, self.z_log_var_x, self.z_log_var_w,
             self.x_decoded_mean_x, self.w_decoded_mean_w])

    def get_model(self, get_new_model = False):
        if 'model' not in dir(self) or get_new_model:
             self.model = Model(inputs=[self.x, self.w, self.xx, self.ww], outputs = self.y)           
        return self.model
    
    def get_encoder_decoder_shared(self):
        '''ATTENTION: This model applies sampling in the hidden layer'''
        return Model([self.x,self.w], [self.x_decoded_mean_shared, self.w_decoded_mean_shared])
    def get_encoder_decoder_x(self):
        '''ATTENTION: This model applies sampling in the hidden layer'''
        return Model(self.xx, [self.x_decoded_mean_x, self.w_decoded_mean_x])
    def get_encoder_decoder_w(self):
        '''ATTENTION: This model applies sampling in the hidden layer'''
        return Model(self.ww, [self.x_decoded_mean_w, self.w_decoded_mean_w])
    def get_encoder_mean_shared(self):
        return Model([self.x,self.w], self.z_mean_shared)
    def get_encoder_mean_x(self):
        return Model(self.xx, self.z_mean_x)
    def get_encoder_mean_w(self):
        return Model(self.ww, self.z_mean_w)
    def get_encoder_logvar_shared(self):
        return Model([self.x,self.w], self.z_log_var_shared)
    def get_encoder_logvar_x(self):
        return Model(self.xx, self.z_log_var_x)
    def get_encoder_logvar_w(self):
        return Model(self.ww, self.z_log_var_w)
    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        if self.deep:
            self._h_decoded_x_2 = self.decoder_h_x_2(self.decoder_input)
            self._h_decoded_w_2 = self.decoder_h_w_2(self.decoder_input)
            self._h_decoded_x = self.decoder_h_x(self._h_decoded_x_2)
            self._h_decoded_w = self.decoder_h_w(self._h_decoded_w_2)
        else:
            self._h_decoded_x = self.decoder_h_x(self.decoder_input)
            self._h_decoded_w = self.decoder_h_w(self.decoder_input)
        self._x_decoded_mean = self.decoder_mean_x(self._h_decoded_x)
        self._w_decoded_mean = self.decoder_mean_w(self._h_decoded_w)
        return Model(self.decoder_input, [self._x_decoded_mean, self._w_decoded_mean])
    def _pin_uni_modal_encoder(self, pin = True):
        layers.set_layerweights_trainable(self.get_model(), ln_start = "enc_img_x", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_start = "enc_lab_x", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_full = "mean_img", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_full = "mean_lab", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_full = "var_img", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_full = "var_lab", trainable = not pin)
    def pin_uni_modal_encoder(self):
        self._pin_uni_modal_encoder(self)
    def unpin_uni_modal_encoder(self):
        self._pin_uni_modal_encoder(self, pin = False)
        
    def _pin_decoder(self, pin = True):
        layers.set_layerweights_trainable(self.get_model(), ln_start = "dec_", trainable = not pin)
        layers.set_layerweights_trainable(self.get_model(), ln_start = "output_", trainable = not pin)
    def pin_decoder(self):
        self._pin_decoder(self)
    def unpin_decoder(self):
        self._pin_decoder(self, pin = False)
    
class MmmVae(sampling.Sampling, GenericVae):

    def __init__(self):
        pass
        
    def configure(self, original_dim, intermediate_dim, intermediate_dim_shared, latent_dim, deep = True, alpha = .1, beta_shared = .1, beta_uni = 1., gamma = 1.):
        self.latent_dim = latent_dim
        intermediate_dim_2 = np.int(intermediate_dim / 2)
        self.deep = deep
        # X Encoder
        self.x = Input(shape=(original_dim,), name='input_img')
        self.h_x = Dense(intermediate_dim, activation='relu', name='enc_img_x')(self.x)
        if deep:
            self.h_x_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_x_2')(self.h_x)
        else:
            self.h_x_2 = self.h_x
        self.z_mean_x = Dense(latent_dim, name='mean_img')(self.h_x_2)
        self.z_log_var_x = Dense(latent_dim, name='var_img')(self.h_x_2)
        # W Encoder
        self.w = Input(shape=(original_dim,), name='input_lab')
        self.h_w = Dense(intermediate_dim, activation='relu', name='enc_lab_w')(self.w)
        if deep:
            self.h_w_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_w_2')(self.h_w)
        else:
            self.h_w_2 = self.h_w
        self.z_mean_w = Dense(latent_dim, name='mean_lab')(self.h_w_2)
        self.z_log_var_w = Dense(latent_dim, name='var_lab')(self.h_w_2)
        # V Encoder
        self.v = Input(shape=(original_dim,), name='input_ref')
        self.h_v = Dense(intermediate_dim, activation='relu', name='enc_lab_v')(self.v)
        if deep:
            self.h_v_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_v_2')(self.h_v)
        else:
            self.h_v_2 = self.h_v
        self.z_mean_v = Dense(latent_dim, name='mean_ref')(self.h_v_2)
        self.z_log_var_v = Dense(latent_dim, name='var_ref')(self.h_v_2)

        # Shared XW Encoder
        self.x_xw = Input(shape=(original_dim,), name='input_img_shared_xw')
        self.w_xw = Input(shape=(original_dim,), name='input_lab_shared_xw')
        self.h_x_xw = Dense(intermediate_dim, activation='relu', name='enc_img_xw')(self.x_xw)
        self.h_w_xw = Dense(intermediate_dim, activation='relu', name='enc_lab_xw')(self.w_xw)
        if deep:
            self.h_x_xw_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_2_xw')(self.h_x_xw)
            self.h_w_xw_2 = Dense(intermediate_dim_2, activation='relu', name='enc_lab_2_xw')(self.h_w_xw)
        else:
            self.h_x_xw_2 = self.h_x_xw
            self.h_w_xw_2 = self.h_w_xw
        self.h_concat_xw = keras.layers.concatenate([self.h_x_xw_2, self.h_w_xw_2], name='concat_xw')
        self.h_xw = Dense(intermediate_dim_shared, activation='relu', name='enc_shared_xw')(self.h_concat_xw)
        self.z_mean_xw = Dense(latent_dim, name='mean_shared_xw')(self.h_xw)
        self.z_log_var_xw = Dense(latent_dim, name='var_shared_xw')(self.h_xw)


        # Shared XV Encoder
        self.x_xv = Input(shape=(original_dim,), name='input_img_shared_xv')
        self.v_xv = Input(shape=(original_dim,), name='input_ref_shared_xv')
        self.h_x_xv = Dense(intermediate_dim, activation='relu', name='enc_img_xv')(self.x_xv)
        self.h_v_xv = Dense(intermediate_dim, activation='relu', name='enc_lab_xv')(self.v_xv)
        if deep:
            self.h_x_xv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_2_xv')(self.h_x_xv)
            self.h_v_xv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_lab_2_xv')(self.h_v_xv)
        else:
            self.h_x_xv_2 = self.h_x_xv
            self.h_v_xv_2 = self.h_v_xv
        self.h_concat_xv = keras.layers.concatenate([self.h_x_xv_2, self.h_v_xv_2], name='concat_xv')
        self.h_xv = Dense(intermediate_dim_shared, activation='relu', name='enc_shared_xv')(self.h_concat_xv)
        self.z_mean_xv = Dense(latent_dim, name='mean_shared_xv')(self.h_xv)
        self.z_log_var_xv = Dense(latent_dim, name='var_shared_xv')(self.h_xv)
        
        # Shared WV Encoder
        self.w_wv = Input(shape=(original_dim,), name='input_img_shared_wv')
        self.v_wv = Input(shape=(original_dim,), name='input_ref_shared_wv')
        self.h_w_wv = Dense(intermediate_dim, activation='relu', name='enc_img_wv')(self.w_wv)
        self.h_v_wv = Dense(intermediate_dim, activation='relu', name='enc_lab_wv')(self.v_wv)
        if deep:
            self.h_w_wv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_2_wv')(self.h_w_wv)
            self.h_v_wv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_lab_2_wv')(self.h_v_wv)
        else:
            self.h_w_wv_2 = self.h_w_wv
            self.h_v_wv_2 = self.h_v_wv
        self.h_concat_wv = keras.layers.concatenate([self.h_w_wv_2, self.h_v_wv_2], name='concat_wv')
        self.h_wv = Dense(intermediate_dim_shared, activation='relu', name='enc_shared_wv')(self.h_concat_wv)
        self.z_mean_wv = Dense(latent_dim, name='mean_shared_wv')(self.h_wv)
        self.z_log_var_wv = Dense(latent_dim, name='var_shared_wv')(self.h_wv)
        
        # Shared XWV Encoder
        self.x_xwv = Input(shape=(original_dim,), name='input_img_shared_xwv')
        self.w_xwv = Input(shape=(original_dim,), name='input_lab_shared_xwv')
        self.v_xwv = Input(shape=(original_dim,), name='input_ref_shared_xwv')
        self.h_x_xwv = Dense(intermediate_dim, activation='relu', name='enc_img_xwv')(self.x_xwv)
        self.h_w_xwv = Dense(intermediate_dim, activation='relu', name='enc_lab_xwv')(self.w_xwv)
        self.h_v_xwv = Dense(intermediate_dim, activation='relu', name='enc_ref_xwv')(self.v_xwv)
        if deep:
            self.h_x_xwv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_img_2_xwv')(self.h_x_xwv)
            self.h_w_xwv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_lab_2_xwv')(self.h_w_xwv)
            self.h_v_xwv_2 = Dense(intermediate_dim_2, activation='relu', name='enc_ref_2_xwv')(self.h_v_xwv)
        else:
            self.h_x_xwv_2 = self.h_x_xwv
            self.h_w_xwv_2 = self.h_w_xwv
            self.h_v_xwv_2 = self.h_v_xwv
        self.h_concat_xwv = keras.layers.concatenate([self.h_x_xwv_2, self.h_w_xwv_2, self.h_v_xwv_2], name='concat_xwv')
        self.h_xwv = Dense(intermediate_dim_shared, activation='relu', name='enc_shared_xwv')(self.h_concat_xwv)
        self.z_mean_xwv = Dense(latent_dim, name='mean_shared_xwv')(self.h_xwv)
        self.z_log_var_xwv = Dense(latent_dim, name='var_shared_xwv')(self.h_xwv)

        
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z_x = Lambda(self.randn, output_shape=(latent_dim,), name='sample_img')([self.z_mean_x, self.z_log_var_x])
        self.z_w = Lambda(self.randn, output_shape=(latent_dim,), name='sample_lab')([self.z_mean_w, self.z_log_var_w])
        self.z_v = Lambda(self.randn, output_shape=(latent_dim,), name='sample_ref')([self.z_mean_v, self.z_log_var_v])
        self.z_xw = Lambda(self.randn, output_shape=(latent_dim,), name='sample_xw')([self.z_mean_xw, self.z_log_var_xw])
        self.z_xv = Lambda(self.randn, output_shape=(latent_dim,), name='sample_xv')([self.z_mean_xv, self.z_log_var_xv])
        self.z_wv = Lambda(self.randn, output_shape=(latent_dim,), name='sample_wv')([self.z_mean_wv, self.z_log_var_wv])
        self.z_xwv = Lambda(self.randn, output_shape=(latent_dim,), name='sample_xwv')([self.z_mean_xwv, self.z_log_var_xwv])

        # we instantiate these layers separately so as to reuse them later
        # X and W decoder template
        self.decoder_h_x_2 = Dense(intermediate_dim_2, activation='relu', name='dec_img_2')
        self.decoder_h_w_2 = Dense(intermediate_dim_2, activation='relu', name='dec_lab_2')
        self.decoder_h_v_2 = Dense(intermediate_dim_2, activation='relu', name='dec_ref_2')
        self.decoder_h_x = Dense(intermediate_dim, activation='relu', name='dec_img')
        self.decoder_h_w = Dense(intermediate_dim, activation='relu', name='dec_lab')
        self.decoder_h_v = Dense(intermediate_dim, activation='relu', name='dec_ref')
        self.decoder_mean_x = Dense(original_dim, activation='sigmoid', name='output_img')
        self.decoder_mean_w = Dense(original_dim, activation='sigmoid', name='output_lab')
        self.decoder_mean_v = Dense(original_dim, activation='sigmoid', name='output_ref')

        # X Decoder
        if deep:
            self.h_x_decoded_x_2 = self.decoder_h_x_2(self.z_x)
            self.h_w_decoded_x_2 = self.decoder_h_w_2(self.z_x)
            self.h_v_decoded_x_2 = self.decoder_h_v_2(self.z_x)
            self.h_x_decoded_x = self.decoder_h_x(self.h_x_decoded_x_2)
            self.h_w_decoded_x = self.decoder_h_w(self.h_w_decoded_x_2)
            self.h_v_decoded_x = self.decoder_h_v(self.h_v_decoded_x_2)
        else:
            self.h_x_decoded_x = self.decoder_h_x(self.z_x)
            self.h_w_decoded_x = self.decoder_h_w(self.z_x)
            self.h_v_decoded_x = self.decoder_h_v(self.z_x)
        self.x_decoded_mean_x = self.decoder_mean_x(self.h_x_decoded_x)
        self.w_decoded_mean_x = self.decoder_mean_w(self.h_w_decoded_x)
        self.v_decoded_mean_x = self.decoder_mean_v(self.h_v_decoded_x)
        # W Decoder
        if deep:
            self.h_x_decoded_w_2 = self.decoder_h_x_2(self.z_w)
            self.h_w_decoded_w_2 = self.decoder_h_w_2(self.z_w)
            self.h_v_decoded_w_2 = self.decoder_h_v_2(self.z_w)
            self.h_x_decoded_w = self.decoder_h_x(self.h_x_decoded_w_2)
            self.h_w_decoded_w = self.decoder_h_w(self.h_w_decoded_w_2)
            self.h_v_decoded_w = self.decoder_h_v(self.h_v_decoded_w_2)
        else:
            self.h_x_decoded_w = self.decoder_h_x(self.z_w)
            self.h_w_decoded_w = self.decoder_h_w(self.z_w)
            self.h_v_decoded_w = self.decoder_h_v(self.z_w)
        self.x_decoded_mean_w = self.decoder_mean_x(self.h_x_decoded_w)
        self.w_decoded_mean_w = self.decoder_mean_w(self.h_w_decoded_w)
        self.v_decoded_mean_w = self.decoder_mean_v(self.h_v_decoded_w)
        # V Decoder
        if deep:
            self.h_x_decoded_v_2 = self.decoder_h_x_2(self.z_v)
            self.h_w_decoded_v_2 = self.decoder_h_w_2(self.z_v)
            self.h_v_decoded_v_2 = self.decoder_h_v_2(self.z_v)
            self.h_x_decoded_v = self.decoder_h_x(self.h_x_decoded_v_2)
            self.h_w_decoded_v = self.decoder_h_w(self.h_w_decoded_v_2)
            self.h_v_decoded_v = self.decoder_h_v(self.h_v_decoded_v_2)
        else:
            self.h_x_decoded_v = self.decoder_h_x(self.z_v)
            self.h_w_decoded_v = self.decoder_h_w(self.z_v)
            self.h_v_decoded_v = self.decoder_h_v(self.z_v)
        self.x_decoded_mean_v = self.decoder_mean_x(self.h_x_decoded_v)
        self.w_decoded_mean_v = self.decoder_mean_w(self.h_w_decoded_v)
        self.v_decoded_mean_v = self.decoder_mean_v(self.h_v_decoded_v)
        # Shared XW Decoder
        # for the shared decoder, we use the same decoder_* layers, we just rewire them
        if deep:
            self.h_x_decoded_xw_2 = self.decoder_h_x_2(self.z_xw)
            self.h_w_decoded_xw_2 = self.decoder_h_w_2(self.z_xw)
            self.h_v_decoded_xw_2 = self.decoder_h_v_2(self.z_xw)
            self.h_x_decoded_xw = self.decoder_h_x(self.h_x_decoded_xw_2)
            self.h_w_decoded_xw = self.decoder_h_w(self.h_w_decoded_xw_2)
            self.h_v_decoded_xw = self.decoder_h_v(self.h_v_decoded_xw_2)
        else:
            self.h_x_decoded_xw = self.decoder_h_x(self.z_xw)
            self.h_w_decoded_xw = self.decoder_h_w(self.z_xw)
            self.h_v_decoded_xw = self.decoder_h_v(self.z_xw)
        self.x_decoded_mean_xw = self.decoder_mean_x(self.h_x_decoded_xw)
        self.w_decoded_mean_xw = self.decoder_mean_w(self.h_w_decoded_xw)
        self.v_decoded_mean_xw = self.decoder_mean_v(self.h_v_decoded_xw)
        # Shared XV Decoder
        if deep:
            self.h_x_decoded_xv_2 = self.decoder_h_x_2(self.z_xv)
            self.h_w_decoded_xv_2 = self.decoder_h_w_2(self.z_xv)
            self.h_v_decoded_xv_2 = self.decoder_h_v_2(self.z_xv)
            self.h_x_decoded_xv = self.decoder_h_x(self.h_x_decoded_xv_2)
            self.h_w_decoded_xv = self.decoder_h_w(self.h_w_decoded_xv_2)
            self.h_v_decoded_xv = self.decoder_h_v(self.h_v_decoded_xv_2)
        else:
            self.h_x_decoded_xv = self.decoder_h_x(self.z_xv)
            self.h_w_decoded_xv = self.decoder_h_w(self.z_xv)
            self.h_v_decoded_xv = self.decoder_h_v(self.z_xv)
        self.x_decoded_mean_xv = self.decoder_mean_x(self.h_x_decoded_xv)
        self.w_decoded_mean_xv = self.decoder_mean_w(self.h_w_decoded_xv)
        self.v_decoded_mean_xv = self.decoder_mean_v(self.h_v_decoded_xv)
        # Shared WV Decoder
        if deep:
            self.h_x_decoded_wv_2 = self.decoder_h_x_2(self.z_wv)
            self.h_w_decoded_wv_2 = self.decoder_h_w_2(self.z_wv)
            self.h_v_decoded_wv_2 = self.decoder_h_v_2(self.z_wv)
            self.h_x_decoded_wv = self.decoder_h_x(self.h_x_decoded_wv_2)
            self.h_w_decoded_wv = self.decoder_h_w(self.h_w_decoded_wv_2)
            self.h_v_decoded_wv = self.decoder_h_v(self.h_v_decoded_wv_2)
        else:
            self.h_x_decoded_wv = self.decoder_h_x(self.z_wv)
            self.h_w_decoded_wv = self.decoder_h_w(self.z_wv)
            self.h_v_decoded_wv = self.decoder_h_v(self.z_wv)
        self.x_decoded_mean_wv = self.decoder_mean_x(self.h_x_decoded_wv)
        self.w_decoded_mean_wv = self.decoder_mean_w(self.h_w_decoded_wv)
        self.v_decoded_mean_wv = self.decoder_mean_v(self.h_v_decoded_wv)
        # Shared XWV Decoder
        if deep:
            self.h_x_decoded_xwv_2 = self.decoder_h_x_2(self.z_xwv)
            self.h_w_decoded_xwv_2 = self.decoder_h_w_2(self.z_xwv)
            self.h_v_decoded_xwv_2 = self.decoder_h_v_2(self.z_xwv)
            self.h_x_decoded_xwv = self.decoder_h_x(self.h_x_decoded_xwv_2)
            self.h_w_decoded_xwv = self.decoder_h_w(self.h_w_decoded_xwv_2)
            self.h_v_decoded_xwv = self.decoder_h_v(self.h_v_decoded_xwv_2)
        else:
            self.h_x_decoded_xwv = self.decoder_h_x(self.z_xwv)
            self.h_w_decoded_xwv = self.decoder_h_w(self.z_xwv)
            self.h_v_decoded_xwv = self.decoder_h_v(self.z_xwv)
        self.x_decoded_mean_xwv = self.decoder_mean_x(self.h_x_decoded_xwv)
        self.w_decoded_mean_xwv = self.decoder_mean_w(self.h_w_decoded_xwv)
        self.v_decoded_mean_xwv = self.decoder_mean_v(self.h_v_decoded_xwv)
        
        self.y = custom_variational_layer.MmmVaeLoss(original_dim, alpha = alpha, beta_shared = beta_shared, beta_uni = beta_uni, gamma = gamma)([
            self.x, self.w, self.v,
            self.x_xw, self.w_xw,
            self.x_xv, self.v_xv,
            self.w_wv, self.v_wv,
            self.x_xwv, self.w_xwv, self.v_xwv,
            self.x_decoded_mean_x, self.x_decoded_mean_w, self.x_decoded_mean_v,
            self.w_decoded_mean_x, self.w_decoded_mean_w, self.w_decoded_mean_v,
            self.v_decoded_mean_x, self.v_decoded_mean_w, self.v_decoded_mean_v,
            self.x_decoded_mean_xw, self.w_decoded_mean_xw, self.v_decoded_mean_xw,
            self.x_decoded_mean_xv, self.w_decoded_mean_xv, self.v_decoded_mean_xv,
            self.x_decoded_mean_wv, self.w_decoded_mean_wv, self.v_decoded_mean_wv,
            self.x_decoded_mean_xwv, self.w_decoded_mean_xwv, self.v_decoded_mean_xwv,
            self.z_mean_x, self.z_log_var_x, 
            self.z_mean_w, self.z_log_var_w,
            self.z_mean_v, self.z_log_var_v,
            self.z_mean_xw, self.z_log_var_xw,
            self.z_mean_xv, self.z_log_var_xv,
            self.z_mean_wv, self.z_log_var_wv,
            self.z_mean_xwv, self.z_log_var_xwv])

    def get_model(self):
        return Model(inputs=[self.x, self.w, self.v,
                             self.x_xw, self.w_xw,
                             self.x_xv, self.v_xv,
                             self.w_wv, self.v_wv,
                             self.x_xwv, self.w_xwv, self.v_xwv], outputs = self.y)
    
    def get_encoder_mean_xwv(self):
        return Model([self.x_xwv,self.w_xwv, self.v_xwv], self.z_mean_xwv)
    
    def get_encoder_mean_xw(self):
        return Model([self.x_xw,self.w_xw], self.z_mean_xw)
    def get_encoder_mean_xv(self):
        return Model([self.x_xv,self.v_xv], self.z_mean_xv)
    def get_encoder_mean_wv(self):
        return Model([self.w_wv,self.v_wv], self.z_mean_wv)
    
    def get_encoder_mean_x(self):
        return Model(self.x, self.z_mean_x)
    def get_encoder_mean_w(self):
        return Model(self.w, self.z_mean_w)
    def get_encoder_mean_v(self):
        return Model(self.v, self.z_mean_v)
    def get_encoder_logvar_xwv(self):
        return Model([self.x_xwv,self.w_xwv, self.v_xwv], self.z_log_var_xwv)
    def get_encoder_logvar_xw(self):
        return Model([self.x_xw,self.w_xw], self.z_log_var_xw)
    def get_encoder_logvar_xv(self):
        return Model([self.x_xv,self.v_xv], self.z_log_var_xv)
    def get_encoder_logvar_wv(self):
        return Model([self.w_wv,self.v_wv], self.z_log_var_wv)
    def get_encoder_logvar_x(self):
        return Model(self.x, self.z_log_var_x)
    def get_encoder_logvar_w(self):
        return Model(self.w, self.z_log_var_w)
    def get_encoder_logvar_v(self):
        return Model(self.v, self.z_log_var_v)
    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        if self.deep:
            self._h_decoded_x_2 = self.decoder_h_x_2(self.decoder_input)
            self._h_decoded_w_2 = self.decoder_h_w_2(self.decoder_input)
            self._h_decoded_v_2 = self.decoder_h_v_2(self.decoder_input)
            self._h_decoded_x = self.decoder_h_x(self._h_decoded_x_2)
            self._h_decoded_w = self.decoder_h_w(self._h_decoded_w_2)
            self._h_decoded_v = self.decoder_h_v(self._h_decoded_v_2)
        else:
            self._h_decoded_x = self.decoder_h_x(self.decoder_input)
            self._h_decoded_w = self.decoder_h_w(self.decoder_input)
            self._h_decoded_v = self.decoder_h_v(self.decoder_input)
        self._x_decoded_mean = self.decoder_mean_x(self._h_decoded_x)
        self._w_decoded_mean = self.decoder_mean_w(self._h_decoded_w)
        self._v_decoded_mean = self.decoder_mean_v(self._h_decoded_v)
        return Model(self.decoder_input, [self._x_decoded_mean, self._w_decoded_mean, self._v_decoded_mean])
    
class MmVaeZero(sampling.Sampling, GenericVae):
    '''This is the MmVae without uni modal encoders. Instead, the unimodal inputs are the ground truth labels
    so that we can feed in zero inputs'''

    def __init__(self):
        pass

    def configure(self, original_dim_x, original_dim_w, intermediate_dim_x, intermediate_dim_w,
                                     intermediate_dim_2, intermediate_dim_shared, latent_dim, deep = True,
                                     beta = 1.):
        self.original_dim_x = original_dim_x
        self.original_dim_w = original_dim_w
        self.intermediate_dim_x = intermediate_dim_x
        self.intermediate_dim_w = intermediate_dim_w
        self.intermediate_dim_2 = intermediate_dim_2
        self.intermediate_dim_shared = intermediate_dim_shared
        self.latent_dim = latent_dim
        self.deep = deep
        self.beta = beta
        
        # Bimodal xw
        self.xw_x = Input(shape=(original_dim_x,))
        self.xw_w = Input(shape=(original_dim_w,))
        self.xw_h_x = Dense(intermediate_dim_x, activation='relu')(self.xw_x)
        self.xw_h_x_2 = Dense(intermediate_dim_2, activation='relu')(self.xw_h_x)
        self.xw_h_w = Dense(intermediate_dim_w, activation='relu')(self.xw_w)
        self.xw_h_w_2 = Dense(intermediate_dim_2, activation='relu')(self.xw_h_w)
        self.xw_h_shared = keras.layers.concatenate([self.xw_h_x_2, self.xw_h_w_2])
        self.xw_h = Dense(intermediate_dim_shared, activation='relu')(self.xw_h_shared)
        self.z_mean_xw = Dense(latent_dim)(self.xw_h)
        self.z_log_var_xw = Dense(latent_dim)(self.xw_h)


        # Unimodal x ground truth
        self.xx = Input(shape=(original_dim_x,), name='img_input_uni')
        # Unimodal w ground truth
        self.ww = Input(shape=(original_dim_w,), name='label_input_uni')
        
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z_xw = Lambda(self.randn, output_shape=(latent_dim,))([self.z_mean_xw, self.z_log_var_xw])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h_x_2 = Dense(intermediate_dim_2, activation='relu', name='dec_img_2')
        self.decoder_h_w_2 = Dense(intermediate_dim_2, activation='relu', name='dec_lab_2')
        self.decoder_h_x = Dense(intermediate_dim_x, activation='relu', name='dec_img')
        self.decoder_h_w = Dense(intermediate_dim_w, activation='relu', name='dec_lab')
        self.decoder_mean_x = Dense(original_dim_x, activation='sigmoid', name='output_img')
        self.decoder_mean_w = Dense(original_dim_w, activation='sigmoid', name='output_lab')
        

        self.h_x_decoded_2_xw = self.decoder_h_x_2(self.z_xw)
        self.h_w_decoded_2_xw = self.decoder_h_w_2(self.z_xw)
        self.h_x_decoded_xw = self.decoder_h_x(self.h_x_decoded_2_xw)
        self.h_w_decoded_xw = self.decoder_h_w(self.h_w_decoded_2_xw)
        self.x_decoded_mean_xw = self.decoder_mean_x(self.h_x_decoded_xw)
        self.w_decoded_mean_xw = self.decoder_mean_w(self.h_w_decoded_xw)
        
        self.y = custom_variational_layer.MmVaeLossZero(self.original_dim_x, self.original_dim_w, beta = beta)([self.xw_x, self.xw_w, self.xx, self.ww, self.x_decoded_mean_xw, self.w_decoded_mean_xw, self.z_mean_xw, self.z_log_var_xw])
        
    def get_model(self):
        return Model(inputs=[self.xw_x, self.xw_w, self.xx, self.ww], outputs = self.y)
 
    def get_encoder_mean_xw(self):
        return Model([self.xw_x, self.xw_w], self.z_mean_xw)
    def get_encoder_logvar_xw(self):
        return Model([self.xw_x, self.xw_w], self.z_log_var_xw)
    def get_encoder_mean_shared(self):
        return self.get_encoder_mean_xw()
    def get_encoder_logvar_shared(self):
        return self.get_encoder_logvar_xw()  
    
    def get_decoder(self):
        self.decoder_input = Input(shape=(self.latent_dim,))
        self._h_decoded_x_2 = self.decoder_h_x_2(self.decoder_input)
        self._h_decoded_w_2 = self.decoder_h_w_2(self.decoder_input)
        self._h_decoded_x = self.decoder_h_x(self._h_decoded_x_2)
        self._h_decoded_w = self.decoder_h_w(self._h_decoded_w_2)
        self._x_decoded_mean = self.decoder_mean_x(self._h_decoded_x)
        self._w_decoded_mean = self.decoder_mean_w(self._h_decoded_w)
        return Model(self.decoder_input, [self._x_decoded_mean, self._w_decoded_mean])