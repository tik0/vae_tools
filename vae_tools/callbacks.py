import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection
import vae_tools.viz
from scipy.stats import norm
import warnings


class TbLosses(keras.callbacks.Callback):
    '''Stores the outputs of the loss layer to tensorboard'''

    def __init__(self, log_dir, data = None, tag_prefix = "training", log_every=1, writer = None, **kwargs):
        warnings.warn("Losses are now added by the get_model function", DeprecationWarning, stacklevel=2)
        super().__init__( **kwargs)
        self.data = data
        self.history = {}
        self.epoch = []
        self.tag_prefix = tag_prefix
        self.log_every = log_every
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        def evaluate_data(data, tag_prefix = "", suffix = ""):    
            # Get the output layer names (we only have loss layers as output)
            output_layer_names = [output_layer.name for output_layer in self.model.output]
            # Get the mean error over all batches in this epoch
            output_layer_values = np.mean(self.model.predict(data), axis = 1)
            # Store it to the history
            for k, v in zip(output_layer_names, output_layer_values):
                tag = tag_prefix + "/" + k.split('/')[0] + '/' + suffix
                tf.summary.scalar(tag, v, epoch)
                self.writer.flush()

        if epoch%self.log_every==0:
            evaluate_data(self.data, tag_prefix = self.tag_prefix)
            self.writer.flush()
        #super().on_epoch_end(epoch, logs)
        
    def on_train_begin(self, logs=None):
        return
 
    def on_train_end(self, logs=None):
        self.writer.close()
 
    def on_epoch_begin(self, epoch, logs=None):
        return
    
    def on_batch_begin(self, batch, logs=None):
        return
 
    def on_batch_end(self, batch, logs=None):
        return


class Losses(keras.callbacks.Callback):
    '''Create a history which stores the ouputs of the loss layer after every epoch

    The object of this class holds its own history attribute, which needs to be
    investigated explicitly (by e.g. vae_tools.viz.plot_losses())
    '''
    def __init__(self, data, only_train_end = False):
        warnings.warn("Losses are now added by the get_model function", DeprecationWarning, stacklevel=2)
        self.data = data
        self.history = {}
        self.epoch = []
        self.only_train_end = only_train_end

    def on_train_begin(self, logs=None):
        return
 
    def on_train_end(self, logs=None):
        if self.only_train_end:
            self.store_losses(0, logs=logs)
 
    def on_epoch_begin(self, epoch, logs=None):
        return
 
    def on_epoch_end(self, epoch, logs=None):
        if not self.only_train_end:
            self.store_losses(epoch, logs=logs)
 
    def on_batch_begin(self, batch, logs=None):
        return
 
    def on_batch_end(self, batch, logs=None):
        return
    
    def store_losses(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        # Get the output layer names (we only have loss layers as output)
        output_layer_names = [output_layer.name for output_layer in self.model.output]
        # Get the mean error over all batches in this epoch
        output_layer_values = np.mean(self.model.predict(self.data), axis = 1)
        # Store it to the history
        for k, v in zip(output_layer_names, output_layer_values):
            self.history.setdefault(k, []).append(v)


def get_tf_summary_image(plot_buf):
    img = Image.open(plot_buf)
    return K.expand_dims(tf.convert_to_tensor(np.array(img)), 0)


class TbDecoding2dGaussian(keras.callbacks.Callback):
    '''Plot the decoding of a images decoder assuming gaussian 2d prior in the latent space'''
    def __init__(self, log_dir, decoder_model, decoded_image_size = [28,28,1], num_images = 15, tag = "validation/decoding", log_every=1, writer = None, **kwargs):
        super().__init__(**kwargs)
        self.decoder_model = decoder_model
        self.tag = tag
        self.decoded_image_size = decoded_image_size
        self.num_images = num_images
        self.history = {}
        self.epoch = []
        self.log_every = log_every
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        def plot_decoder(decoder_model):
            # store a 2D manifold of the digits
            n = self.num_images  # figure with nxn digits
            figure = np.zeros((self.decoded_image_size[0] * n, self.decoded_image_size[1] * n))
            # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            # to produce values of the latent variables z, since the prior of the latent space is Gaussian
            grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.array([[xi, yi]])
                    x_decoded = decoder_model.predict(z_sample)
                    digit = x_decoded[0].reshape(self.decoded_image_size[0], self.decoded_image_size[1])
                    figure[(n-1-i) * self.decoded_image_size[1]: ((n-1-i) + 1) * self.decoded_image_size[1],
                           j * self.decoded_image_size[0]: (j + 1) * self.decoded_image_size[0]] = digit

            plt.figure(figsize=(10, 10))
            plt.imshow(figure, cmap='Greys_r')
            #plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close() # needs to be called, otherwise the figures show up in the end

            buf.seek(0)
            return buf

        if epoch%self.log_every==0:
            img = get_tf_summary_image(plot_buf=plot_decoder(decoder_model=self.decoder_model))
            tf.summary.image(self.tag, img, epoch)
            self.writer.flush()
        #super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        return
 
    def on_train_end(self, logs=None):
        self.writer.close()
 
    def on_epoch_begin(self, epoch, logs=None):
        return
 
    def on_batch_begin(self, batch, logs=None):
        return
 
    def on_batch_end(self, batch, logs=None):
        return
        
class TbEmbedding(keras.callbacks.Callback):
    '''Plot the embeddings of an encoder to scatter plot and store as image to tensorboard'''
    def __init__(self, log_dir, data, encoder_model, labels, images = None, tag = "validation/embedding", perplexity = 30, log_every=1, writer = None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.encoder_model = encoder_model
        self.tag = tag
        self.labels = labels
        self.images = images
        self.perplexity = perplexity
        self.history = {}
        self.epoch = []
        self.log_every = log_every
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        def plot_encoder(encoder_model, data = None):
            data_embedded = np.array(encoder_model.predict(data))
            latent_dim = data_embedded.shape[1]
            if latent_dim == 2: # Display the VAE embedding
                f, ax = vae_tools.viz.plot_embedding(embeddings = data_embedded, labels = self.labels,
                               images = self.images, figsize=(10,10), dpi=155)
            else: # Display the tSNE embedding
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, n_iter = 250, perplexity = self.perplexity)
                X_tsne = tsne.fit_transform(data_embedded)
                f, ax = vae_tools.viz.plot_embedding(embeddings = X_tsne, labels = self.labels, images = self.images,
                               figsize=(10,10), dpi=155)
            # Get the buffer
            buf = io.BytesIO()
            f.savefig(buf, format='png')
            plt.close() # needs to be called, otherwise the figures show up in the end
            buf.seek(0)
            return buf

        if epoch%self.log_every==0:
            img = get_tf_summary_image(plot_buf=plot_encoder(encoder_model=self.encoder_model, data = self.data                                                                             ))
            tf.summary.image(self.tag, img, epoch)
            self.writer.flush()
        #super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        return
 
    def on_train_end(self, logs=None):
        self.writer.close()
 
    def on_epoch_begin(self, epoch, logs=None):
        return
 
    def on_batch_begin(self, batch, logs=None):
        return
 
    def on_batch_end(self, batch, logs=None):
        return