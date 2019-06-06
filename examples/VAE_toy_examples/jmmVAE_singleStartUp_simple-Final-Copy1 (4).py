
# coding: utf-8

# Iterration: 0001
# 
# # Info
# 
# This training has the following properties
# * First try of implementation
# 

# In[1]:


'''This script demonstrates how to build a multi-modal variational autoencoder with Keras.
'''
import sys, os
import matplotlib
try:
    os.environ['USE_AGG']
    matplotlib.use('Agg') # Use this to plot to Nirvana (assuming we run in a terminal)
    print("matplotlib: Use agg")
except:
    pass
import matplotlib.pyplot as plt
print("matplotlib uses: ", matplotlib.rcParams['backend']) 

# Set random seeds (https://machinelearningmastery.com/reproducible-results-neural-networks-keras/)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
from scipy.stats import norm
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import keras
from numpy import random
from keras.datasets import mnist
import tensorflow as tf

sys.path.append(os.path.expanduser('~/notebook'))
from tools import plot_confusion_matrix, plot_model, layers, sampling, custom_variational_layer

from tools import nb_tools, viz, loader, build_model, sanity, sampling, custom_variational_layer, metrics
from tools import build_model

nb_tools.notebook_resize()
# sanity.check()


# In[2]:


# Setup
batch_size = 1024
original_dim_img = 2
original_dim_label = 2
intermediate_dim_img = 128
intermediate_dim_label = 128
intermediate_dim_shared = 64
latent_dim = 2
epochs = 2
epsilon_std = 1.

# MMVAE
#working
beta_shared = .01 # 0.1 # KL of multi modal encoder
beta_uni = .01 # 0.001 # KL of uni modal encoder
alpha = .01 # 1. # KL between uni- and shared encoder
gamma = 1. # reconstruction losses of single encoders
#test
#beta_shared = 0.001 # 0.1 # KL of multi modal encoder
#beta_uni = 0.001 # 0.001 # KL of uni modal encoder
#alpha = 0.001 # 1. # KL between uni- and shared encoder
#gamma = 1. # reconstruction losses of single encoders

# JMMVAE
#beta_shared = .01 # 0.1 # KL of multi modal encoder
#beta_uni = 0. # 0.001 # KL of uni modal encoder
#alpha = .01 # 1. # KL between uni- and shared encoder
#gamma = 0. # reconstruction losses of single encoders

# check if the parameters have been set from the environment

try:
    _beta_shared = np.float(os.environ['BETA_SHARED'])
    _beta_uni = np.float(os.environ['BETA_UNI'])
    _alpha = np.float(os.environ['ALPHA'])
    _gamma = np.float(os.environ['GAMMA'])
    _epochs = np.int(os.environ['EPOCHS'])
    beta_shared = _beta_shared
    beta_uni = _beta_uni
    alpha = _alpha
    gamma = _gamma
    epochs = _epochs
    print("WARNING: Set parameters from environment")
except:
    pass

use_binary_classifier_lidar_camera = False
use_vae_classifier_lidar_camera = False
use_mnist = False
use_didactical = True

setup_mask = 0b0000 | (use_binary_classifier_lidar_camera<<0 | use_vae_classifier_lidar_camera<<1 | use_mnist<<2 | use_didactical<<3)

#setup_mask = [use_didactical,use_mnist, use_vae_classifier_lidar_camera, use_binary_classifier_lidar_camera]
setup_str = "_bshared-" + str(beta_shared) + "_buni-" + str(beta_uni) + "_a-" + str(alpha) + "_g-" + str(gamma) + "_e-" + str(epochs)
prefix_str = "MMVAE_" + "setup-" + bin(setup_mask) + "_"

print(prefix_str)

model_obj = build_model.MmVae()
model_obj.configure(original_dim_x = original_dim_img,
                      original_dim_w = original_dim_label,
                      intermediate_dim_x = intermediate_dim_img,
                      intermediate_dim_w = intermediate_dim_label,
                      intermediate_dim_shared = intermediate_dim_shared,
                      latent_dim = latent_dim,
                      alpha = alpha, beta_shared = beta_shared, beta_uni = beta_uni, gamma = gamma)
vae = model_obj.get_model()
vae.compile(optimizer='rmsprop', loss=None)
# plot_model.plot_model(vae, file = 'jvae_shared', print_svg = False, verbose = True)


# In[3]:


# Define training set
if use_binary_classifier_lidar_camera:
    size = 30000 # Needs to be devidable by 3

    blob_red = np.matlib.repmat(np.append(np.ones((1,1)), np.zeros((1,1)), axis=1), np.int(size * 2 / 3), 1)
    blob_green = np.matlib.repmat(np.append(np.zeros((1,1)), np.ones((1,1)), axis=1), np.int(size / 3), 1)
    x_set = np.concatenate((blob_red, blob_green))
    x_set_gt = x_set[:,1] > .5 # Green is True, Red is False

    shape_round_red = np.matlib.repmat(np.append(np.ones((1,1)), np.zeros((1,1)), axis=1), np.int(size / 3), 1)
    shape_cube_red = np.matlib.repmat(np.append(np.zeros((1,1)), np.ones((1,1)), axis=1), np.int(size / 3), 1)
    shape_round_green = np.matlib.repmat(np.append(np.ones((1,1)), np.zeros((1,1)), axis=1), np.int(size / 3), 1)
    w_set = np.concatenate((shape_round_red, shape_cube_red, shape_round_green))
    w_set_gt = w_set[:,1] > .5 # Cube is True, Round is False

    # Define the ground truth set
    gt_set = -np.ones(w_set_gt.shape)
    gt_set[(x_set_gt == False) & (w_set_gt == False)] = 0 # Red and Round
    gt_set[(x_set_gt == False) & (w_set_gt == True)] = 1  # Red and Cube
    gt_set[(x_set_gt == True)  & (w_set_gt == False)] = 2 # Green and Round
    # gt_set[(x_set_gt == True) & (w_set_gt == True)] = 4 # Green and Cube: Does not exists

    if add_noise_to_input:
        noise_amplitude = 0.3
        x_set_tmp = np.zeros(x_set.shape)
        x_set_tmp[x_set < 0.5] = noise_amplitude / 2 + np.random.random(size=x_set_tmp[x_set < 0.5].shape) * noise_amplitude
        x_set_tmp[x_set > 0.5] = 1 - noise_amplitude / 2 + np.random.random(size=x_set_tmp[x_set > 0.5].shape) * noise_amplitude
        x_set = x_set_tmp
        w_set_tmp = np.zeros(w_set.shape)
        w_set_tmp[w_set < 0.5] = noise_amplitude / 2 + np.random.random(size=w_set_tmp[w_set < 0.5].shape) * noise_amplitude
        w_set_tmp[w_set > 0.5] = 1 - noise_amplitude / 2 + np.random.random(size=w_set_tmp[w_set > 0.5].shape) * noise_amplitude
        w_set = w_set_tmp
        #x_set = x_set + np.random.random(size=x_set.shape) * noise_amplitude
        #x_set = x_set / np.sum(x_set, axis=1)[:,np.newaxis]
        #w_set = w_set + np.random.random(size=w_set.shape) * noise_amplitude
        #w_set = w_set / np.sum(w_set, axis=1)[:, np.newaxis]
if use_vae_classifier_lidar_camera:
    # Define training set by VAEs
    model_generic = build_model.GenericVae()
    decoder_mean_lidar = model_generic.load_model("lidar_conv_encoder_mean")
    decoder_mean_cameraRG = model_generic.load_model("cameraRG_conv_encoder_mean")

    # Generate the set by loading the data which has been used for training the lidar and camera
    with np.load('cameraRG_conv_set.npz') as data:
        xx_set  = data['X_set']
        xx_used = data['X_used']
        xx_label = data['X_label']
    with np.load('lidar_conv_set.npz') as data:
        ww_set  = data['X_set']
        ww_used = data['X_used']
        ww_label = data['X_label']
    mask = ww_used & xx_used
    ww_set = ww_set[mask,:]
    xx_set = xx_set[mask,:]
    size = np.sum(mask)
    #zx_set = np.zeros((num_samples, original_dim_img))
    #zw_set = np.zeros((num_samples, original_dim_label))
    x_set = decoder_mean_cameraRG.predict(xx_set)
    w_set = decoder_mean_lidar.predict(ww_set)
    gt_set = xx_label
    #for idx in np.arange(num_samples):
    #    zx_set[idx,:] = decoder_mean_lidar.predict(x_set[])
if use_mnist:
    # Define training set by VAEs
    model_generic = build_model.GenericVae()
    decoder_mean_mnist = model_generic.load_model("mnist_conv_encoder_mean")
    images, gt_set, _, _ = loader.mnist((28, 28, 1))
    x_set = decoder_mean_mnist.predict(images)
    size = len(gt_set)
    # Project labels to the unit cicle
    num_mnist_class = 10
    label_pose_rad = np.linspace(0, 2*np.pi, num=num_mnist_class, endpoint=False, dtype=float)
    w_set = np.zeros(x_set.shape)
    for idx in np.arange(num_mnist_class):
        mask = gt_set == idx
        w_set[mask,0] = np.cos(label_pose_rad[idx])
        w_set[mask,1] = np.sin(label_pose_rad[idx])
    # Add some Gaussion noise
    noise_amp = 0.1
    w_set = w_set + noise_amp * np.random.randn(w_set.shape[0], w_set.shape[1])
    
if use_didactical:
    _, gt_set, _, _ = loader.mnist()
    size = len(gt_set)
    num_mnist_class = 10
    label_pose_rad = np.linspace(0, 2*np.pi, num=num_mnist_class, endpoint=True, dtype=float) # collapse first and last mean
    label_pose_lin_x_2 = np.linspace(.25, .75, num=2, endpoint=True, dtype=float)
    label_pose_lin_x_3 = np.linspace(0, 1., num=3, endpoint=True, dtype=float)
    label_pose_lin_y = np.linspace(0, 1., num=4, endpoint=True, dtype=float)
    x_set = np.zeros((size,2))
    w_set = np.zeros(x_set.shape)
    for idx in np.arange(num_mnist_class):
        mask = gt_set == idx
        w_set[mask,0] = np.cos(label_pose_rad[idx])
        w_set[mask,1] = np.sin(label_pose_rad[idx])
    
    #x_set[gt_set == 0,:] = [label_pose_lin_y[0], label_pose_lin_x_3[0]]
    x_set[gt_set == 1,:] = [label_pose_lin_y[0], label_pose_lin_x_3[1]]
    x_set[gt_set == 2,:] = [label_pose_lin_y[0], label_pose_lin_x_3[2]]
    x_set[gt_set == 3,:] = [label_pose_lin_y[1], label_pose_lin_x_2[0]]
    x_set[gt_set == 4,:] = [label_pose_lin_y[1], label_pose_lin_x_2[1]]
    #x_set[gt_set == 5,:] = [label_pose_lin_y[2], label_pose_lin_x_3[0]]
    #x_set[gt_set == 6,:] = [label_pose_lin_y[2], label_pose_lin_x_3[1]]
    #x_set[gt_set == 7,:] = [label_pose_lin_y[2], label_pose_lin_x_3[2]]
    x_set[gt_set == 5,:] = [label_pose_lin_y[2], label_pose_lin_x_3[1]] # collapse
    x_set[gt_set == 6,:] = [label_pose_lin_y[2], label_pose_lin_x_3[1]] # collapse
    x_set[gt_set == 7,:] = [label_pose_lin_y[2], label_pose_lin_x_3[1]] # collapse
    x_set[gt_set == 8,:] = [label_pose_lin_y[3], label_pose_lin_x_2[0]]
    x_set[gt_set == 0,:] = [label_pose_lin_y[3], label_pose_lin_x_2[0]] # collapse
    x_set[gt_set == 9,:] = [label_pose_lin_y[3], label_pose_lin_x_2[1]]   
    # Add some Gaussion noise
    noise_amp_x = 0.06
    noise_amp_w = 0.1
    w_set = w_set + noise_amp_w * np.random.randn(w_set.shape[0], w_set.shape[1])
    x_set = x_set + noise_amp_x * np.random.randn(x_set.shape[0], x_set.shape[1])
    
    
normalize = True
if normalize:
    min_max = True
    mean_var = False
    if min_max:
        x_set[:,0] = x_set[:,0] - np.amin(x_set[:,0])
        x_set[:,1] = x_set[:,1] - np.amin(x_set[:,1])
        x_set[:,0] = x_set[:,0] / np.amax(x_set[:,0])
        x_set[:,1] = x_set[:,1] / np.amax(x_set[:,1])
        w_set[:,0] = w_set[:,0] - np.amin(w_set[:,0])
        w_set[:,1] = w_set[:,1] - np.amin(w_set[:,1])
        w_set[:,0] = w_set[:,0] / np.amax(w_set[:,0])
        w_set[:,1] = w_set[:,1] / np.amax(w_set[:,1])
    if mean_var:
        x_set[:,0] = x_set[:,0] - np.mean(x_set[:,0])
        x_set[:,1] = x_set[:,1] - np.mean(x_set[:,1])
        x_set[:,0] = x_set[:,0] / np.var(x_set[:,0])
        x_set[:,1] = x_set[:,1] / np.var(x_set[:,1])
        w_set[:,0] = w_set[:,0] - np.mean(w_set[:,0])
        w_set[:,1] = w_set[:,1] - np.mean(w_set[:,1])
        w_set[:,0] = w_set[:,0] / np.var(w_set[:,0])
        w_set[:,1] = w_set[:,1] / np.var(w_set[:,1])

# Shuffel and define training and test sets
shuffel_index = np.arange(size)
random.shuffle(shuffel_index)
w_set_shuffel = np.copy(w_set[shuffel_index,:])
x_set_shuffel = np.copy(x_set[shuffel_index,:])
gt_set_shuffel = np.copy(gt_set[shuffel_index])
train_size = np.int(len(w_set) * 0.99)
w_train = w_set_shuffel[:train_size,:]
w_test = w_set_shuffel[train_size:,:]
x_train = x_set_shuffel[:train_size,:]
x_test = x_set_shuffel[train_size:,:]
gt_train = gt_set_shuffel[:train_size]
gt_test = gt_set_shuffel[train_size:]

x_train_shared = x_train
w_train_shared = w_train


# In[4]:


# Plot the classes
fig, ax = plt.subplots(ncols=2, figsize=(20,15))
x_plot = x_test
w_plot = w_test
color_plot = gt_test
x1_plot = ax[0].scatter(x_plot[:,0], x_plot[:,1], c = color_plot, cmap = 'tab10')
x2_plot = ax[1].scatter(w_plot[:,0], w_plot[:,1], c = color_plot, cmap = 'tab10')

asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
ax[0].set_aspect(asp)
asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[1].set_aspect(asp)
# remove the x and y ticks
#for axes in ax:
#    axes.set_xticks([])
#    axes.set_yticks([])

plt.show(block=False)
plt.savefig(prefix_str + "input" + ".svg")


# In[5]:


# Train the shared VAE
if False:
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    vae.fit(x = [x_train_shared, w_train_shared, x_train_shared, w_train_shared],
            y = None,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([x_test, w_test, x_test, w_test], None),
            callbacks=[tbCallBack])


# In[6]:


# The elbo check
def _elbo_check(decoder, encoder_mean, encoder_logvar, data, data_expected_output, sample_from_enc_dec = False, batch_size = 128):
    #data = np.array(data)
    #data_expected_output = np.array(data_expected_output)
    # Latent mean => (N,2)
    encoded_mean    = encoder_mean.predict(data, batch_size=batch_size)
    # Latent var => (N,2)
    encoded_log_var = encoder_logvar.predict(data, batch_size=batch_size)
    # Sum KL over latent dimensions => (N,)
    kl = np.sum(metrics.kl_loss_n(encoded_mean, encoded_log_var) , axis=-1)
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
            diff[:,idx] = np.sum(np.square(np.mean(_encoded_mean - _data_expected_output[:,[idx],:], axis=1)),axis=-1)
    # return the reconstruction loss of first and second modality, and kl loss
    return diff[0,:], diff[1,:], kl

def elbo_check(model_obj, x_set, w_set, _elbo_xw, _elbo_x, _elbo_w):
    
    decoder = model_obj.get_decoder()
    xw_rec_x, xw_rec_w, xw_kl = _elbo_check(decoder, model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [x_set, w_set], [x_set, w_set])
    x_rec_x, x_rec_w, x_kl = _elbo_check(decoder, model_obj.get_encoder_mean_x(), model_obj.get_encoder_logvar_x(), x_set, [x_set, w_set])
    w_rec_x, w_rec_w, w_kl = _elbo_check(decoder, model_obj.get_encoder_mean_w(), model_obj.get_encoder_logvar_w(), w_set, [x_set, w_set])

    elbo_xw = np.sum( xw_rec_x + xw_rec_w - xw_kl)
    elbo_x = np.sum( x_rec_x + x_rec_w - x_kl)
    elbo_w = np.sum(  w_rec_x + w_rec_w - w_kl)
    return np.append(_elbo_xw, elbo_xw), np.append(_elbo_x, elbo_x), np.append(_elbo_w, elbo_w)


# In[7]:


elbo_xw = np.array([])
elbo_x  = np.array([])
elbo_w  = np.array([])
for idx in np.arange(epochs):
    # ELBO test
    elbo_xw, elbo_x, elbo_w = elbo_check(model_obj, x_set, w_set, elbo_xw, elbo_x, elbo_w)
    vae.fit(x = [x_train_shared, w_train_shared, x_train_shared, w_train_shared],
        y = None,
        shuffle=True,
        epochs=1,
        batch_size=batch_size,
        validation_data=([x_test, w_test, x_test, w_test], None))
# ELBO test
elbo_xw, elbo_x, elbo_w = elbo_check(model_obj, x_set, w_set, elbo_xw, elbo_x, elbo_w)


# In[8]:


# Show the elbo
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.array([elbo_xw, elbo_x, elbo_w]).T)
ax.legend(('elbo_xw', 'elbo_x', 'elbo_w'))
plt.savefig(prefix_str + "elbo" + setup_str + ".svg")
plt.show(block=False)
np.savez_compressed(prefix_str + "elbo" + setup_str, elbo_xw=elbo_xw, elbo_x=elbo_x, elbo_w=elbo_w)


# In[9]:


#layers.set_layerweights_trainable(vae_shared, ln_full = 'dec_img')
#layers.set_layerweights_trainable(vae_shared, ln_full = 'dec_lab')
#layers.set_layerweights_trainable(vae_shared, ln_full = 'output_img')
#layers.set_layerweights_trainable(vae_shared, ln_full = 'output_lab')
# layers.reset_layerweights(vae)


# In[10]:


# build a model to project inputs on the latent space
encoder_shared = model_obj.get_encoder_mean_shared()
encoder_x = model_obj.get_encoder_mean_x()
encoder_w = model_obj.get_encoder_mean_w()




# Single sample: red & round
use_single_sample = False
x_set_ = np.array([[1,0]])
w_set_ = np.array([[1,0]])

cmap = 'tab10'
#cmap = 'rainbow'
fig, ax = plt.subplots(ncols=3, figsize=(20,15))
dropout_values = np.ones((len(x_test), intermediate_dim_img+intermediate_dim_label))
recloss_values = np.ones((len(x_test), 1))

# display a 2D plot of the digit classes in the latent space
#ax[0].set_title("input (x,w)")
encoded_1 = encoder_shared.predict([x_test, w_test], batch_size=batch_size)
x1_plot = ax[0].scatter(encoded_1[:, 0], encoded_1[:, 1], c=gt_test, cmap=cmap)

#ax[1].set_title("input (x)")
dropout_values_noLabel = np.copy(dropout_values)
dropout_values_noLabel[:,intermediate_dim_img:] = 0;
encoded_2 = encoder_x.predict(x_test, batch_size=batch_size)
x2_plot = ax[1].scatter(encoded_2[:, 0], encoded_2[:, 1], c=gt_test, cmap=cmap)

#ax[2].set_title("input (w)")
dropout_values_noImg = np.copy(dropout_values)
dropout_values_noImg[:,:intermediate_dim_img] = 0;
encoded_3 = encoder_w.predict(w_test, batch_size=batch_size)
x3_plot = ax[2].scatter(encoded_3[:, 0], encoded_3[:, 1], c=gt_test, cmap=cmap)

ymin = np.amin([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
ymax = np.amax([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
xmin = np.amin([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
xmax = np.amax([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])

# ymin = -0.15
# ymax = 0.15
# xmin = -0.1
# xmax = 0.1

ax[0].set_xlim(xmin,xmax)
ax[1].set_xlim(xmin,xmax)
ax[2].set_xlim(xmin,xmax)
ax[0].set_ylim(ymin,ymax)
ax[1].set_ylim(ymin,ymax)
ax[2].set_ylim(ymin,ymax)

asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
ax[0].set_aspect(asp)
asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[1].set_aspect(asp)
asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
ax[2].set_aspect(asp)
if not use_single_sample and cmap != 'tab10':
    plt.colorbar(x3_plot, ax=list(ax))
# remove the x and y ticks
for axes in ax:
    axes.set_xticks([])
    axes.set_yticks([])
plt.savefig(prefix_str + "classes" + setup_str + ".svg")

plt.show(block=False)


# In[11]:


# build a model to project variances into the latent space
encoder_var = model_obj.get_encoder_logvar_shared()
encoder_var_x = model_obj.get_encoder_logvar_x()
encoder_var_w = model_obj.get_encoder_logvar_w()

fig, ax = plt.subplots(ncols=3, figsize=(20,15))
_shared_test_encoded_log_var = encoder_var.predict([x_test, w_test], batch_size=batch_size)
_x_test_encoded_log_var = encoder_var_x.predict(x_test, batch_size=batch_size)
_w_test_encoded_log_var = encoder_var_w.predict(w_test, batch_size=batch_size)
# sum of powered exponents
shared_test_encoded_var = np.sqrt(np.sum(np.exp(_shared_test_encoded_log_var / 2.0)**2.0, axis=1))
x_test_encoded_var      = np.sqrt(np.sum(np.exp(_x_test_encoded_log_var / 2.0)**2.0, axis=1))
w_test_encoded_var      = np.sqrt(np.sum(np.exp(_w_test_encoded_log_var / 2.0)**2.0, axis=1))

shared_test_encoded_var = np.sum(metrics.kl_loss_n(encoded_1, _shared_test_encoded_log_var), axis=1)
x_test_encoded_var      = np.sum(metrics.kl_loss_n(encoded_2, _x_test_encoded_log_var), axis=1)
w_test_encoded_var      = np.sum(metrics.kl_loss_n(encoded_3, _w_test_encoded_log_var), axis=1)

decoder = model_obj.get_decoder()
x_sample = x_test
w_sample = w_test
# Generate losses via sampling multiple times from latent space
#xw_rec_x, xw_rec_w, xw_kl = _elbo_check(decoder, model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [x_sample, w_sample], [x_sample, w_sample])
#x_rec_x, x_rec_w, x_kl = _elbo_check(decoder, model_obj.get_encoder_mean_x(), model_obj.get_encoder_logvar_x(), x_sample, [x_sample, w_sample])
#w_rec_x, w_rec_w, w_kl = _elbo_check(decoder, model_obj.get_encoder_mean_w(), model_obj.get_encoder_logvar_w(), w_sample, [x_sample, w_sample])
# Generate losses via sampling multiple times from latent space
xw_rec_x, xw_rec_w, xw_kl = _elbo_check(model_obj.get_encoder_decoder_shared(), model_obj.get_encoder_mean_shared(), model_obj.get_encoder_logvar_shared(), [x_sample, w_sample], [x_sample, w_sample], sample_from_enc_dec = True)
x_rec_x, x_rec_w, x_kl =    _elbo_check(model_obj.get_encoder_decoder_x(), model_obj.get_encoder_mean_x(), model_obj.get_encoder_logvar_x(), x_sample, [x_sample, w_sample], sample_from_enc_dec = True)
w_rec_x, w_rec_w, w_kl =    _elbo_check(model_obj.get_encoder_decoder_w(), model_obj.get_encoder_mean_w(), model_obj.get_encoder_logvar_w(), w_sample, [x_sample, w_sample], sample_from_enc_dec = True)

# calculate the energy
elbo_xw = xw_rec_x + xw_rec_w - xw_kl
elbo_x = x_rec_x + x_rec_w - x_kl
elbo_w = w_rec_x + w_rec_w - w_kl
shared_test_encoded_var = elbo_xw
x_test_encoded_var = elbo_x
w_test_encoded_var = elbo_w

# Show only one column
#column = 1
#shared_test_encoded_var = np.exp(shared_test_encoded_log_var[:,column])
#x_test_encoded_var      = np.exp(x_test_encoded_log_var[:,column])
#w_test_encoded_var      = np.exp(w_test_encoded_log_var[:,column])

use_global_minmax = True
if use_global_minmax:
    vmin = np.amin([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
    vmax = np.amax([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
    print("global min: ", vmin)
    print("global max: ", vmax)
else:
    vmin = None
    vmax = None

# display a 2D plot of the digit classes in the latent space
#ax[0].set_title("input (x,w)")
x1_plot = ax[0].scatter(encoded_1[:, 0], encoded_1[:, 1], c=shared_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
#ax[1].set_title("input (x)")
x2_plot = ax[1].scatter(encoded_2[:, 0], encoded_2[:, 1], c=x_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
#ax[2].set_title("input (w)")
x3_plot = ax[2].scatter(encoded_3[:, 0], encoded_3[:, 1], c=w_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
ymin = np.amin([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
ymax = np.amax([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
xmin = np.amin([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
xmax = np.amax([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
ax[0].set_xlim(xmin,xmax)
ax[1].set_xlim(xmin,xmax)
ax[2].set_xlim(xmin,xmax)
ax[0].set_ylim(ymin,ymax)
ax[1].set_ylim(ymin,ymax)
ax[2].set_ylim(ymin,ymax)

asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
ax[0].set_aspect(asp)
asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[1].set_aspect(asp)
asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
ax[2].set_aspect(asp)
if use_global_minmax:
    plt.colorbar(x3_plot, ax=list(ax), fraction=0.014, pad=0.04)
else:
    plt.colorbar(x1_plot, ax=ax[0])
    plt.colorbar(x2_plot, ax=ax[1])
    plt.colorbar(x3_plot, ax=ax[2])
for axes in ax:
    axes.set_xticks([])
    axes.set_yticks([])
plt.savefig(prefix_str + "energy" + setup_str + ".svg")
plt.show(block=False)


# In[12]:


fig, ax = plt.subplots(ncols=3, figsize=(20,15))

# calculate the energy
shared_test_encoded_var = xw_rec_x + xw_rec_w - elbo_xw
x_test_encoded_var = x_rec_x + x_rec_w - elbo_x
w_test_encoded_var = w_rec_x + w_rec_w - elbo_w

# Show only one column
#column = 1
#shared_test_encoded_var = np.exp(shared_test_encoded_log_var[:,column])
#x_test_encoded_var      = np.exp(x_test_encoded_log_var[:,column])
#w_test_encoded_var      = np.exp(w_test_encoded_log_var[:,column])

use_global_minmax = True
if use_global_minmax:
    vmin = np.amin([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
    vmax = np.amax([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
    print("global min: ", vmin)
    print("global max: ", vmax)
else:
    vmin = None
    vmax = None

# display a 2D plot of the digit classes in the latent space
#ax[0].set_title("input (x,w)")
x1_plot = ax[0].scatter(encoded_1[:, 0], encoded_1[:, 1], c=shared_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
#ax[1].set_title("input (x)")
x2_plot = ax[1].scatter(encoded_2[:, 0], encoded_2[:, 1], c=x_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
#ax[2].set_title("input (w)")
x3_plot = ax[2].scatter(encoded_3[:, 0], encoded_3[:, 1], c=w_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
ymin = np.amin([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
ymax = np.amax([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
xmin = np.amin([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
xmax = np.amax([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
ax[0].set_xlim(xmin,xmax)
ax[1].set_xlim(xmin,xmax)
ax[2].set_xlim(xmin,xmax)
ax[0].set_ylim(ymin,ymax)
ax[1].set_ylim(ymin,ymax)
ax[2].set_ylim(ymin,ymax)

asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
ax[0].set_aspect(asp)
asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[1].set_aspect(asp)
asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
ax[2].set_aspect(asp)
if use_global_minmax:
    plt.colorbar(x3_plot, ax=list(ax), fraction=0.014, pad=0.04)
else:
    plt.colorbar(x1_plot, ax=ax[0])
    plt.colorbar(x2_plot, ax=ax[1])
    plt.colorbar(x3_plot, ax=ax[2])
for axes in ax:
    axes.set_xticks([])
    axes.set_yticks([])
plt.savefig(prefix_str + "kldivergence" + setup_str + ".svg")
plt.show(block=False)


# In[13]:


if False:
    fig, ax = plt.subplots(nrows=3, figsize=(20,15))
    shared_test_encoded_log_var = encoder_var.predict([x_test, w_test], batch_size=batch_size)
    x_test_encoded_log_var = encoder_var_x.predict(x_test, batch_size=batch_size)
    w_test_encoded_log_var = encoder_var_w.predict(w_test, batch_size=batch_size)
    # sum of powered exponents
    shared_test_encoded_var = np.sqrt(np.sum(np.exp(shared_test_encoded_log_var), axis=1) / 2.0)
    x_test_encoded_var      = np.sqrt(np.sum(np.exp(x_test_encoded_log_var), axis=1) / 2.0)
    w_test_encoded_var      = np.sqrt(np.sum(np.exp(w_test_encoded_log_var), axis=1) / 2.0)
    # Show only one column
    #column = 1
    #shared_test_encoded_var = np.exp(shared_test_encoded_log_var[:,column])
    #x_test_encoded_var      = np.exp(x_test_encoded_log_var[:,column])
    #w_test_encoded_var      = np.exp(w_test_encoded_log_var[:,column])

    use_global_minmax = True
    if use_global_minmax:
        vmin = np.amin([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
        vmax = np.amax([shared_test_encoded_var, x_test_encoded_var, w_test_encoded_var])
        print("global min: ", vmin)
        print("global max: ", vmax)
    else:
        vmin = None
        vmax = None

    # display a 2D plot of the digit classes in the latent space
    ax[0].set_title("input (x,w)")
    x1_plot = ax[0].scatter(encoded_1[:, 0], encoded_1[:, 1], c=shared_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
    ax[1].set_title("input (x)")
    x2_plot = ax[1].scatter(encoded_2[:, 0], encoded_2[:, 1], c=x_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
    ax[2].set_title("input (w)")
    x3_plot = ax[2].scatter(encoded_3[:, 0], encoded_3[:, 1], c=w_test_encoded_var, cmap='rainbow', vmin=vmin, vmax=vmax)
    ymin = np.amin([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
    ymax = np.amax([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()])
    xmin = np.amin([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
    xmax = np.amax([ax[0].get_xlim(), ax[1].get_xlim(), ax[2].get_xlim()])
    ax[0].set_xlim(xmin,xmax)
    ax[1].set_xlim(xmin,xmax)
    ax[2].set_xlim(xmin,xmax)
    ax[0].set_ylim(ymin,ymax)
    ax[1].set_ylim(ymin,ymax)
    ax[2].set_ylim(ymin,ymax)

    asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
    ax[0].set_aspect(asp)
    asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
    ax[1].set_aspect(asp)
    asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
    ax[2].set_aspect(asp)
    if use_global_minmax:
        plt.colorbar(x3_plot, ax=list(ax))
    else:
        plt.colorbar(x1_plot, ax=ax[0])
        plt.colorbar(x2_plot, ax=ax[1])
        plt.colorbar(x3_plot, ax=ax[2])
    plt.show(block=False)


# In[14]:


run_test = False
if run_test:
    #Show some input output results
    decoder = model_obj.get_decoder()


    values = 100
    z_range_hor = np.linspace(0.,1.,num=values,dtype=np.float32)
    z_range_ver = np.linspace(.02,.005,num=values,dtype=np.float32)
    for idx in np.arange(0,values,dtype=np.int32):
        tmp = decoder.predict(np.array([[z_range_hor[idx], z_range_ver[idx]]]), batch_size=batch_size)
        print(str(idx/values) + ": " + str(np.round(tmp[0],2)) + ", " + str(np.round(tmp[1],2)))


    # print(decoder.predict(np.array([[-1, 8]]), batch_size=batch_size))
    # print(decoder.predict(np.array([[6, 0]]), batch_size=batch_size))
    z_test_values = [[0., 0.]]
    decoded_val = decoder.predict(np.array(z_test_values), batch_size=batch_size)
    print("decoded values:")
    print(decoded_val)
    print("z_test_values (true vs. reencode):" + str(z_test_values)  + " vs. " + str(encoder_shared.predict([decoded_val[0], decoded_val[1]], batch_size=batch_size)))
    print("")
    x_test_encdec = np.array([[0.88458996, 0.11993087]])
    w_test_encdec = np.array([[0.09362522, 0.92664626]])

    # xw_encdec = encdec.predict([x_test_encdec, w_test_encdec], batch_size=batch_size)
    xw_encdec = decoder.predict(encoder_shared.predict([x_test_encdec, w_test_encdec], batch_size=batch_size), batch_size=batch_size)
    x_encdec = decoder.predict(encoder_x.predict(x_test_encdec, batch_size=batch_size), batch_size=batch_size)
    w_encdec = decoder.predict(encoder_w.predict(w_test_encdec, batch_size=batch_size), batch_size=batch_size)

    print("xw_encdec, x: ", xw_encdec[0])
    print("xw_encdec, w: ", xw_encdec[1])
    print("")
    print("x_encdec,  x: ", x_encdec[0])
    print("x_encdec,  w: ", x_encdec[1])
    print("")
    print("w_encdec,  x: ", w_encdec[0])
    print("w_encdec,  w: ", w_encdec[1])
    print("")
    print("min shared_test_encoded_var: ", np.min(shared_test_encoded_var))
    print("max shared_test_encoded_var: ", np.max(shared_test_encoded_var))
    print("")
    print("min x_test_encoded_var     : ", np.min(x_test_encoded_var))
    print("max x_test_encoded_var     : ", np.max(x_test_encoded_var))
    print("")
    print("min w_test_encoded_var     : ", np.min(w_test_encoded_var))
    print("max w_test_encoded_var     : ", np.max(w_test_encoded_var))

