
# coding: utf-8

# In[1]:


'''This script demonstrates how to entangle digit with fashion mnist
'''
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
import keras
sys.path.append(os.path.expanduser('~/notebook'))
from tools import plot_confusion_matrix, plot_model, layers, nb_tools, viz, loader, build_model, sanity, sampling, custom_variational_layer, metrics, build_model

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3
# batch size
batch_size = 128
# latent dim
latent_dim = 2

# Hyperparameters
# Beta
beta = 4
# label set
label = 6
# use fashion mnist or not (digit)
use_fashion = 0
# epochs
epochs = 20

try:
    _beta = np.float(os.environ['BETA'])
    _label = np.int(os.environ['LABEL'])
    _epochs = np.int(os.environ['EPOCHS'])
    _use_fashion = np.int(os.environ['USE_FASHION'])
    beta = _beta
    label = _label
    epochs = _epochs
    use_fashion = _use_fashion
    print("WARNING: Set parameters from environment")
except:
    pass

prefix = "eMNSIT_fashion-" + str(use_fashion) + "_label-" + str(label) + "_beta-" + str(beta) + "_epochs-" + str(epochs)


model_obj = build_model.Vae2dConv()
model_obj.configure(img_rows = img_rows, img_cols = img_cols, img_chns = img_chns,
                    batch_size = batch_size, filters = 64, num_conv = 3,
                    intermediate_dim = 128,
                    latent_dim = latent_dim,
                    beta = beta)

vae = model_obj.get_model()
vae.compile(optimizer='rmsprop', loss=None)
# vae.summary()
# SVG(model_to_dot(vae).create(prog='dot', format='svg'))


# In[2]:


# train the VAE on MNIST digits
if use_fashion==1:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

#print('x_train.shape:', x_train.shape)
#print('min value:', np.amin(x_train))
#print('max value:', np.amax(x_train))
#print('mean:', np.mean(x_train))
#print('variance:', np.var(x_train))

# Refine set
label = 5
x_train = x_train[y_train == label, :]
x_test = x_test[y_test == label, :]
y_train = y_train[y_train == label]
y_test = y_test[y_test == label]


# In[3]:


# train
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
model_obj.store_model(prefix + "_model-decoder", model_obj.get_decoder(), overwrite = True)


# In[4]:


# build a model to project inputs on the latent space
encoder = model_obj.get_encoder_mean()

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
# plt.colorbar()
save_str = prefix + "_z" + ".png"
plt.savefig(save_str, dpi=300)
print("save to: ", save_str)
plt.show(block=False)

# build a digit generator that can sample from the learned distribution
generator = model_obj.get_decoder()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
save_str = prefix + "_x" + ".png"
plt.savefig(save_str, dpi=300)
print("save to: ", save_str)
plt.show(block=False)


# In[5]:


#from scipy.stats import norm
#print(norm.ppf(np.linspace(0.05, 0.95, 3)))

