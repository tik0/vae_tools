#!/usr/bin/python

import sys
import vae_tools  # Always import first to define if keras or tf.kreas should be used
import vae_tools.sanity
import vae_tools.viz
import vae_tools.callbacks
from vae_tools.mmvae import MmVae, ReconstructionLoss

vae_tools.sanity.check()
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Layer
from keras.datasets import mnist
import numpy as np
import numpy
from scipy.stats import norm
# Set the seed for reproducible results
import vae_tools.sampling

vae_tools.sampling.set_seed(0)
# resize the notebook if desired
import vae_tools.nb_tools

vae_tools.nb_tools.notebook_resize()
# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def loader(filename):
    try:
        data = np.load(filename, encoding = 'bytes')
    except:
        print("File not found.")
        exit(0)

    range_data_t = data[:, 0]
    camera_data_t = data[:, 1]
    reflectivity_data_t = data[:, 2]

    print("loaded file.")
    print("range_data", range_data_t.shape)
    print("camera_data", camera_data_t.shape)
    print("reflectivity_data", reflectivity_data_t.shape)
    return range_data_t, camera_data_t, reflectivity_data_t

def get_one_image_and_show_result():
    image_data = _data.tobytes()  # byte values of the image
    image = numpy.asarray(Image.frombytes('RGB', resolution, image_data))
    # Show
    plt.imshow(image)
    plt.show()
    print("Cut the image to ", image[220:260, :, :].shape)
    plt.imshow(image[220:260, :, :])
    plt.show()

def preprocess_images():
    # Process the images
    print("preprocess image.")
    print(len(camera_data), resolution[1], resolution[0], 3)
    images = np.zeros(shape = (len(camera_data), resolution[1], resolution[0], 3))
    images_windowed = np.zeros(shape = (len(camera_data), window_height[1] - window_height[0], resolution[0], 3))
    images_windowed_flat = np.zeros(shape = (len(camera_data), (window_height[1] - window_height[0]) * resolution[0] * 3))
    for idx in range(len(camera_data)):
        # print(camera_data)
        # print(np.frombuffer(camera_data[idx,1], dtype=np.uint8).reshape(resolution[1],resolution[0],3).shape)
        images[idx, :, :, :] = numpy.asarray(Image.frombytes('RGB', resolution, camera_data[idx, 1].tobytes())) / 255.0
        # images[idx,:,:,:] = np.frombuffer(camera_data[idx,1], dtype=np.uint8).reshape(resolution[0],resolution[1],3)/255.0
        images_windowed[idx, :, :, :] = images[idx, window_height[0]:window_height[1], :, :]
        images_windowed_flat[idx, :] = images_windowed[idx].flatten()
        # images[idx]

    print("preprocess image done.")
    return images_windowed_flat

def preprocess_lidar():
    print("preprocess lidar.")
    ranges = np.zeros(shape = (len(range_data), len(range_data[0, 1])))
    reflections = np.zeros(shape = (len(reflectivity_data), len(reflectivity_data[0, 1])))
    for idx in range(len(range_data)):
        ranges[idx] = range_data[idx, 1]
        reflections[idx] = reflectivity_data[idx, 1]
    # normalization
    ranges[ranges == np.inf] = 0
    # print(np.min(ranges))
    # print(np.max(ranges))
    print(np.min(reflections))
    print(np.max(reflections))
    ranges = ranges - np.min(ranges)
    ranges = ranges / np.max(ranges)
    reflections = reflections / np.max([np.max(reflections), -np.min(reflections)])
    reflections = reflections - np.min(reflections)
    reflections = reflections / np.max(reflections)
    # print(np.min(reflections))
    # print(np.max(reflections))

    if debug:
        img_idx = 8
        plt.plot(ranges[img_idx])
        plt.show()
        plt.plot(reflections[img_idx])
        plt.show()

    print("preprocess lidar done.")

    return ranges, reflections

if __name__ == '__main__':

    print("sys.version:", sys.version)

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', default = "", help = "filename")
    parser.add_argument('-d', '--debug', default = 0, help = "debug")
    args = parser.parse_args()

    filename = args.file
    debug = args.debug

    # load data
    range_data, camera_data, reflectivity_data = loader(filename)

    # Get one image and show the result
    _data = camera_data[17, 1]
    resolution = (640, 480)  # (800,800)
    if debug:
        get_one_image_and_show_result()
    window_height = [240, 241]
    # Process the images
    X_image = preprocess_images()


    # Process th ranges and reflections
    W_ranges, V_reflection = preprocess_lidar()


    # input image dimensions
    print("create encoder/decoder")
    img_rows, img_cols, img_chns = window_height[1] - window_height[0], resolution[0], 3
    batch_size = 128
    img_original_dim = img_rows * img_cols * img_chns
    lidar_original_dim = len(range_data[0, 1])
    intermediate_dim = 256
    epochs = 100
    z_dim = 2
    beta = 1.

    encoder = [[
        Input(shape = (img_original_dim,)),
        Dense(intermediate_dim, activation = 'relu'),
        Dense(int(intermediate_dim / 2), activation = 'relu')
    ],
        [
            Input(shape = (lidar_original_dim,)),
            Dense(intermediate_dim, activation = 'relu'),
            Dense(int(intermediate_dim / 2), activation = 'relu')
        ],
        [
            Input(shape = (lidar_original_dim,)),
            Dense(intermediate_dim, activation = 'relu'),
            Dense(int(intermediate_dim / 2), activation = 'relu')
        ]]

    decoder = [[
        Dense(int(intermediate_dim / 2), activation = 'relu'),
        Dense(intermediate_dim, activation = 'relu'),
        Dense(img_original_dim, activation = 'sigmoid')
    ],
        [
            Dense(int(intermediate_dim / 2), activation = 'relu'),
            Dense(intermediate_dim, activation = 'relu'),
            Dense(lidar_original_dim, activation = 'sigmoid')
        ],
        [
            Dense(int(intermediate_dim / 2), activation = 'relu'),
            Dense(intermediate_dim, activation = 'relu'),
            Dense(lidar_original_dim, activation = 'sigmoid')
        ]]

    # encoder = [[
    #     Input(shape=(img_original_dim,)),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(int(intermediate_dim/2), activation='relu')
    # ],
    # [
    #     Input(shape=(lidar_original_dim,)),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(int(intermediate_dim/2), activation='relu')
    # ]]

    # decoder = [[
    #     Dense(int(intermediate_dim/2), activation='relu'),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(img_original_dim, activation='sigmoid')
    # ],
    # [
    #     Dense(int(intermediate_dim/2), activation='relu'),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(lidar_original_dim, activation='sigmoid')
    # ]]

    # encoder = [[
    #     Input(shape=(lidar_original_dim,)),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(int(intermediate_dim/2), activation='relu')
    # ]]

    # decoder = [[
    #     Dense(int(intermediate_dim/2), activation='relu'),
    #     Dense(intermediate_dim, activation='relu'),
    #     Dense(lidar_original_dim, activation='sigmoid')
    # ]]

    print("create vae_obj.")
    vae_obj = MmVae(z_dim, encoder, decoder, [img_original_dim, lidar_original_dim, lidar_original_dim], beta = 0.01, beta_is_normalized = True, reconstruction_loss_metrics = [ReconstructionLoss.MSE], name = 'MmVae')
    # vae_obj = MmVae(z_dim, encoder, decoder, [lidar_original_dim], beta = beta, reconstruction_loss_metrics = [ReconstructionLoss.MSE], name='MmVae')

    vae = vae_obj.get_model()
    vae.compile(optimizer = 'adam', loss = None)
    # vae_tools.viz.plot_model(vae, file = 'myVAE', print_svg = False, verbose = True)

    x_train = [X_image, W_ranges, V_reflection]
    # x_train = [images_windowed_flat, ranges]
    # x_train = [images_windowed_flat]
    # x_train = [reflections]

    # Store the losses to a history object for plotting
    print("train model.")
    losses_cb = vae_tools.callbacks.Losses(data = x_train)
    # Train
    vae.fit(x_train,
            shuffle = True,
            epochs = epochs,
            batch_size = batch_size,
            verbose = 0,
            callbacks = [losses_cb])
    print("train model done.")

    # Show the losses
    import matplotlib
    import matplotlib.pyplot as plt

    print("plot losses.")
    num_losses = len(losses_cb.history.values())
    f, axs = plt.subplots(num_losses, 1, sharex = True, figsize = [10, 50], dpi = 96)
    axs[0].set_title("Losses")
    for idx in range(num_losses):
        axs[idx].plot(list(losses_cb.history.values())[idx])
        axs[idx].set_xlabel([list(losses_cb.history.keys())[idx]])
    plt.show()

    # Store the models
    print("store models.")
    vae_obj.store_model_powerset(prefix = 'models/enc_mean_xwv_', model_inputs = vae_obj.encoder_inputs,
                                 get_model_callback = vae_obj.get_encoder_mean)
    vae_obj.store_model_powerset(prefix = 'models/enc_logvar_xwv_', model_inputs = vae_obj.encoder_inputs,
                                 get_model_callback = vae_obj.get_encoder_logvar)
    vae_obj.store_model(name = 'models/decoder_xwv', model = vae_obj.get_decoder(), overwrite = True)
    print("store models done.")

    # Load the models
    # encoder_mean_models, bitmask = vae_obj.load_model_powerset(prefix = 'models/enc_mean_xwv_', num_elements = len(vae_obj.encoder_inputs))
    # encoder_logvar_models, bitmask = vae_obj.load_model_powerset(prefix = 'models/enc_logvar_xwv_', num_elements = len(vae_obj.encoder_inputs))
    # decoder_model = vae_obj.load_model(name = 'models/decoder_xwv')
