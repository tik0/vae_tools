#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

# Show channels of unit8 RGB image
def image_channels(C, as_float = True):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    C = resize(C, (64, 64))
    if as_float:
        C = C.astype('float32') / 255.
    ax1.imshow(C[:,:,0])
    ax1.set_title('R')
    ax2.imshow(C[:,:,1])
    ax2.set_title('G')
    ax3.imshow(C[:,:,2])
    ax3.set_title('B')
    plt.show()


# Show image/lidar pais
def image_lidar_pair(C, L_dist, L_angle):
    fig, (ax, ax2) = plt.subplots(ncols=2)
    ax.imshow(C)
    ax.set_title('Image')
    ax2.plot(L_angle, L_dist, 'b.-')
    ax2.set_title('LiDAR')
    ax2.set_xlabel('angle (°)')
    ax2.set_ylabel('distance (m)')
    # set the aspect ratio
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    plt.show()

def get_image_dec_enc_samples(grid_x, grid_y, decoder, encoder_mean, encoder_log_var, image_size, Dz):
    # grid_x: latent grid vector in x direction
    # grid_y: latent grid vector in y direction
    # decoder: the decoder network
    # encoder_mean: the encoder network wich samples the mean
    # encoder_log_var: the encoder network wich samples the log variance
    # image_size: The image dimensions from the deoder
    # Dz: Latent dimension size
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    
    image_width = image_size[0]
    image_height = image_size[1]
    image_depth = image_size[2]
    if image_depth == 1:
        figure = np.zeros((image_height * ny, image_width * nx, 1), dtype='uint8')
    else:
        figure = np.zeros((image_height * ny, image_width * nx, 3), dtype='uint8')
    z_reencoded_mean = np.zeros(shape=(ny,nx,Dz))
    z_reencoded_std = np.zeros(shape=(ny,nx))
    # Sample  from the latent space
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            z_reencoded_mean[i,j,:] = encoder_mean.predict(x_decoded)
            z_reencoded_std[i,j] = np.sum(np.exp(encoder_log_var.predict(x_decoded) / 2.0))
            x_decoded_reshaped = x_decoded[0].reshape(image_width, image_height, image_depth)
            if image_depth == 1 or image_depth == 3:
                digit = 255. - x_decoded_reshaped * 255.
            elif image_depth == 2:
                digit = 255. - np.concatenate((x_decoded_reshaped, np.ones((image_width,image_height,1))), axis=2) * 255.
            else:
                assert False, "Cannot handle image depth!"
            i_corrected = nx-1-i
            figure[i_corrected * image_height: (i_corrected + 1) * image_height,
                j * image_width: (j + 1) * image_width] = digit.astype('uint8')
    return figure, z_reencoded_mean, z_reencoded_std

def random_images_from_set(X_set, image_rows_cols_chns, n):
    # n(int): how many samples are shown (it is actually nxn)
    sampleArray = np.zeros((n, n), dtype='int')
    figure = np.zeros((image_rows_cols_chns[0] * n, image_rows_cols_chns[1] * n, 3), dtype='uint8')
    for i in np.arange(0, n):
        for j in np.arange(0, n):
            sampleArray[i,j] = np.random.randint(0, high=len(X_set))
            X = X_set[sampleArray[i,j],:,:,0:2]
            if image_rows_cols_chns[2] == 1 or image_rows_cols_chns[2] == 3:
                digit = 255. - X * 255.
            else:
                digit = 255. - np.concatenate((X, np.ones((image_rows_cols_chns[0],image_rows_cols_chns[1],1))), axis=2) * 255.
            figure[i * image_rows_cols_chns[0]: (i + 1) * image_rows_cols_chns[0],
                   j * image_rows_cols_chns[1]: (j + 1) * image_rows_cols_chns[1]] = digit.astype('uint8')

    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    # plt.imshow(figure, cmap='Greys_r')
    plt.show()
    return sampleArray

