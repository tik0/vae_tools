#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import numpy
import numpy.matlib
from scipy.stats import norm
from skimage.transform import resize
import os
import PIL
import warnings
from PIL import Image
import matplotlib
from matplotlib import offsetbox
import keras                                                                             
from random import randint
from IPython.display import SVG
from IPython.display import display
try:
    from keras.utils.vis_utils import model_to_dot
except:
    from tensorflow.python.keras.utils.vis_utils import model_to_dot
import vae_tools.mmvae, vae_tools.metrics

def plot_model(model, file = None, folder = 'tmp', is_notebook = True, print_svg = False, verbose = True):
    if file == None:
        rand = str(randint(0, 65000))
        filename_png = folder + '/model_' + rand + '.png'
        filename_svg = folder + '/model_' + rand + '.svg'
    else:
        filename_png = folder + '/' + file + '.png'
        filename_svg = folder + '/' + file + '.svg'
    if not os.path.exists(folder):
        try:
            os.makedirs('tmp')
        except OSError:
            print('Error creating temporary directory')
            return
    if verbose:
        print('Store model to filename: ' + filename_png + ' and ' + filename_svg)
        model.summary()
    keras.utils.plot_model(model, to_file=filename_png, show_shapes=True)
    img = Image.open(filename_png)
    svg = model_to_dot(model).create(prog='dot', format='svg')
    f = open(filename_svg, 'wb')
    f.write(svg)
    f.close()
    if is_notebook:
        if print_svg:
            display(SVG(svg))
        else:
            display(img)
    else:
        if print_svg:
            SVG(svg)
        else:
            img.show()


# Show channels of unit8 RGB image
def image_channels(C, as_float = True, figsize=(5,5), dpi=96):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize, dpi=dpi)
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
def image_lidar_pair(C, L_dist, L_angle, figsize=(5,5), dpi=96):
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)
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

def get_image_dec_enc_samples_2(grid_x, grid_y, encoder_mean, encoder_log_var, decoder, Dz, image_size):
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

def get_image_dec_enc_samples(grid_x, grid_y, model_obj, image_size):
    # grid_x: latent grid vector in x direction
    # grid_y: latent grid vector in y direction
    # image_size: The image dimensions from the decoder
    warnings.warn("deprecated, there might be a generic function for calculating the latent statistics", DeprecationWarning)

    try:
        encoder_mean = model_obj.get_encoder_mean([model_obj.encoder[0][0]])
        encoder_log_var = model_obj.get_encoder_logvar([model_obj.encoder[0][0]])
    except: # Backward compatibility
        encoder_mean = model_obj.get_encoder_mean()
        encoder_log_var = model_obj.get_encoder_logvar()
    decoder = model_obj.get_decoder()
    try:
        Dz = model_obj.z_dim
    except: # Backward compatibility
        Dz = model_obj.latent_dim
    return get_image_dec_enc_samples_2(grid_x, grid_y, encoder_mean, encoder_log_var, decoder, Dz, image_size)

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

def scatter_encoder_2(X, label, grid_x, grid_y, encoder_mean, batch_size = 128, figsize=(7.5, 7.5), dpi = 96):
    ''' Plot the latent space with z_dim=2 asuming an uni-modal VAE'''
    
    warnings.warn("deprecated, use plot_embedding", DeprecationWarning)
    
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    Z = encoder_mean.predict(X, batch_size=batch_size)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)
    ax1.scatter(Z[:, 0], Z[:, 1], c=label)
    ax1.set_aspect("equal")
    ax2.scatter(Z[:, 0], Z[:, 1])
    ax2.scatter(np.flipud(np.matlib.repmat(grid_x,nx,1)), np.flipud(np.matlib.repmat(grid_y,ny,1).transpose()), c='r', marker='.')
    ax2.set_aspect("equal")
    plt.show()

def scatter_encoder(X, label, grid_x, grid_y, model_obj, batch_size = 128, figsize=(7.5, 7.5), dpi = 96):
    ''' Plot the latent space with z_dim=2 asuming an uni-modal VAE'''
    
    warnings.warn("deprecated, use plot_embedding", DeprecationWarning)
    
    try:
        encoder_mean = model_obj.get_encoder_mean([model_obj.encoder[0][0]])
    except: # Backward compatibility
        encoder_mean = model_obj.get_encoder_mean([model_obj.encoder[0][0]])
    scatter_encoder_2(X, label, grid_x, grid_y, encoder_mean, batch_size = batch_size, figsize=figsize, dpi = dpi)

def lidar_in_out_2(X_set, num_subplots, encoder_mean, decoder, title = "In- (blue) vs Output (red)"):
    ''' Plot the lidar inputs vs. outputs asuming an uni-modal VAE '''
    f, axs = plt.subplots(1, num_subplots, figsize=(20,10), sharex=True, sharey=True)
 
    for i in np.arange(num_subplots):
        ax = axs[i]
        try: # try if it is data for convolutional network
            X = X_set[i,:,:]
        except:
            X = X_set[i,:]
        ax.plot(X, 'b')
        x_encoded = encoder_mean.predict(np.expand_dims(X, axis=0))
        x_decoded = decoder.predict(x_encoded)
        ax.plot(np.squeeze(x_decoded), 'r')
        ax.set_ylim([0.0,1.0])
        # plt.axis('off')
    f.subplots_adjust(wspace=0)
    f.suptitle(title)
    f.show()
    
def lidar_in_out(X_set, num_subplots, model_obj, title = "In- (blue) vs Output (red)"):
    ''' Plot the lidar inputs vs. outputs asuming an uni-modal VAE '''
    f, axs = plt.subplots(1, num_subplots, figsize=(20,10), sharex=True, sharey=True)
    decoder = model_obj.get_decoder()
    try:
        encoder_mean = model_obj.get_encoder_mean([model_obj.encoder[0][0]])
    except: # Backward compatibility
        encoder_mean = model_obj.get_encoder_mean()
    lidar_in_out_2(X_set, num_subplots, encoder_mean, decoder, title)    

def get_lidar_dec_enc_samples_2(grid_x, grid_y, encoder_mean, encoder_log_var, decoder, Dz, stride = 5, plot_mean = False):
    nx = grid_x.shape[0]
    ny = grid_y.shape[0]
    figures = []
    z_reencoded_mean = np.zeros(shape=(ny,nx,Dz))
    z_reencoded_std = np.zeros(shape=(ny,nx))
    # Sample  from the latent space and show the means
    figure = plt.figure()
    for y_idx, y_val in enumerate(grid_y):
        figures.append([])
        for x_idx, x_val in enumerate(grid_x):
            z_sample = np.array([[x_val, y_val]])
            x_decoded = decoder.predict(z_sample)
            z_reencoded_mean[y_idx,x_idx,:] = encoder_mean.predict(x_decoded)
            z_reencoded_std[y_idx,x_idx] = np.sum(np.exp(encoder_log_var.predict(x_decoded) / 2.0))
            figures[-1].append(x_decoded)
            if np.mod(x_idx,stride) == 0 and np.mod(y_idx,stride) == 0:
                idx = 1 + x_idx/stride + nx/stride * (ny/stride-1-y_idx/stride)
                figure.add_subplot(ny/stride, nx/stride, int(idx))
                plt.plot(x_decoded[0], 'b')
                plt.gca().set_ylim([.0,1.0])
                plt.axis('off')
    if plot_mean:
        plt.show()
    return figures, z_reencoded_mean, z_reencoded_std

def get_lidar_dec_enc_samples(grid_x, grid_y, model_obj, stride = 5, plot_mean = False):
    # grid_x: latent grid vector in x direction
    # grid_y: latent grid vector in y direction
    # model_obj: VAE network
    # stride(int): Stepsize, when to draw an decoding
    # plot_mean(bool): Plot the encoding every stride's sample
    
    warnings.warn("deprecated, there might be a generic function for calculating the latent statistics", DeprecationWarning)
    
    try:
        encoder_mean = model_obj.get_encoder_mean([model_obj.encoder[0][0]])
        encoder_log_var = model_obj.get_encoder_logvar([model_obj.encoder[0][0]])
    except: # Backward compatibility
        encoder_mean = model_obj.get_encoder_mean()
        encoder_log_var = model_obj.get_encoder_logvar()
    decoder = model_obj.get_decoder()
    try:
        Dz = model_obj.z_dim
    except: # Backward compatibility
        Dz = model_obj.latent_dim
    return get_lidar_dec_enc_samples_2(grid_x, grid_y, encoder_mean, encoder_log_var, decoder, Dz, stride = stride, plot_mean = plot_mean)

def get_xy_min_max(ax):
    if type(ax) is not np.ndarray:
        ax = np.asarray([ax])
    _ax = ax.flatten('C')
    num_axis = len(_ax)
    ylim = np.zeros((num_axis,2))
    xlim = np.zeros((num_axis,2))
    for axis_idx in np.arange(num_axis):
        ylim[axis_idx,:] = _ax[axis_idx].get_ylim()
        xlim[axis_idx,:] = _ax[axis_idx].get_xlim()
    return np.amin(ylim), np.amax(ylim), np.amin(xlim), np.amax(xlim)

def set_xy_equal_lim(ax, ylim = None, xlim = None):
    if type(ax) is not np.ndarray:
        ax = np.asarray([ax])
    _ax = ax.flatten('C')
    ymin, ymax, xmin, xmax = get_xy_min_max(ax)
    min, max = np.minimum(ymin, xmin), np.maximum(ymax, xmax) 
    if ylim is None:
        ylim = (min, max) 
    if xlim is None:
        xlim = (min, max) 
    for axis in _ax:
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
    return min, max

def set_xy_aspect(ax):
    _ax = ax.flatten('C')
    for axis in _ax:
        asp = np.diff(axis.get_xlim())[0] / np.diff(axis.get_ylim())[0]
        axis.set_aspect(asp)

def remove_xy_ticks(ax):
    _ax = ax.flatten('C')
    # remove the x and y ticks
    for axis in _ax:
        axis.set_xticks([])
        axis.set_yticks([])
        
def image_resize(image, basewidth = 300, method = PIL.Image.ANTIALIAS):
    ''' Resize a image from https://stackoverflow.com/a/451580/2084944
    image:     The image in a 2D numpy array
    basewidth: Desired image width
    method:    Method for resizing

    Returns the resized image as 2D numpy array
    '''

    img = PIL.Image.fromarray(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    return numpy.asarray(img.resize((basewidth,hsize), method))

def plot_embedding(embeddings, labels, images = None, image_distance_min = float(8e-3),
                   image_width = 16, colormap="tab10", figsize=None, dpi=None, title=None, show_ticks = False):
    ''' Plots the two dimensional embedding (you need to call plt.show() afterwards)
    embedding:          2D The embedding vector (num_samples, features)
    labels:             Ground truth labels as float or integers (num_samples, labels)
    images:             Corresponding highlevel visualization as grayscale
                        image (num_samples, rows, cols)
    image_distance_min: don't show points that are closer than this value
    image_width:        Image width in the plot
    colormap:           Colormap for the scatterplot
    figsize:            The plot's size
    dpi:                The plot's DPI
    title:              The plot title
    show_ticks:         Show the ticks
    
    returns figure object
    '''
    X = embeddings
    y = labels
    num_samples = embeddings.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    if images is not None: # FIXME: Show the images even with the correct embedding !!!
        X = (X - x_min) / (x_max - x_min)     
    figure = plt.figure(figsize = figsize, dpi = dpi)
    ax = plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], c=y/np.max(y), cmap=colormap)
    if hasattr(offsetbox, 'AnnotationBbox') and not np.any(images == None):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(num_samples):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < image_distance_min:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image_resize(images[i], image_width), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    if not show_ticks:
        plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    return figure, ax

def plot_losses(losses, plot_elbo = True, figsize=[10,5], dpi=96):
    ''' Plot all losses
    losses     (obj): Object of the vae_tools.callbacks.Losses class
    plot_elbo (bool): Sums all losses and shows them as ELBO
    figsize   (list): The figure size
    dpi        (int): The plot's DPI
    
    return the figure handle
    '''
    num_plots = len(list(losses.history.values())) + int(plot_elbo)
    f, axs = plt.subplots(num_plots, 1, sharex=True, figsize=[10,5], dpi=96)
    for idx in range(len(axs)-int(plot_elbo)):
        axs[idx].plot(list(losses.history.values())[idx])
        axs[idx].set_xlabel(list(losses.history.keys())[idx])
    if plot_elbo:
        axs[-1].plot([sum(values) for values in zip(*list(losses.history.values()))])
        axs[-1].set_xlabel("ELBO")
    return f
    
def get_latent_space_statistics(decoder, encoder_mean, encoder_logvar, encoder_decoder = None, reconstruction_loss = vae_tools.mmvae.ReconstructionLoss.MSE, steps = 100, grid_min = norm.ppf(0.001), grid_max = norm.ppf(0.999), z_dim = int(2), alpha = 1., beta = 1.):
    ''' Calculates latent space specific statistics
    decoder             (keras.models.Model): VAE decoder model
    encoder_mean        (keras.models.Model): VAE encoder model with mean value output for Gaussian sampling
    encoder_logvar      (keras.models.Model): VAE encoder model with log(var) value output for Gaussian sampling
    encoder_decoder     (keras.models.Model): Optional combined encoder/decoder model with active sampling in the latent layer
    reconstruction_loss (vae_tools.mmvae.ReconstructionLoss): The reconstruction loss type
    steps                              (int): Steps for the resolution of the latent space
    grid_min                         (float): norm.ppf(norm.cdf(grid_min)) is the minimum value for calculating the latent space statistics
    grid_max                         (float): norm.ppf(norm.cdf(grid_max)) is the maximum value for calculating the latent space statistics
    z_dim                              (int): Number of dimension in the latent space
    alpha                            (float): Weighting of the reconstruction error for calculating the ELBO
    beta                             (float): Weighting of the KL divergence for calculating the ELBO
    
    returns statistics (stat_*) about variance, KL divergence, ELBO, and reconstruction error for the latent locations grid_x and grid_y
    '''
    if z_dim is not int(2):
        raise Exception("z_dim != 2 is not supported yet")
    n = steps
    #grid_x = norm.ppf(np.linspace(0.001, 0.999, n))
    #grid_y = norm.ppf(np.linspace(0.001, 0.999, n))
    grid_x = np.linspace(norm.ppf(norm.cdf(grid_min)), norm.ppf(norm.cdf(grid_max)), n)
    grid_y = np.linspace(norm.ppf(norm.cdf(grid_min)), norm.ppf(norm.cdf(grid_max)), n)
    stat_var = np.zeros(shape=(n,n))
    stat_kld = np.zeros(shape=(n,n))
    stat_elbo = np.zeros(shape=(n,n))
    z_inputs = np.zeros(shape=(n,n,2))
    z_reencodings = np.zeros(shape=(n,n,2))
    stat_reconstruction = np.zeros(shape=(n,n))
    for y_idx, y_val in enumerate(grid_y):
        for x_idx, x_val in enumerate(grid_x):
            z_input = np.array([[x_val, y_val]]) # input for the decoder
            z_inputs[y_idx,x_idx] = z_input
            # print(x_val)
            # print(z_input)
            x_decoded = decoder.predict(z_input)
            z_logvar_reencoded = encoder_logvar.predict(x_decoded)
            stat_var[y_idx,x_idx] = np.sum(np.exp(z_logvar_reencoded / 2.0))
            z_reencoded = encoder_mean.predict(x_decoded)
            z_reencodings[y_idx,x_idx] = z_reencoded
            if encoder_decoder is None: # generate decodings w/o sampling layer
                x_redecoded = decoder.predict(z_reencoded)
            else: # generate decodings w/ sampling layer
                x_redecoded = encoder_decoder.predict(x_decoded)
            if reconstruction_loss == vae_tools.mmvae.ReconstructionLoss.BCE:
                reconstruction_error = np.nansum(np.array([x_decoded * np.log(x_redecoded), (1-x_decoded) * np.log( 1 - x_redecoded)]))
            elif reconstruction_loss == vae_tools.mmvae.ReconstructionLoss.MSE:
                reconstruction_error = np.nansum(np.sqrt((x_decoded - x_redecoded)**2))
            else:
                raise Exception("Reconstruction loss not supported!")
            #_y_idx = n-1-y_idx
            _y_idx = y_idx
            _x_idx = x_idx
            #if _x_idx > 90 and _y_idx < 90:
            #    stat_elbo[_y_idx,_x_idx] = 100
            #    stat_kld[_y_idx,_x_idx] = 20
            stat_elbo[_y_idx,_x_idx] = alpha * reconstruction_error
            stat_reconstruction[_y_idx,_x_idx] = reconstruction_error
            for idx in np.arange(0, z_dim):
                #kl_divergence = vae_tools.metrics.kl_loss_n(z_input[0,idx], z_logvar_reencoded[0,idx])
                kl_divergence = vae_tools.metrics.kl_loss_n(z_reencoded[0,idx], z_logvar_reencoded[0,idx])
                stat_kld[_y_idx,_x_idx] = stat_kld[_y_idx,_x_idx] + kl_divergence
                stat_elbo[_y_idx,_x_idx] = stat_elbo[_y_idx,_x_idx] + beta * kl_divergence
    return stat_var, stat_kld, stat_elbo, stat_reconstruction, grid_x, grid_y, z_inputs, z_reencodings



def plot_latent_statistics(X, Y, stat_var, stat_kld, stat_reconstruction, stat_elbo, figsize=(10,10), dpi=96, use_subplots = True):
    if use_subplots:
        f, axs = plt.subplots( 2, 2, figsize=figsize, dpi=dpi, sharex = True, sharey = True)
        f = np.array([[f, f],[f, f]])
    else:
        f1, ax1 = plt.subplots( 1, 1, figsize=figsize, dpi=dpi, sharex = False, sharey = False)
        f2, ax2 = plt.subplots( 1, 1, figsize=figsize, dpi=dpi, sharex = False, sharey = False)
        f3, ax3 = plt.subplots( 1, 1, figsize=figsize, dpi=dpi, sharex = False, sharey = False)
        f4, ax4 = plt.subplots( 1, 1, figsize=figsize, dpi=dpi, sharex = False, sharey = False)
        f = np.array([[f1, f2],[f3, f4]])
        axs = np.array([[ax1, ax2],[ax3, ax4]])

    def _plot_latent_statistics(idy, idx, values, vmin, vmax, title):
        ax = axs[idy,idx]
        c = ax.pcolor(X, Y, values, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        #axs[idy,idx].axis([X.min(), X.max(), Y.min(), Y.max()])
        #divider = make_axes_locatable(ax)
        #f[idy,idx].colorbar(c, ax=ax, cax=divider.append_axes("right", size="5%", pad=0.05))
        f[idy,idx].colorbar(c, ax=ax)
        #axs[idy,idx].axis("equal") # not allowed when sharex = True and sharey = True
        ax.set_aspect("equal")

    _plot_latent_statistics(idy = 0, idx = 0, values = stat_var, vmin = stat_var.min(), vmax = stat_var.max(), title = "variance(z)")
    _plot_latent_statistics(idy = 0, idx = 1, values = stat_kld, vmin = stat_kld.min(), vmax = stat_kld.max(), title = "KL-D")
    _plot_latent_statistics(idy = 1, idx = 0, values = stat_reconstruction, vmin = np.min(stat_reconstruction[stat_reconstruction != -np.inf]), vmax = np.max(stat_reconstruction[stat_reconstruction != np.inf]), title = "Reconstruction Error" )
    _plot_latent_statistics(idy = 1, idx = 1, values = stat_elbo, vmin = np.min(stat_elbo[stat_elbo != -np.inf]), vmax = np.max(stat_elbo[stat_elbo != np.inf]), title = "ELBO (KL-D + Reconstruction Loss)")
    return f, axs

