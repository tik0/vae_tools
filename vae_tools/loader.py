#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from PIL import Image
import sys
import os
import os.path
from glob import glob
from tensorflow import keras
import random
import requests
import git
import csv

class GoogleDriveDownloader():
    def __init__(self):
        pass

    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"
        if os.path.isfile(destination):
            #print("File already available")
            return
        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        self.save_response_content(response, destination) 
        #print("Done")

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    
def mnist(new_shape = (28, 28, 1), kind='digit', get_single_label = None):
    '''
    digit (str): digit or fashion
    get_single_label (int): lable between 0 .. 9
    '''
    if kind == 'digit':
        (train, train_label), (test, test_label) = keras.datasets.mnist.load_data()
    elif kind == 'fashion':
        (train, train_label), (test, test_label) = keras.datasets.fashion_mnist.load_data()
    else:
        raise
    train = train.astype('float32') / 255.
    train = train.reshape((train.shape[0],) + new_shape)
    test = test.astype('float32') / 255.
    test = test.reshape((test.shape[0],) + new_shape)
    if get_single_label is not None:
        label = get_single_label
        train = train[train_label == label, :]
        test = test[test_label == label, :]
        train_label = train_label[train_label == label]
        test_label = test_label[test_label == label]
    return train, train_label, test, test_label

def camera_lidar(filename, folder_sets, filename_camera, filename_lidar, measurements_per_file, image_shape_old, image_shape_new, lidar_shape_old, lidar_shape_new, lidar_range_max = 5., overwrite = False):
    ''' Loading the camera/lidar set and storing the processed data in a temporary file for faster reloading
    filename        (str): Filename to store or load from
    folder_sets     (str): Folders containing npz files to load from
    filename_camera (str): Filename to the camera files
    filename_lidar  (str): Filename to the camera files
    measurements_per_file (int) : measurements per file which is loaded for proper allocation
    image_shape_old (tuple): Original image size
    image_shape_new (tuple): Target image size for resizing
    lidar_shape_old (int): Original lidar array size
    lidar_shape_new (int): Target lidar array size located around the center
    lidar_range_max (float): Prune measurements grater than this value
    overwrite       (bool): Overwrite the temporary file
    '''
    
    filename_npz = filename + ".npz"
    if not os.path.isfile(filename_npz) or overwrite:
        # Traverse the sets of folders, while every set get one label
        num_folder_sets = len(folder_sets)
        numFolder = 0
        for idx_set in range(0, num_folder_sets):
            numFolder = numFolder + len(glob(folder_sets[idx_set]))
        measurementsPerFile = measurements_per_file
        numMeasurements = measurements_per_file * numFolder
        X = np.zeros(shape=(numMeasurements, np.prod(image_shape_old)), dtype = 'uint8')
        Y = np.zeros(shape=(numMeasurements, np.prod(lidar_shape_old)), dtype = 'float')
        label_idx = np.zeros(shape=(numMeasurements, ), dtype = 'uint8')
        label_str = list()

        # Load the raw data
        label_counter = 0
        idx = 0
        for idx_set in range(0, num_folder_sets):
            folder = glob(folder_sets[idx_set])
            num_folder_per_set = len(folder)
            for idx_folder in range(0, num_folder_per_set):
                # camera data
                tmp = np.load(folder[idx_folder] + filename_camera)
                X[idx*measurementsPerFile:(idx+1)*measurementsPerFile,:] = np.asarray( tmp['x'], dtype = 'uint8').transpose()
                # lidar data
                tmp = np.load(folder[idx_folder] + filename_lidar)
                tmp = np.asarray( tmp['x'], dtype = 'float')
                Y[idx*measurementsPerFile:(idx+1)*measurementsPerFile,:] = np.squeeze(tmp[0,:,:]).transpose()
                # label
                label_idx[idx*measurementsPerFile:(idx+1)*measurementsPerFile] = label_counter
                label_str.append(folder_sets[idx_set])
                idx = idx + 1
            label_counter = label_counter + 1

        # Resize, strip the green/blue channel ( it is alrady scaled to [0, 1] when casting to float)
        X_c = np.zeros(shape=(len(X), np.prod(image_shape_new)))
        for idx in range(0, len(X)):
            img = Image.fromarray(X[idx,:].reshape(image_shape_old))
            img = np.asarray( img.resize((image_shape_new[1], image_shape_new[0]), Image.ANTIALIAS), dtype="uint8" )
            X_c[idx,:] = img[:,:,0:image_shape_new[2]].astype('float32').reshape((np.prod(image_shape_new))) / 255.
        # Flip, strip lidar measurement which are not in the frustum of the camera, and scale to [0, 1]
        X_l = np.fliplr(Y[:,lidar_shape_new-1:2*lidar_shape_new-1]).astype('float32') / lidar_range_max
        X_l[X_l == np.inf] = 0
        np.savez_compressed(filename, X_l=X_l, X_c=X_c, label_idx=label_idx, label_str=label_str)
    else:
        loaded = np.load(filename_npz)
        X_l = loaded["X_l"]
        X_c = loaded["X_c"]
        label_idx = loaded["label_idx"]
        label_str = loaded["label_str"]
    return X_l, X_c, label_idx, label_str


def image_lidar_pair(filename_image, filename_lidar, sample):
    # tmp = np.load("2018-06-02/box_r_0.world.2018-06-02_02-16-31.bag.npz/_amiro1_sync_front_camera_image_raw-X-pixeldata.npz")
    tmp = np.load(filename_image)
    X_c = np.asarray( tmp['x'], dtype = 'uint8').transpose()
    # tmp = np.load("2018-06-02/box_r_0.world.2018-06-02_02-16-31.bag.npz/_amiro1_sync_laser_scan-X-ranges_intensities_angles.npz")
    tmp = np.load(filename_lidar)
    tmp = np.asarray( tmp['x'], dtype = 'float').transpose()
    X_l = np.squeeze(tmp[sample,:,0])
    X_l[X_l == np.inf] = 0
    return X_c[sample,:], X_l

def get_steps_around_hokuyo_center(degree_around_center = 80.):
    # The Hokuyo scans from 2.0944rad@120° to -2.0944rad@-120° with 683 steps (s.t. ~0.36°(360°/1,024 steps))
    # The camera has a hor. FoV of 80deg
    fov_hokuyo = 240
    steps_hokuyo = 683
    factor = fov_hokuyo / degree_around_center
    steps_around_center = np.int(steps_hokuyo / factor)
    angles_around_center = np.arange(start=-(steps_around_center-1)/2, stop=(steps_around_center-1)/2+1, step=1, dtype=float) * degree_around_center / steps_around_center
    return steps_around_center, angles_around_center

def overlay_sets(x_set_input, w_set_input, x_set_label_input, w_set_label_input):
    '''Overlay two data sets by their labels'''
    # reorder
    x_set_label_input_argument = np.argsort(x_set_label_input)
    w_set_label_input_argument = np.argsort(w_set_label_input)
    x_set = x_set_input[x_set_label_input_argument,:]
    w_set = w_set_input[w_set_label_input_argument,:]
    x_set_label_input = x_set_label_input[x_set_label_input_argument]
    w_set_label_input = w_set_label_input[w_set_label_input_argument]
    # cut each class to the same lenght
    x_set_idx = np.array([], dtype=np.int)
    w_set_idx = np.array([], dtype=np.int)
    for idx in np.arange(0,np.min([len(x_set), len(w_set)]), dtype=np.int):
        if x_set_label_input[idx] == w_set_label_input[idx]:
            x_set_idx = np.concatenate((x_set_idx, [idx]))
            w_set_idx = np.concatenate((w_set_idx, [idx]))
    x_set = x_set[x_set_idx,:]
    w_set = w_set[w_set_idx,:]
    x_set_label_input = x_set_label_input[x_set_idx]
    w_set_label_input = w_set_label_input[w_set_idx]
    # Check if the labels overlay
    if np.all(x_set_label_input != w_set_label_input):
        raise Exception("Labels do not overlay with each other")
    label_set = x_set_label_input
    return x_set, w_set, label_set

def mnist_digit_fashion(new_shape = (28, 28, 1), flatten = False):
    ''' Load the mnist digit and fashion data set 
    new_shape   : Desired shape of the mnist images
    flatten     : Flatten the images
    shuffle     : Shuffle the data set
    '''
    
    # Load the data
    digit_train, digit_train_label, digit_test, digit_test_label = mnist(new_shape, 'digit')
    fashion_train, fashion_train_label, fashion_test, fashion_test_label = mnist(new_shape, 'fashion')
    # Overlay train and test set
    x_train, w_train, label_train = overlay_sets(digit_train, fashion_train, digit_train_label, fashion_train_label)
    x_test, w_test, label_test = overlay_sets(digit_test, fashion_test, digit_test_label, fashion_test_label)
    
    if flatten:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        w_train = w_train.reshape((len(w_train), np.prod(w_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        w_test = w_test.reshape((len(w_test), np.prod(w_test.shape[1:])))
    # Shuffle the training set
    shuffle_idx = np.arange(0,len(x_train))
    random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    w_train = w_train[shuffle_idx]
    return x_train, w_train, label_train, x_test, w_test, label_test
        

def emnist(flatten = False, split = 0.99):
    ''' Load the eMnist digit and fashion data set 
    flatten     : Flatten the images
    split       : Percentage of the training set
    '''
    # Download the data
    gdd = GoogleDriveDownloader()
    file_id = '1vHTjzlr6vm5rPk1BQaTnijzGzxmOxbcZ'
    destination = '/tmp/eMNSIT_CVAE_latent_dim-2_beta-4.0_epochs-100.npz'
    gdd.download_file_from_google_drive(file_id, destination)
    
    # Load the data from drive
    loaded = np.load(destination)
    x_digits = loaded['x_digits']
    x_fashion = loaded['x_fashion']
    x_label = loaded['x_label']
    x_set = x_digits
    w_set = x_fashion
    label_set = x_label
    if flatten:
        x_set = x_set.reshape((len(x_set), np.prod(x_set.shape[1:])))
        w_set = w_set.reshape((len(w_set), np.prod(w_set.shape[1:])))

    # Shuffel and define training and test sets
    shuffel_index = np.arange(len(label_set))
    random.shuffle(shuffel_index)
    w_set = w_set[shuffel_index,:]
    x_set = x_set[shuffel_index,:]
    label_set = label_set[shuffel_index]
    train_size = np.int(len(w_set) * split)
    w_train = w_set[:train_size,:]
    w_test = w_set[train_size:,:]
    x_train = x_set[:train_size,:]
    x_test = x_set[train_size:,:]
    label_train = label_set[:train_size]
    label_test = label_set[train_size:]

    return x_train, w_train, label_train, x_test, w_test, label_test



def split(flatten = False, split = 'hor'):
    ''' Get the mnist data set w/ split modalities

    flatten    (bool): Flat the data
    split       (str): Splitting technique. One of ['hor', 'ver', 'quad']

    # Show horizontal split image
    (x_train_a, x_train_b), (x_test_a, x_test_b), y_train, y_test = split(flatten = True, split = 'hor')
    img_rows_2, img_cols, idx = 14, 28, 0
    _, ax = plt.subplots(2,1,sharex=True)
    ax[0].imshow(x_train_a[idx,:].reshape(((img_rows_2, img_cols))))
    ax[1].imshow(x_train_b[idx,:].reshape((img_rows_2, img_cols))))
    plt.show()

    # Show vertical split image
    (x_train_a, x_train_b), (x_test_a, x_test_b), y_train, y_test = split(flatten = True, split = 'ver')
    img_rows, img_cols_2, idx = 28, 14, 0
    _, ax = plt.subplots(1,2,sharey=True)
    ax[0].imshow(x_train_a[idx,:].reshape(((img_rows, img_cols_2))))
    ax[1].imshow(x_train_b[idx,:].reshape(((img_rows, img_cols_2))))
    plt.show()

    # Show quad split image
    (x_train_a, x_train_b, x_train_c, x_train_d), (x_test_a, x_test_b, x_test_c, x_test_d), y_train, y_test = split(flatten = True, split = 'quad')
    img_rows_2, img_cols_2, idx = 14, 14, 0
    _, ax = plt.subplots(2,2,sharey=True, sharex=True)
    ax[0,0].imshow(x_train_a[idx,:].reshape(((img_rows_2, img_cols_2))))
    ax[0,1].imshow(x_train_b[idx,:].reshape(((img_rows_2, img_cols_2))))
    ax[1,0].imshow(x_train_c[idx,:].reshape(((img_rows_2, img_cols_2))))
    ax[1,1].imshow(x_train_d[idx,:].reshape(((img_rows_2, img_cols_2))))
    plt.show()

    returns the data set
    '''

    # Get the MNIST digits
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # input image dimensions
    img_rows, img_cols, img_chns = 28, 28, 1
    img_rows_2, img_cols_2 = int(img_rows / 2), int(img_cols / 2)
    original_dim = img_rows * img_cols * img_chns
    split_dim, num_sets = -1, -1
    num_train_images = len(y_train)
    num_test_images = len(y_test)

    if split == 'hor':
        split_dim, num_sets = int(original_dim / 2), 2
        # Split it horizontally
        x_train_a = x_train[:,:img_rows_2,:]
        x_train_b = x_train[:,img_rows_2:,:]
        x_test_a = x_test[:,:img_rows_2,:]
        x_test_b = x_test[:,img_rows_2:,:]
        _x_train = [x_train_a, x_train_b]
        _x_test = [x_test_a, x_test_b]
    elif split == 'ver':
        split_dim, num_sets = int(original_dim / 2), 2
        # Split it horizontally
        x_train_a = x_train[:,:,:img_cols_2]
        x_train_b = x_train[:,:,img_cols_2:]
        x_test_a = x_test[:,:,:img_cols_2]
        x_test_b = x_test[:,:,img_cols_2:]
        _x_train = [x_train_a, x_train_b]
        _x_test = [x_test_a, x_test_b]
    elif split == 'quad':
        split_dim, num_sets = int(original_dim / 4), 4
        # Split it horizontally
        x_train_a = x_train[:,:img_rows_2,:img_cols_2]
        x_train_b = x_train[:,:img_rows_2,img_cols_2:]
        x_train_c = x_train[:,img_rows_2:,:img_cols_2]
        x_train_d = x_train[:,img_rows_2:,img_cols_2:]
        x_test_a = x_test[:,:img_rows_2,:img_cols_2]
        x_test_b = x_test[:,:img_rows_2,img_cols_2:]
        x_test_c = x_test[:,img_rows_2:,:img_cols_2]
        x_test_d = x_test[:,img_rows_2:,img_cols_2:]
        _x_train = [x_train_a, x_train_b, x_train_c, x_train_d]
        _x_test = [x_test_a, x_test_b, x_test_c, x_test_d]
    else:
        raise ValueError("Not supported")

    if flatten:
        for idx in range(num_sets):
            _x_train[idx] = _x_train[idx].reshape((num_train_images, split_dim))
            _x_test[idx] = _x_test[idx].reshape((num_test_images, split_dim))

    return tuple(_x_train), tuple(_x_test), y_train, y_test


def lidar_camera_set(lidar_degree_around_center = 80., image_target_rows_cols_chns = (64, 64, 2)):
    ''' Load the lidar/camera data set 
    flatten     : Flatten the images
    split       : Percentage of the training set
    '''
    # Load the data
    steps_around_center, angles_around_center = vae_tools.loader.get_steps_around_hokuyo_center(degree_around_center = lidar_degree_around_center)
    image_original_rows_cols_chns= (800,800,3)
    new_shape = image_target_rows_cols_chns
    old_shape = image_original_rows_cols_chns
    # measurements_per_file = 111 # 2018-06-02 dataset
    measurements_per_file = 300 # 2018-06-05 dataset
    # Download the data (TBD)
    gdd = GoogleDriveDownloader()
    file_id = '1_EBi68TBXYbrBV4-9m8gsNJe1-DOnCzg'
    destination = '/tmp/2018-06-05.zip'
    gdd.download_file_from_google_drive(file_id, destination)
    # Process the data
    # folder = glob('2018-06-02/cyl_r*/') + glob('2018-06-02/box_r*/');
    # folder = glob('2018-06-05/cyl_r*/') + glob('2018-06-05/box_r*/') + glob('2018-06-05/cyl_g*/');
    folder = ('../../mVAE/2018-06-05/cyl_r*/', '../../mVAE/2018-06-05/box_r*/', '../../mVAE/2018-06-05/cyl_g*/');
    X_l, X_c, X_set_label_idx, X_set_label_str = vae_tools.loader.camera_lidar("./Xl-80deg_Xc-64-64-2", folder, 
                        "_amiro1_sync_front_camera_image_raw-X-pixeldata.npz", 
                        "_amiro1_sync_laser_scan-X-ranges_intensities_angles.npz", 
                        measurements_per_file, old_shape, new_shape, 683, steps_around_center, 5., overwrite = False)
    return X_l, X_c, X_set_label_idx, X_set_label_str

def didactical_set(normalize_min_max = True, normalize_mean_var = False, noise_amp_x = 0.06, noise_amp_w = 0.1):
    ''' Load the didactical set as discribed in IEEE FUSION2019 Jointly Trained Variational Autoencoder for Multi-Modal Sensor Fusion Fig. 3a 
    normalize_min_max    : 0 to 1 normalization
    normalize_mean_var   : Normalize mean and variance
    noise_amp_x          : Attribute for noise in modality x
    noise_amp_w          : Attribute for noise in modality w
    '''
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
    w_set = w_set + noise_amp_w * np.random.randn(w_set.shape[0], w_set.shape[1])
    x_set = x_set + noise_amp_x * np.random.randn(x_set.shape[0], x_set.shape[1])

    if normalize_min_max:
        x_set[:,0] = x_set[:,0] - np.amin(x_set[:,0])
        x_set[:,1] = x_set[:,1] - np.amin(x_set[:,1])
        x_set[:,0] = x_set[:,0] / np.amax(x_set[:,0])
        x_set[:,1] = x_set[:,1] / np.amax(x_set[:,1])
        w_set[:,0] = w_set[:,0] - np.amin(w_set[:,0])
        w_set[:,1] = w_set[:,1] - np.amin(w_set[:,1])
        w_set[:,0] = w_set[:,0] / np.amax(w_set[:,0])
        w_set[:,1] = w_set[:,1] / np.amax(w_set[:,1])
    if normalize_mean_var:
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

    return x_train, w_train, gt_train, x_test, w_test, gt_test


def rubiks(num_tuples=int(10000), target_size=(30, 40), working_dir='/tmp'):
    ''' Loads the rubiks data set

    :param num_tuples: number of action tuples to create
    :param target_size: target size of the images
    :param working_dir: workfolder for git
    :return: actionvector, viewpoint 1, viewpoint 2, cube states, cube colors
    '''

    repo_name = 'rubiks-dataset'
    repo_url = 'https://github.com/tik0/' + repo_name + '.git'
    try:
        git.Git(working_dir).clone(repo_url)
    except Exception as e:
        print(e)

    data_dir = working_dir + '/' + repo_name + '/asset'
    num_data = 19

    sample_folders = ['bag_front_state_blue',
                      'bag_front_state_green',
                      'bag_front_state_orange',
                      'bag_front_state_red',
                      'bag_front_state_white',
                      'bag_front_state_yellow',
                      'bag_left_state_blue',
                      'bag_left_state_green',
                      'bag_left_state_orange',
                      'bag_left_state_red',
                      'bag_left_state_white',
                      'bag_left_state_yellow',
                      'bag_right_state_blue',
                      'bag_right_state_green',
                      'bag_right_state_orange',
                      'bag_right_state_red',
                      'bag_right_state_white',
                      'bag_right_state_yellow']
    cube_colors = ['red', 'white', 'green', 'orange', 'yellow', 'blue']

    def sample_cube_state(num_samples):
        cube_opposit_sides = [['blue', 'green'], ['white', 'yellow'], ['orange', 'red'], ['green', 'blue'],
                              ['yellow', 'white'], ['red', 'orange']]
        cube_states = {'left': [], 'front': [], 'right': []}
        # sample left color
        lookup_cube_colors = np.random.randint(0, high=len(cube_colors), size=num_samples)
        for idx in lookup_cube_colors:
            # Add the left view color
            cube_states['left'].append(cube_colors[idx])
            # Get the color of the opposit side
            cube_opposit_side_color = [color[1] for color in cube_opposit_sides if color[0] == cube_colors[idx]]
            cube_states['right'].append(cube_opposit_side_color[0])
            # Sample the front view from the remaining colors
            cube_colors_wo_left_right = cube_colors.copy()
            cube_colors_wo_left_right.remove(cube_states['left'][-1])
            cube_colors_wo_left_right.remove(cube_states['right'][-1])
            cube_states['front'].append(cube_colors_wo_left_right[np.random.randint(0, high=4, size=1)[0]])
        return cube_states

    with open(data_dir + "/actions.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        actions = [[x[0], x[1], float(x[2]), float(x[3]), float(x[4])] for x in list(reader)[1:]]
        f.close()

    with open(data_dir + "/colors.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        colors = list(reader)[1:]
        f.close()

    # Read the poses and observations
    def get_iterator(batch_size, data_sub_dir=""):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.0)  # randomly shift images vertically (fraction of total height))

        train_generator = train_datagen.flow_from_directory(data_dir + "/" + data_sub_dir, interpolation='nearest',
                                                            color_mode='rgb', shuffle=False, seed=None,
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            # save_to_dir='img_0_augmented',
                                                            class_mode=None)
        return train_generator

    X_train_set = {}
    for sample_folder in sample_folders:
        X_train_set[sample_folder] = get_iterator(num_data, sample_folder).next()

    # Create the action/perception tuples
    action_train = np.zeros((num_tuples, 3))
    v1_train = np.zeros((num_tuples, target_size[0], target_size[1], 3))
    v2_train = np.zeros((num_tuples, target_size[0], target_size[1], 3))
    cube_states = sample_cube_state(num_samples=num_tuples)

    lookup = np.random.randint(0, high=len(actions), size=num_tuples)
    lookup_cube_state = np.random.randint(0, high=num_data, size=num_tuples)

    for idx_lookup, idx in zip(lookup, range(num_tuples)):
        # Get the action for the random sample
        v1_text = actions[idx_lookup][0]
        v2_text = actions[idx_lookup][1]
        action_train[idx] = np.asarray(actions[idx_lookup][2:])
        # Get the views for the action sample and a given cube state
        random_vp_sample = np.random.randint(0, high=num_data, size=1)
        v1_train[idx, :] = X_train_set['bag_' + v1_text + '_state_' + cube_states[v1_text][idx]][random_vp_sample[0]]
        random_vp_sample = np.random.randint(0, high=num_data, size=1)
        v2_train[idx, :] = X_train_set['bag_' + v2_text + '_state_' + cube_states[v2_text][idx]][random_vp_sample[0]]

    # Normalize actions
    action_train = (action_train + 1.) / 2.
    return action_train, v1_train, v2_train, cube_states, cube_colors
