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