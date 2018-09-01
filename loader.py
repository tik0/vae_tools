#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import sys
import os

def camera_lidar(filename, folder, filename_camera, filename_lidar, measurements_per_file, image_shape_old, image_shape_new, lidar_shape_old, lidar_shape_new, lidar_range_max = 5., overwrite = False):
    # filename(str): Filename to store or load from
    # folder(str): Folders containing npz files to load from
    filename_npz = filename + ".npz"
    if not os.path.isfile(filename_npz) or overwrite:
        numFolder = len(folder)
        numMeasurements = measurements_per_file * numFolder
        X = np.zeros(shape=(numMeasurements, np.prod(image_shape_old)), dtype = 'uint8')
        Y = np.zeros(shape=(numMeasurements, np.prod(lidar_shape_old)), dtype = 'float')

        for idx in range(0, numFolder):
            tmp = np.load(folder[idx] + filename_camera)
            X[idx*measurementsPerFile:(idx+1)*measurementsPerFile,:] = np.asarray( tmp['x'], dtype = 'uint8').transpose()
            tmp = np.load(folder[idx] + filename_lidar)
            tmp = np.asarray( tmp['x'], dtype = 'float')
            Y[idx*measurementsPerFile:(idx+1)*measurementsPerFile,:] = np.squeeze(tmp[0,:,:]).transpose()

        # Resize, strip the green/blue channel ( it is alrady scaled to [0, 1] when casting to float)
        X_c = np.zeros(shape=(len(X), np.prod(image_shape_new)))
        for idx in range(0, len(X)):
            # X_c[idx,:] = resize(X[idx,:].reshape(image_shape_old), (image_shape_new[1], image_shape_new[0]))[:,:,0:image_shape_new[2]].reshape((numelNewShape)).astype('float32')
            img = Image.fromarray(X[idx,:].reshape(image_shape_old))
            img = np.asarray( img.resize((image_shape_new[1], image_shape_new[0]), Image.ANTIALIAS), dtype="uint8" )
            X_c[idx,:] = img[:,:,0:2].astype('float32').reshape((numelNewShape)) / 255.
        # Flip, strip lidar measurement which are not in the frustum of the camera, and scale to [0, 1]
        X_l = np.fliplr(Y[:,lidar_shape_new-1:2*lidar_shape_new-1]).astype('float32') / lidar_range_max
        X_l[X_l == np.inf] = 0
        np.savez_compressed(filename, X_l=X_l, X_c=X_c)
    else:
        loaded = np.load(filename_npz)
        X_l = loaded["X_l"]
        X_c = loaded["X_c"]
    return X_l, X_c


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

