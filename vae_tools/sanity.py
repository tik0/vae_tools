#!/usr/bin/python

import keras
from keras import backend as K

def check():
    print("Keras version: " + str(keras.__version__))
    avail_gpus = K.tensorflow_backend._get_available_gpus()
    if not avail_gpus:
        assert False, 'No GPUs available'
    else:
        print("Available GPUs" + str(avail_gpus))

