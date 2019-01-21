#!/usr/bin/python

try:
    import tensorflow
except:
    print("tensorflow module is not available")
import keras
from keras import backend as K
import matplotlib

def check():
    print("keras version: " + str(keras.__version__))
    try:
        print("tensorflow version: " + str(tensorflow.VERSION))
    except:
        pass
    print("matplotlib uses: ", matplotlib.rcParams['backend']) 
    try: # keras was loaded
        avail_gpus = K.tensorflow_backend._get_available_gpus()
    except: # tf.keras was loaded
        avail_gpus = tensorflow.test.is_gpu_available()
    if not avail_gpus:
        print('No GPUs available')
    else:
        print("Available GPUs " + str(avail_gpus))

