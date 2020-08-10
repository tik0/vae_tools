#!/usr/bin/python

try:
    import tensorflow
except:
    print("tensorflow module is not available")
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib
import platform

def check():
    print("python version: ", platform.python_version())
    print("keras version: " + str(keras.__version__))
    try:
        print("tensorflow version: " + str(tensorflow.version.VERSION))
    except:
        try: # backwards compat
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

def lr_normalizer(lr, optimizer):
    """Assuming a default learning rate 1, rescales the learning rate
    such that learning rates amongst different optimizers are more or less
    equivalent.
    Parameters
    ----------
    lr : float
        The learning rate.
    optimizer : keras optimizer
        The optimizer. For example, Adagrad, Adam, RMSprop.
    """

    from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
    from tensorflow.keras.optimizers import Nadam

    if optimizer == Adadelta:
        pass
    elif optimizer == SGD or optimizer == Adagrad:
        lr /= 100.0
    elif optimizer == Adam or optimizer == RMSprop:
        lr /= 1000.0
    elif optimizer == Adamax or optimizer == Nadam:
        lr /= 500.0
    else:
        raise ValueError(str(optimizer) + " is not supported by lr_normalizer")
    return lr