import keras
import numpy as np

def set_layerweights_trainable(model, ln_full = None, ln_start = None, ln_any = None, trainable = False):
    """
    Set layers with given layername (ln) expression (non-) trainable
    """
    for layer in model.layers:
        if ln_full != None:
            if layer.name == ln_full:
                layer.trainable = trainable
        if ln_start != None:
            if layer.name.find(ln_start) == 0:
                layer.trainable = trainable
        if ln_any != None:
            if layer.name.find(ln_any) >= 0:
                layer.trainable = trainable
    return model

def set_layerweights_trainable_global(model, trainable = False):
    """
    Set all layers (non) trainable
    """
    for layer in model.layers:
        layer.trainable = trainable

def set_layerweights(model, ln_full = None, ln_start = None, ln_any = None, gain = 0., bias = 0.):
    """
    Reset decoder weights of a layer with a given name expression
    """
    session = keras.backend.get_session()
    for layer in model.layers:
        set_layer = False
        if ln_full != None:
            if layer.name == ln_full:
                set_layer = True
        if ln_start != None:
            if layer.name.find(ln_start) == 0:
                set_layer = True
        if ln_any != None:
            if layer.name.find(ln_any) >= 0:
                set_layer = True
        if set_layer:
            layer.set_weights([gain * np.ones(layer.get_weights()[0].shape),
                               bias * np.zeros(layer.get_weights()[1].shape)])

def reset_layerweights(model, ln_full = None, ln_start = None, ln_any = None):
    """
    Reset decoder weights of a layer with a given name expression
    """
    session = keras.backend.get_session()
    for layer in model.layers:
        reset = False
        if ln_full != None:
            if layer.name == ln_full:
                reset = True
        if ln_start != None:
            if layer.name.find(ln_start) == 0:
                reset = True
        if ln_any != None:
            if layer.name.find(ln_any) >= 0:
                reset = True
        if reset:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

def get_config(model, layername = None):
    """
    Get layer config of all or one specific layer
    """
    for layer in model.layers:
        if layer.name == layername or layername == None:
            print('Layer config:')
            print(layer.get_config())

def get_weights(model, layername = None):
    """
    Get layer weights of all or one specific layer
    """
    for layer in model.layers:
        if layer.name == layername or layername == None:
            print('Layer weights:') # list of numpy arrays
            print(layer.get_weights())
