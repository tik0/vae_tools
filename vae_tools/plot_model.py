import os
import keras
from PIL import Image                                                                                
from random import randint
from IPython.display import SVG
from IPython.display import display

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
    svg = keras.utils.vis_utils.model_to_dot(model).create(prog='dot', format='svg')
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
