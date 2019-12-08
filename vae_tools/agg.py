import os, matplotlib

def set_agg(force = False):
    ''' Checks if the 'USE_AGG' environment variable exists
    Needs to be imported before "import matplotlib.pyplot"
    '''
    try:
        if not force:
            os.environ['USE_AGG']
        matplotlib.use('Agg') # Use this to plot to Nirvana (assuming we run in a terminal)
        print("matplotlib: Use agg")
    except:
        pass

def check_backends():
    ''' Check all available backends
    Ref.: https://stackoverflow.com/a/43015816/2084944
    '''
    gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
    non_gui_backends = matplotlib.rcsetup.non_interactive_bk
    print ("Non Gui backends are:", non_gui_backends)
    print ("Gui backends I will test for", gui_env)
    for gui in gui_env:
        print ("testing", gui)
        try:
            matplotlib.use(gui,warn=False, force=True)
            from matplotlib import pyplot as plt
            print ("    ",gui, "Is Available")
            plt.plot([1.5,2.0,2.5])
            fig = plt.gcf()
            fig.suptitle(gui)
            plt.show()
            print ("Using ..... ",matplotlib.get_backend())
        except:
            print ("    ",gui, "Not found")