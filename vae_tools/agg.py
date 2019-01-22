import os, matplotlib

# Checks if the 'USE_AGG' environment variable exists
# Needs to be imported before "import matplotlib.pyplot"
def set_agg(force = False):
    try:
        if not force:
            os.environ['USE_AGG']
        matplotlib.use('Agg') # Use this to plot to Nirvana (assuming we run in a terminal)
        print("matplotlib: Use agg")
    except:
        pass