"""A simple one-zone GCE model.

TODO: add a description of where the config can be changed, 
and where data will be loaded from.

Typical usage example:

TODO: add a typical usage example, including setup, evolve, and plotting
results.
"""

def main():
    """
    The main function to call, to evolve the galaxy.
    """"

    nuclei = cfg.SETUP['nuclei']

    models_cc = setup_models_cc(model_set = '...', )

    models_agb = setup_models_agb(model_set = '...')

    models_ia = setup_models_ia(model_set = '...')

    # check nuclei match all models?

    imf, models_

    ...

    # t = 0 output
    while t < t_max:
        dt = calc_timestep()
        t += dt
        hn_frac = calc_hnfrac(data['Z'])
        data = evolve_timestep(imf, models, ...)




def setup_models_cc(model_set = '...', ):
    """
    Read raw models into dataframe, interpolate log(Z),  
    """

def setup_models_agb(model_set = '...')
    """
    Read raw models into dataframe, interpolate log(Z).
    Dummy option for AGB models as in Grimmett 2020.  
    """

def setup_models_ia(model_set = '...', ):
    """
    Read raw models into dataframe, interpolate log(Z),  
    """