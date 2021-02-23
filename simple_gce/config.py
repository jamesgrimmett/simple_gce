"""Configuration parameters"""

import os

INSTALL_DIR = '/Users/james/Documents/github_code/simple_gce'

# Relevant filepaths
FILEPATHS = {
    # Path to stellar models. Must be a list, even if singular.
    'stellar_models' : [os.path.join(INSTALL_DIR,'data/stellar_models.csv')], 
    'test_stellar_models' : [os.path.join(INSTALL_DIR,'data/test_stellar_models.csv')],
}

IMF_PARAMS = {
    'mass_min' : 0.05,  # solar masses, minimum stellar mass.
    'mass_max' : 100.0, # solar masses, maximum stellar mass.
    'slope'    : 1.35,  # exponent of the IMF (phi(m) = m**-slope).
}

STELLAR_MODELS = {
    'mass_min_cc' : 10, # solar masses, minimum stellar mass for CCSNe.
    'mass_co' : 1.38,   # solar masses, CO white dwarf mass for SNeIa.
}

GALAXY_PARAMS = {
    'infall_timescale' : 5.e9,  # years. Kobayashi (2006) have 5.e9.
    'sfr_timescale' : 2.2e9,    # years. Kobayashi (2006) have 2.2e9.
    'sfr_init' : 2.e9,          # years. Kobayashi (2006) have 2.e9.
    'h_init' : 0.7513,          # mass fraction, hydrogen initial.
    'he_init' : 0.2487,         # mass fraction, helium initial.
}