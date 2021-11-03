"""Configuration parameters to run a model similar to Kobayashi et al., (2006)
    https://arxiv.org/abs/astro-ph/0608688
"""

import os

INSTALL_DIR = "/Users/james/Documents/github_code/simple_gce"

# Relevant filepaths
FILEPATHS = {
    # Path to stellar models. Must be a list, even if singular.
    "stellar_models": [
        os.path.join(INSTALL_DIR, "data/kobayashi-2006/kobayashi-2006_sn_scaled.csv"),
        os.path.join(INSTALL_DIR, "data/kobayashi-2006/kobayashi-2006_hn_scaled.csv"),
    ],
    "ia_model": os.path.join(
        INSTALL_DIR, "data/kobayashi-2006/kobayashi-2006_ia_scaled.csv"
    ),
}

IMF_PARAMS = {
    "mass_min": 0.05,  # solar masses, minimum stellar mass.
    "mass_max": 50.0,  # solar masses, maximum stellar mass.
    "slope": 1.35,  # exponent of the IMF (phi(m) ~ m**-slope).
}

# Parameters for the Ia system, outlined in Kobayashi 2000 Section 2.1
IA_PARAMS = {
    "imf_donor_slope": 0.35,  # exponent of the IMF (phi_d(m) ~ m**-slope).
    "mass_co": 1.38,  # solar masses, CO white dwarf mass for SNeIa.
    "mdl_rg": 0.9,  # lower limit of the red giant scenario
    "mdu_rg": 1.5,  # upper limit of the red giant scenario
    "mdl_ms": 1.8,  # lower limit of the main-sequence scenario
    "mdu_ms": 2.6,  # upper limit of the main-sequence scenario
    "mpl": 3.0,  # lower limit of WD progenitor
    "mpu": 8.0,  # upper limit of WD progenitor
    "b_rg": 0.04,  # fraction of primary stars that produce Ia
    "b_ms": 0.04,  # fraction of primary stars that produce Ia
}

STELLAR_MODELS = {
    "mass_min_cc": 10,  # solar masses, minimum stellar mass for CCSNe.
    "include_hn": True,
    "hn_frac": 0.5,  # fraction of massive stars which explode as HNe.
}

GALAXY_PARAMS = {
    "total_mass": 1.0e12,  # solar masses, total baryonic mass available to form a galaxy.
    "infall_timescale": 5.0e9,  # years. Kobayashi (2006) have 5.e9.
    "sfr_timescale": 2.2e9,  # years. Kobayashi (2006) have 2.2e9.
    "sfr_init": 0.0,  # solar masses per year.
    "z_init": 0.0,  # metallicity
    "h_init": 0.7513,  # mass fraction, hydrogen initial.
    "he_init": 0.2487,  # mass fraction, helium initial.
}
