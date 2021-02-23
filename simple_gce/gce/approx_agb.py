"""Approximate AGB models."""

import numpy as np
import pandas as pd
import itertools
from .. import config

def fill_agb(stellar_models):
    """
    """

    mass_min_agb = 0.8  # solar masses, minimum stellar mass for AGB
    mass_max_wd = 1.4   # solar masses, maximum white dwarf mass
    z_vals = stellar_models.Z.unique()
    m_vals = np.array([max(mass_min_agb,1.5*config.IMF_PARAMS['mass_min'])] + [i for i in np.arange(1,10)])
    cols = stellar_models.columns
    models_agb = pd.DataFrame(np.zeros((len(m_vals)*len(z_vals),len(cols))),columns=cols)

    z_m = itertools.product(z_vals,m_vals)

    for i in models_agb.index:
        z,m = next(z_m)
        models_agb.loc[i,'mass'] = m
        models_agb.loc[i,'Z'] = z
        models_agb.loc[i,'expl_energy'] = 0.0
        models_agb.loc[i,'type'] = str('AGB')
        # Iben & Tutukov 1984, in Pagel 2009 after eqn 7.10
        if (m <= 0.506):
            models_agb.loc[i,'remnant_mass'] = m
            models_agb.loc[i,'mass_final'] = m
        elif (m <= 9.5):
            models_agb.loc[i,'remnant_mass'] = 0.45 + 0.11 * m
            models_agb.loc[i,'mass_final'] = 0.45 + 0.11 * m
        else:
            models_agb.loc[i,'remnant_mass'] = 1.5
            models_agb.loc[i,'mass_final'] = 1.5

        models_agb.loc[i,'lifetime'] = 1.e9#lt.lifetime_m(m,z)

    stellar_models = pd.concat((models_agb, stellar_models), ignore_index = True)

    return stellar_models
