"""Approximate AGB models."""

import numpy as np
import pandas as pd
import itertools
from . import approx_lifetime
from .. import config
from ..utils import chem_elements

def fill_agb(stellar_models):
    """
    """

    el2z = chem_elements.el2z
    mass_min_agb = 0.8  # solar masses, minimum stellar mass for AGB
    mass_max_wd = 1.4   # solar masses, maximum white dwarf mass
    z_vals = stellar_models.Z.unique()
    m_vals = np.array([max(mass_min_agb,1.5*config.IMF_PARAMS['mass_min'])] + [i for i in np.arange(1,10)])
    cols = stellar_models.columns
    elements = list(set(cols).intersection(set(el2z.keys())))
    models_agb = pd.DataFrame(np.zeros((len(m_vals)*len(z_vals),len(cols))),columns=cols)

    z_m = itertools.product(z_vals,m_vals)

    for i in models_agb.index:
        z,m = next(z_m)
        models_agb.loc[i,'mass'] = m
        models_agb.loc[i,'Z'] = z
        models_agb.loc[i,'expl_energy'] = 0.0
        models_agb.loc[i,'type'] = str('agb')
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

    models_agb = approx_lifetime.fill_lifetimes(models_agb)

    models_agb[elements] = np.nan
    stellar_models = pd.concat((models_agb, stellar_models), ignore_index = True)

    return stellar_models

def fill_composition(ej_x_wind, z, x, x_idx):
    """
    Set the composition of AGB/wind models where it is missing 
    as the composition of the ISM with same metallicity. 
    """
 
    # Avoid error propagation 
    f_conserve = 1.0 / np.sum(x)

    for el, idx in x_idx.items():
        mask = np.where(np.isnan(ej_x_wind[:,:,idx]))
        ej_x_wind[mask[0], mask[1], idx] = f_conserve * float(x[idx])

    return ej_x_wind
