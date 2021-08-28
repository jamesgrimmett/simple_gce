"""
Load stellar models from CSV file
"""

import os
import pandas as pd
import numpy as np
from .. import config
from . import check_stellar_models
from ..utils import chem_elements, error_handling


def read_stellar_csv():
    filepath = config.FILEPATHS['stellar_models']
    df = pd.DataFrame()

    for fp in filepath:
        df_ = pd.read_csv(fp)
        df = pd.concat((df,df_))

    df = check_stellar_models.check_initial(df) 
    
    return df

def read_ia_csv():
    filepath = config.FILEPATHS['ia_model']
    df = pd.read_csv(filepath)

    check = float(df.sum(axis = 'columns'))
    diff = abs(check - 1.0)
    
    if diff > 1.e-3:
        raise error_handling.ProgramError("Ia yields do not sum to one.")

    scale = 1 / check

    df = df * scale

    check2 = float(df.sum(axis = 'columns'))
    diff2 = abs(check2 - 1.0)
    if diff2 >= 1.e-12:
        raise error_handling.ProgramError("Unable to scale mass fractions.")

    return df

def fill_arrays(models):
    """
    Fill arrays in mass-Z space with model properties.

    Args:
        models : Pandas dataframe of stellar models as described in ....
    Returns:
        dictionary containing:
        mass_dim : The coordinates in the mass dimension, i.e., 
                    an ordered list of model masses. 
        z_dim : The coordinates in the Z dimension, i.e., 
                    an ordered list of model metallicities. 
        x_idx : Dictionary mapping elements to their index in the array
        lifetime : 2D array of model lifetimes
        mass_final : 2D array of final mass (e.g., pre-supernova, after winds)
        mass_remnant : 2D array of compact remnant mass (NS/BH/WD).
        ej_x_cc : 3D array of ejecta chemical abundances (mass fractions)
                    from CCSNe.
        ej_x_wind : 3D array of ejecta chemical abundances (mass fractions)
                    from winds of CC/AGB models.
        ej_x_hn, ej_x_hn_wind, ej_weight : ..If more than one ejecta mode, then will need weights
    """
    include_hn = config.STELLAR_MODELS['include_hn']
    elements_all = chem_elements.elements
    el2z = chem_elements.el2z
    # Include only the elements listed in the dataset.
    elements = list(set(models.columns).intersection(set(elements_all)))
    # Sort by charge number
    elements.sort(key = lambda x : el2z[x])
    # Store the index for each element in the array.
    x_idx = {el : int(i) for i,el in enumerate(elements)}
    mass_dim = np.sort(models.mass.unique())
    z_dim = np.sort(models.Z.unique())

    lifetime = np.zeros((len(mass_dim), len(z_dim)))
    mass_final = np.zeros((len(mass_dim), len(z_dim)))
    mass_remnant = np.zeros((len(mass_dim), len(z_dim)))
    x_cc = np.zeros((len(mass_dim), len(z_dim), len(elements)))
    x_wind = np.zeros((len(mass_dim), len(z_dim), len(elements)))
    if include_hn:
        x_hn = np.zeros((len(mass_dim), len(z_dim), len(elements)))
        x_hn_wind = np.zeros((len(mass_dim), len(z_dim), len(elements)))
        mass_final_hn = np.zeros((len(mass_dim), len(z_dim)))
        mass_remnant_hn = np.zeros((len(mass_dim), len(z_dim)))

    for i,m in enumerate(mass_dim):
        for j,z in enumerate(z_dim):
            model = models[(models.mass == m) & (models.Z == z)]
            # TODO: move this to a process row type function
            lifetime_ = model.lifetime.unique()
            if len(lifetime_) > 1:
                raise ValueError(f"Trying to process \n"
                            f"{model[['mass','Z','type','lifetime']]}\n"
                            f"No functionality to deal with models of "
                            f"different lifetimes at the same (m,z) co-ordinate.")
            else:
                lifetime[i,j] = float(lifetime_)

            # fill the arrays for each model type present at (m,z)
            if 'cc' in list(model.type):
                model_cc = model[model.type == 'cc']
                x_cc[i,j,:] = np.array(model_cc[elements]).squeeze()
                mass_final[i,j] = float(model_cc.mass_final)
                mass_remnant[i,j] = float(model_cc.remnant_mass)
            else:
                x_cc[i,j,:] = 0.0
                
            if 'wind' in list(model.type):
                model_w = model[model.type == 'wind']
                x_wind[i,j,:] = np.array(model_w[elements]).squeeze()
            elif 'agb' in list(model.type):
                model_w = model[model.type == 'agb']
                x_wind[i,j,:] = np.array(model_w[elements]).squeeze()
                mass_final[i,j] = float(model_w.mass_final)
                mass_remnant[i,j] = float(model_w.remnant_mass)
            else:
                x_wind[i,j,:] = 0.0

            if include_hn == True:
                if 'hn' in list(model.type):
                    model_hn = model[model.type == 'hn']
                    x_hn[i,j,:] = np.array(model_hn[elements]).squeeze()
                    mass_final_hn[i,j] = float(model_hn.mass_final)
                    mass_remnant_hn[i,j] = float(model_hn.remnant_mass)
                else:
                    x_hn[i,j,:] = 0.0
                    mass_final_hn[i,j] = m
                    mass_remnant_hn[i,j] = m

                if 'hn_wind' in list(model.type):
                    model_w = model[model.type == 'wind']
                    x_hn_wind[i,j,:] = np.array(model_w[elements]).squeeze()
                else:
                    x_hn_wind[i,j,:] = 0.0

    arrays = {
        'mass_dim' : mass_dim,
        'z_dim' : z_dim,
        'x_idx' : x_idx,
        'lifetime' : lifetime,
        'mass_final' : mass_final,
        'mass_remnant' : mass_remnant,
        'x_cc' : x_cc,
        'x_wind' : x_wind,
    }
    if include_hn:
        arrays['x_hn'] = x_hn,
        arrays['x_hn_wind'] = x_hn_wind
        arrays['mass_final_hn'] = mass_final_hn
        arrays['mass_remnant_hn'] = mass_remnant_hn

    return arrays
