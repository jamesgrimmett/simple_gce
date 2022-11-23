"""Approximate stellar winds and AGB models."""

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config
from ..utils import chem_elements
from . import approx_lifetime


def fill_agb(stellar_models: pd.DataFrame) -> pd.DataFrame:
    """Extend a set of stellar models to include approximate AGB models."""

    el2z = chem_elements.el2z
    mass_min_agb = 0.8  # solar masses, minimum stellar mass for AGB
    z_vals = stellar_models.Z.unique()
    m_vals = np.array(
        [max(mass_min_agb, 1.5 * config.IMF_PARAMS["mass_min"])] + [i for i in np.arange(1, 10)]
    )
    cols = stellar_models.columns
    elements = [
        col for col in cols if chem_elements.parse_chemical_symbol(col, silent=True) != (None, None)
    ]
    models_agb = pd.DataFrame(np.zeros((len(m_vals) * len(z_vals), len(cols))), columns=cols)

    z_m = itertools.product(z_vals, m_vals)

    for i in models_agb.index:
        z, m = next(z_m)
        models_agb.loc[i, "mass"] = m
        models_agb.loc[i, "Z"] = z
        models_agb.loc[i, "expl_energy"] = 0.0
        models_agb.loc[i, "type"] = str("agb")
        # Iben & Tutukov 1984, in Pagel 2009 after eqn 7.10
        if m <= 0.506:
            models_agb.loc[i, "remnant_mass"] = m
            models_agb.loc[i, "mass_final"] = m
        elif m <= 9.5:
            models_agb.loc[i, "remnant_mass"] = 0.45 + 0.11 * m
            models_agb.loc[i, "mass_final"] = 0.45 + 0.11 * m
        else:
            models_agb.loc[i, "remnant_mass"] = 1.5
            models_agb.loc[i, "mass_final"] = 1.5

    models_agb = approx_lifetime.fill_lifetimes(models_agb)

    models_agb[elements] = np.nan
    stellar_models = pd.concat((models_agb, stellar_models), ignore_index=True)

    return stellar_models


def fill_composition(ej_x_wind: np.ndarray, x: List[float], x_idx: Dict[str, int]) -> np.ndarray:
    """
    Set the composition of AGB/wind models where it is missing
    as the composition of the ISM with same metallicity.
    """

    # Avoid error propagation
    f_conserve = 1.0 / np.sum(x)

    for el, idx in x_idx.items():
        mask = np.where(np.isnan(ej_x_wind[:, :, idx]))
        ej_x_wind[mask[0], mask[1], idx] = f_conserve * float(x[idx])

    return ej_x_wind


def recycle_ism_composition(
    x_wind: np.ndarray,
    z_dim: List[float],
    z: float,
    x: List[float],
    x_idx: Dict[str, int],
    include_hn: bool,
    x_hn_wind: np.ndarray = None,
) -> tuple:
    """Fill the composition of AGB/wind models if they are to be approximated.

    This will recycle the composition of the ISM from the time of formation.

    Parameters
    ----------
    x_wind: np.ndarray
        The chemical composition (mass fraction) of wind ejecta for each model in initial
        mass - metallicity coordinates.
    z_dim: np.array
        The metallicities corresponding to the second axis coordinates of the model arrays.
    z: np.number
        The metallicity of the ISM in the Galaxy at the current timestep.
    x: np.array
        The chemical composition (mass fraction) of the ISM at the current timestep.
    x_idx: dict
        The element name : index dictionary to axis the arrays containing chemical compositions.
    include_hn: bool
        If True, HNe are included in the evolution of the Galaxy.
    x_hn_wind: np.ndarray
        The chemical composition (mass fraction) of wind ejecta for each HNe model in initial
        mass - metallicity coordinates.

    Returns
    -------
    x_wind_update: np.ndarray
        The chemical composition of wind ejecta, where every model with metallicity less than
        or equal to the current metallicity has been filled with the corresponding ISM abundance.
    x_wind_tmp: np.ndarray
        The same as `x_wind_update`, but the chemical composition of wind ejecta for models with
        metallicity greater than the current ISM metallicity have been filled with the current ISM
        chemical abundances for interpolation purposes.
    x_hn_wind_update: np.ndarray
        The same as above, but for HNe models, if they are included.
    x_hn_wind_tmp: np.ndarray
        The same as above, but for HNe models, if they are included.
    """
    x_wind_update = np.copy(x_wind)
    if np.isnan(x_wind_update[:, z_dim <= z]).any():
        x_wind_update[:, z_dim <= z] = fill_composition(x_wind_update[:, z_dim <= z], x, x_idx)
    x_wind_tmp = np.copy(x_wind_update)
    x_wind_tmp[:, z_dim > z] = fill_composition(x_wind_tmp[:, z_dim > z], x, x_idx)

    if include_hn:
        x_hn_wind_update = np.copy(x_hn_wind)
        if np.isnan(x_hn_wind_update[:, z_dim <= z]).any():
            x_hn_wind_update[:, z_dim <= z] = fill_composition(
                x_hn_wind_update[:, z_dim <= z], x, x_idx
            )
        x_hn_wind_tmp = np.copy(x_hn_wind_update)
        x_hn_wind_tmp[:, z_dim > z] = fill_composition(x_hn_wind_tmp[:, z_dim > z], x, x_idx)

        return x_wind_update, x_wind_tmp, x_hn_wind_update, x_hn_wind_tmp
    else:
        return x_wind_update, x_wind_tmp
