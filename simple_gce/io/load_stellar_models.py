"""
Load stellar models from CSV file
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config
from ..utils import chem_elements, error_handling
from . import check_stellar_models


def read_stellar_csv():
    filepath = config.FILEPATHS["stellar_models"]
    df = pd.DataFrame()

    for fp in filepath:
        df_ = pd.read_csv(fp)
        df = pd.concat((df, df_))

    df = check_stellar_models.check_initial(df)

    return df


def read_ia_csv():
    filepath = config.FILEPATHS["ia_model"]
    df = pd.read_csv(filepath)

    check = float(df.sum(axis="columns"))
    diff = abs(check - 1.0)

    if diff > 1.0e-3:
        raise error_handling.ProgramError("Ia yields do not sum to one.")

    scale = 1 / check

    df = df * scale

    check2 = float(df.sum(axis="columns"))
    diff2 = abs(check2 - 1.0)
    if diff2 >= 1.0e-12:
        raise error_handling.ProgramError("Unable to scale mass fractions.")

    return df


def generate_stellarmodels_dataclass():
    """Load stellar data CSV and use to generate a StellarModels dataclass."""
    model_data = read_stellar_csv()
    stellar_models = create_dataclass_from_df(model_data)
    return stellar_models


def create_dataclass_from_df(models):
    """Read dataframe to populate StellarModels attributes."""
    include_hn = config.STELLAR_MODELS["include_hn"]
    elements_all = chem_elements.elements
    el2z = chem_elements.el2z
    # Include only the elements listed in the dataset.
    elements = list(set(models.columns).intersection(set(elements_all)))
    # Sort by charge number
    elements.sort(key=lambda x: el2z[x])
    # Store the index for each element in the array.
    x_idx = {el: int(i) for i, el in enumerate(elements)}
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
        min_mass_hn = models[models.type == "hn"].mass.min()

    for i, m in enumerate(mass_dim):
        for j, z in enumerate(z_dim):
            model = models[(models.mass == m) & (models.Z == z)]
            # TODO: move this to a process row function
            lifetime_ = model.lifetime.unique()
            if len(lifetime_) > 1:
                raise ValueError(
                    f"Trying to process \n"
                    f"{model[['mass','Z','type','lifetime']]}\n"
                    f"No functionality to deal with models of "
                    f"different lifetimes at the same (m,z) co-ordinate."
                )
            else:
                lifetime[i, j] = float(lifetime_)

            # fill the arrays for each model type present at (m,z)
            if "cc" in list(model.type):
                model_cc = model[model.type == "cc"]
                x_cc[i, j, :] = np.array(model_cc[elements]).squeeze()
                mass_final[i, j] = float(model_cc.mass_final)
                mass_remnant[i, j] = float(model_cc.remnant_mass)
            else:
                x_cc[i, j, :] = 0.0

            if "wind" in list(model.type):
                model_w = model[model.type == "wind"]
                x_wind[i, j, :] = np.array(model_w[elements]).squeeze()
            elif "agb" in list(model.type):
                model_w = model[model.type == "agb"]
                x_wind[i, j, :] = np.array(model_w[elements]).squeeze()
                mass_final[i, j] = float(model_w.mass_final)
                mass_remnant[i, j] = float(model_w.remnant_mass)
            else:
                x_wind[i, j, :] = 0.0

            if include_hn is True:
                if "hn" in list(model.type):
                    model_hn = model[model.type == "hn"]
                    x_hn[i, j, :] = np.array(model_hn[elements]).squeeze()
                    mass_final_hn[i, j] = float(model_hn.mass_final)
                    mass_remnant_hn[i, j] = float(model_hn.remnant_mass)
                else:
                    x_hn[i, j, :] = 0.0
                    mass_final_hn[i, j] = m
                    mass_remnant_hn[i, j] = m

                if "hn_wind" in list(model.type):
                    model_w = model[model.type == "hn_wind"]
                    x_hn_wind[i, j, :] = np.array(model_w[elements]).squeeze()
                else:
                    x_hn_wind[i, j, :] = 0.0

    w_cc = ((mass_final - mass_remnant).T / mass_dim).T
    w_wind = ((mass_dim - mass_final.T) / mass_dim).T
    if include_hn:
        w_hn = ((mass_final_hn - mass_remnant_hn).T / mass_dim).T
        w_hn_wind = ((mass_dim - mass_final_hn.T) / mass_dim).T

    if not include_hn:
        stellar_models = StellarModels(
            mass_dim=mass_dim,
            z_dim=z_dim,
            elements=elements,
            x_idx=x_idx,
            lifetime=lifetime,
            mass_final=mass_final,
            mass_remnant=mass_remnant,
            x_cc=x_cc,
            x_wind=x_wind,
            w_cc=w_cc,
            w_wind=w_wind,
        )
    else:
        stellar_models = StellarModels(
            mass_dim=mass_dim,
            z_dim=z_dim,
            elements=elements,
            x_idx=x_idx,
            lifetime=lifetime,
            mass_final=mass_final,
            mass_remnant=mass_remnant,
            x_cc=x_cc,
            x_wind=x_wind,
            w_cc=w_cc,
            w_wind=w_wind,
            mass_final_hn=mass_final_hn,
            mass_remnant_hn=mass_remnant_hn,
            x_hn=x_hn,
            x_hn_wind=x_hn_wind,
            w_hn=w_hn,
            w_hn_wind=w_hn_wind,
            min_mass_hn=min_mass_hn,
        )

    return stellar_models


@dataclass
class StellarModels:
    """Data representing the stellar models that drive the Galactic evolution.

    Attributes
    ----------
    mass_dim: (M,) np.array[np.number]
        An ordered list of model masses (ZAMS; solar mass).
    z_dim: (Z,) np.array[np.number]
        An ordered list of model metallicities.
    elements: (E,) list[str]
        The chemical elements included in the evolution.
    x_idx: dict[str, int]
        Dictionary mapping elements to their index in arrays containing chemical abundances.
    lifetime: (M,Z) np.ndarray[float]
        Main sequence lifetimes (years).
    mass_final: (M,Z) np.ndarray[float]
        Final mass, i.e., pre-supernova, after winds (solar mass).
    mass_remnant: (M,Z) np.ndarray[float]
        Mass of the compact remnant (NS/BH/WD; solar mass).
    x_cc: (M,Z,E) np.ndarray[float]
        Chemical abundances (mass fractions) in the ejecta of CCSNe.
    x_wind: (M,Z,E) np.ndarray[float]
        Chemical abundances (mass fractions) in the wind ejecta (including AGB).
    w_cc: (M,Z) np.ndarray[float]
        Total ejecta mass from CCSNe, as a fraction of total initial stellar mass.
    w_wind: (M,Z) np.ndarray[float]
        Total ejecta mass from winds, as a fraction of total initial stellar mass.
    mass_final_hn: (M,Z) np.ndarray[float]
        Final mass, i.e., pre-supernova, after winds (solar mass).
    mass_remnant_hn: (M,Z) np.ndarray[float]
        Mass of the compact remnant (NS/BH/WD; solar mass).
    x_hn: (M,Z,E) np.ndarray[float]
        Chemical abundances (mass fractions) in the ejecta of HNe.
    x_hn_wind: (M,Z,E) np.ndarray[float]
        Chemical abundances (mass fractions) in the wind ejecta (inc. AGB).
    w_hn: (M,Z) np.ndarray[float]
        Total ejecta mass from HNe, as a fraction of total initial stellar mass.
    w_hn_wind: (M,Z) np.ndarray[float]
        Total ejecta mass from winds, as a fraction of total initial stellar mass.
    min_mass_hn: np.number
        The minimum ZAMS mass of the included HNe models.
    """

    mass_dim: np.array
    z_dim: np.array
    elements: List[str]
    x_idx: Dict[str, int]
    lifetime: np.ndarray
    mass_final: np.ndarray
    mass_remnant: np.ndarray
    x_cc: np.ndarray
    x_wind: np.ndarray
    w_cc: np.ndarray
    w_wind: np.ndarray
    mass_final_hn: np.ndarray = None
    mass_remnant_hn: np.ndarray = None
    x_hn: np.ndarray = None
    x_hn_wind: np.ndarray = None
    w_hn: np.ndarray = None
    w_hn_wind: np.ndarray = None
    min_mass_hn: np.number = None
