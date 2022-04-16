"""
Load stellar models from CSV file
"""
from dataclasses import dataclass
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..gce import imf
from ..io import check_stellar_models
from ..utils import chem_elements


def read_stellar_csv() -> pd.DataFrame:
    """Reads a stellar model CSV from the filepath specified in config."""
    filepath = config.FILEPATHS["stellar_models"]
    df = pd.DataFrame()

    for fp in filepath:
        df_ = pd.read_csv(fp)
        df = pd.concat((df, df_))

    df = check_stellar_models.check_initial(df)

    return df


def read_ia_csv() -> pd.Series:
    """Reads a SNe Ia model CSV from the filepath specified in config.

    The composition of ejecta must sum to one, i.e. the composition must be provided in
    mass fraction of the ejecta. If there is a small error (<=1.e-3), scaling will be
    applied to ensure mass conservation during evolution.

    Returns
    -------
    pd.DataFrame
        The SNe Ia dataframe, with rescale chemical composition of there were small errors.

    Raises
    ------
    ValueError: If there is a large error in sum(mass fractions)
    """

    filepath = config.FILEPATHS["ia_model"]
    df = pd.read_csv(filepath)

    check = float(df.sum(axis="columns"))
    diff = abs(check - 1.0)

    if diff <= 1.0e-3:
        # Allow scaling of mass fractions. Sometimes there are rounding errors in data tables, etc.
        scale = 1 / check

        df = df * scale
    elif diff <= 1.0e-2:
        warnings.warn(
            f"SNe Ia model has significant errors ({np.round(diff, 3)}) "
            f"in sum(mass fractions). Evolution will likely fail to conserve mass."
        )
    else:
        raise ValueError(
            f"Large errors ({np.round(diff, 3)}) exist in "
            f"sum(mass fractions) for Ia input data. Should sum to one."
        )
    return df


def generate_stellarmodels_dataclass() -> dataclass:
    """Load stellar data CSV and use to generate a StellarModels dataclass."""
    model_data = read_stellar_csv()
    stellar_models = create_stellarmodels_dataclass_from_df(model_data)
    return stellar_models


def generate_iasystem_dataclass() -> dataclass:
    """Load Ia data CSV and use to generate a IaSystem dataclass."""
    model_data = read_ia_csv()
    ia_system = create_iasystem_dataclass_from_df(model_data)
    return ia_system


def create_stellarmodels_dataclass_from_df(models: pd.DataFrame) -> dataclass:
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


def create_iasystem_dataclass_from_df(model: pd.DataFrame) -> dataclass:
    """Read dataframe to populate IaSystem attributes."""
    elements_all = chem_elements.elements
    el2z = chem_elements.el2z
    # Include only the elements listed in the dataset.
    elements = list(set(model.columns).intersection(set(elements_all)))
    # Sort by charge number
    elements.sort(key=lambda x: el2z[x])
    # Store the index for each element in the array.
    x_idx = {el: int(i) for i, el in enumerate(elements)}
    x_ia = np.zeros(len(x_idx))
    for el, idx in x_idx.items():
        if el in model.columns:
            x_ia[idx] = float(model[el])
    mass_co = float(config.IA_PARAMS["mass_co"])
    imf_donor_slope = float(config.IA_PARAMS["imf_donor_slope"])
    mdl_rg = float(config.IA_PARAMS["mdl_rg"])
    mdu_rg = float(config.IA_PARAMS["mdu_rg"])
    mdl_ms = float(config.IA_PARAMS["mdl_ms"])
    mdu_ms = float(config.IA_PARAMS["mdu_ms"])
    mpl = float(config.IA_PARAMS["mpl"])
    mpu = float(config.IA_PARAMS["mpu"])
    b_rg = float(config.IA_PARAMS["b_rg"])
    b_ms = float(config.IA_PARAMS["b_ms"])

    masses_rg = np.arange(mdl_rg + 0.1, mdu_rg, 0.1)
    imf_ia_rg = imf.IMF(
        masses=masses_rg,
        slope=imf_donor_slope,
        mass_min=mdl_rg,
        mass_max=mdu_rg,
    )
    masses_ms = np.arange(mdl_ms + 0.1, mdu_ms, 0.1)
    imf_ia_ms = imf.IMF(
        masses=masses_ms,
        slope=imf_donor_slope,
        mass_min=mdl_ms,
        mass_max=mdu_ms,
    )

    ia_system = IaSystem(
        x_ia=x_ia,
        x_idx=x_idx,
        mass_co=mass_co,
        imf_donor_slope=imf_donor_slope,
        mdl_rg=mdl_rg,
        mdu_rg=mdu_rg,
        mdl_ms=mdl_ms,
        mdu_ms=mdu_ms,
        mpl=mpl,
        mpu=mpu,
        b_rg=b_rg,
        b_ms=b_ms,
        imf_ia_rg=imf_ia_rg,
        imf_ia_ms=imf_ia_ms,
    )

    return ia_system


@dataclass
class StellarModels:
    """Data representing the stellar models that contribute to the Galactic evolution.

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


@dataclass
class IaSystem:
    """Data representing the Ia systems that contribute to the Galactic evolution.

    Attributes
    ----------
    x_ia: np.array[float]
        Chemical abundances (mass fractions) in the ejecta of SNe Ia.
    x_idx: dict[str, int]
        Dictionary mapping elements to their index in arrays containing chemical abundances.
    mass_co: float
        The mass of the CO white dwarf (solar mass).
    imf_donor_slope: float
        Exponent of the IMF (phi_d(m) ~ m**-slope).
    mdl_rg: float
        Lower mass limit for red giants forming Ia systems.
    mdu_rg: float
        Upper mass limit for red giants forming Ia systems.
    mdl_ms: float
        Lower mass limit for main-sequence stars forming Ia systems.
    mdu_ms: float
        Upper mass limit for main-sequence stars forming Ia systems.
    mpl: float
        Lower mass limit of the progenitor stars forming WD's.
    mpu: float
        Upper mass limit of the progenitor stars forming WD's.
    b_rg: float
        Fraction of red giant stars that will form Ia systems.
    b_ms: float
        Fraction of main-sequence stars that will form Ia systems.
    imf_ia_rg: IMF
        The IMF representing the mass distribution of red giant stars.
    imf_ia_ms: IMF
        The IMF representing the mass distribution of main-sequence stars.
    """

    x_ia: np.array
    x_idx: Dict[str, int]
    mass_co: float
    imf_donor_slope: float
    mdl_rg: float
    mdu_rg: float
    mdl_ms: float
    mdu_ms: float
    mpl: float
    mpu: float
    b_rg: float
    b_ms: float
    imf_ia_rg: imf.IMF
    imf_ia_ms: imf.IMF
