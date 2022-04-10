"""Check stellar model data for completeness, inconsistencies, etc."""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..gce import approx_lifetime, approx_winds
from ..utils.chem_elements import el2z

REQUIRED_COLUMNS = [
    "mass",  # ZAMS mass
    "mass_final",  # Final mass (e.g. presupernova, after wind-loss)
    "type",  # CC, AGB, Ia
    "remnant_mass",  # Mass of the compact remnant.
    "Z",  # Initial metallicity
    "lifetime",  # Stellar lifetime.
]

OPTIONAL_COLUMNS = ["expl_energy"]


def check_initial(df: pd.DataFrame) -> pd.DataFrame:
    """Call the relevant checks after loading the initial data.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe with any modifications made during validation.
    """
    include_hn = config.STELLAR_MODELS["include_hn"]

    df = check_massfracs(df)
    df = check_model_types(df)
    check_columns(df)
    df = check_missing_vals(df)
    check_mass_metallicity_consistent(df)
    if include_hn:
        df = check_hn_models(df)
    df = check_wind_component(df)

    return df


def check_columns(df: pd.DataFrame) -> None:
    """Check that all of the necessary columns are present.

    Where possible, fill missing columns appropriately.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Raises
    -------
    ValueError: If required columns are missing from input data, or if there are
        unknown columns in the data.
    """

    # Columns excluding required
    non_req_columns = set(df.columns).difference(set(REQUIRED_COLUMNS))
    # Columns excluding required and optional
    remaining_columns = non_req_columns.difference(set(OPTIONAL_COLUMNS))
    # Columns excluding required, optional, and chemical elements
    unknown_columns = remaining_columns.difference(set(el2z.keys()))
    # Required columns that are missing
    missing_columns = list(set(REQUIRED_COLUMNS) - set(df.columns))

    if len(missing_columns) != 0:
        raise ValueError(f"Stellar data is missing required columns:\n{missing_columns}")
    elif len(unknown_columns) != 0:
        raise ValueError(f"Unknown columns/elements in stellar data:\n{unknown_columns}")


def check_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Check that all required data is present.

    `mass`, `mass_final`, `remnant_mass`, `type`, and `Z` must be filled for each row.
    `lifetime` can be NaN, it will be approximated from mass and metallicity.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe with filled lifetime values if they were NaN.

    Raises
    ------
    ValueError: If any required column contains NaNs
    """

    if not df[df["mass"].isna()].empty:
        raise ValueError(
            "Missing some entries for mass. This column should contain ZAMS for each model."
        )
    if not df[df["type"].isna()].empty:
        raise ValueError(
            "Missing some entries for type. This column should contain a string indicating "
            "the type of each model, i.e., one of 'cc', 'cc_wind', 'agb', 'hn', or 'hn_wind'."
        )
    if not df[df["mass_final"].isna()].empty:
        raise ValueError(
            "Missing some entries for mass_final. "
            "If there are no winds/mass loss in a model, set mass_final = mass."
        )
    if not df[df["remnant_mass"].isna()].empty:
        raise ValueError(
            "Missing some entries for remnant_mass. "
            "If there is no remnant (e.g. SNe Ia), set this column to zero. "
            "For AGB stars, set mass_remnant = mass_final"
        )
    if not df[df["Z"].isna()].empty:
        raise ValueError("Missing some entries for Z (metallicity).")
    if not df[df["lifetime"].isna()].empty:
        warnings.warn("Missing stellar lifetime data. Filling with approx values.")
        df = approx_lifetime.fill_lifetimes(df)

    return df


def check_model_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model types may include core-collapse (cc), asymptotic giant-branch (agb),
    thermonuclear (ia), and hypernovae (hn). Core-collapse and hypernovae models
    may also have wind components (cc_wind, hn_wind).

    Models must include core-collapse. If AGB models are not included, they will
    be approximated by recycling the ISM compisition from time of birth. Wind
    components for CCSNe and HNe will likewise be approximated if they are not included.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe, with AGB models added if they were missing.

    Raises
    ------
    ValueError: If cc models are not present, or unknown models types are present.
    """
    known_types = ["cc", "agb", "hn", "cc_wind", "hn_wind"]

    types = df.type.unique()

    if not set(types).issubset(set(known_types)):
        unknown_types = list(set(types).difference(set(known_types)))
        raise ValueError(
            f"Unknown model type/s : {unknown_types}. \n"
            f"Model types must be one of {known_types}."
        )

    if "cc" not in types:
        raise ValueError('Stellar models must include core-collapse (type = "CC")')

    if "agb" not in types:
        mass_min_cc = config.STELLAR_MODELS["mass_min_cc"]
        message = (
            f"No AGB models found. Wind mass from stars with mass < {mass_min_cc} Msun wil be "
            f"approximated, and wind chemical composition will be unchanged from formation."
        )
        warnings.warn(message)

        df = approx_winds.fill_agb(df)

    if config.STELLAR_MODELS["include_hn"] is True:
        if "hn" not in types:
            warnings.warn(
                "HNe are not included in model input data, "
                "though include_hn is specified in config."
            )
    else:
        if "hn" in types or "hn_wind" in types:
            warnings.warn("Config is set to exclude HNe. Removing HNe models.")
            df = df[(df.type != "hn") & (df.type != "hn_wind")]

    return df


def check_massfracs(df: pd.DataFrame) -> pd.DataFrame:
    """
    The composition of ejecta must sum to one, i.e. the composition must be provided in
    mass fraction of the ejecta. If there is a small error (<=1.e-3), scaling will be
    applied to ensure mass conservation during evolution.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe, with rescale chemical composition of there were small errors.

    Raises
    ------
    ValueError: If cc models are not present, or unknown models types are present.
    """
    elements = el2z.keys()
    elements = list(set(df.columns).intersection(set(elements)))

    check = df[elements].sum(axis=1)
    diff = abs(check - 1.0)

    if diff.max() <= 1.0e-3:
        # Allow scaling of mass fractions. Sometimes there are rounding errors in data tables, etc.
        scale = 1 / check

        df.loc[:, elements] = df[elements].mul(scale, axis="rows")
    elif diff.max() <= 1.0e-2:
        warnings.warn(
            f"Some stellar models have significant errors ({np.round(diff.max(), 3)}) "
            f"in sum(mass fractions). Evolution will likely fail to conserve mass."
        )
    else:
        raise ValueError(
            f"Large errors ({np.round(diff.max(), 3)}) exist in "
            f"sum(mass fractions) for input data. Should sum to one."
        )

    return df


def check_mass_metallicity_consistent(df: pd.DataFrame) -> None:
    """
    Each model of mass `m` must have the same set of metallicities `z`.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Raises
    ------
    ValueError: If metallicities are not consistent between models.
    """
    z_list = df.Z.unique()
    m_list = df.mass.unique()

    for m in m_list:
        z_ = df[df.mass == m].Z.unique()
        if sorted(z_) != sorted(z_list):
            raise ValueError(
                f"Error in models with mass {m}. Each model of mass `m` must have the same "
                f"set of metallicities (Z)."
            )


def check_wind_component(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that each CC/HN model has a corresponding row for the wind composition.
    If any are missing, add a row, where the composition will be filled during the evolution
    calculation with the ISM composition at the time of birth.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe, with wind rows added if they were missing.
    """
    elements = list(set(df.columns).intersection(set(el2z.keys())))

    for i, model in df[(df.type == "cc")].iterrows():
        if "wind" not in df[(df.mass == model.mass) & (df.Z == model.Z)].type:
            new_model = model.copy()
            new_model["type"] = "wind"
            new_model[elements] = np.nan
            df = df.append(new_model, ignore_index=True)

    if config.STELLAR_MODELS["include_hn"] is True:
        for i, model in df[(df.type == "hn")].iterrows():
            if "hn_wind" not in df[(df.mass == model.mass) & (df.Z == model.Z)].type:
                new_model = model.copy()
                new_model["type"] = "hn_wind"
                new_model[elements] = np.nan
                df = df.append(new_model, ignore_index=True)

    return df


def check_hn_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    If HNe models are included, there masses must be either;
    (i) a subset of the SNe models, or,
    (ii) intersect with the upper end of SNe masses.

    The chemical contribution from each mass above the minimum HN mass is calculated as a
    weighted average of the HN and SN models. If the case is (ii), then then SNe masses
    will be extended to match the upper mass of HNe models, but the additional sn models
    will be assumed to collapse to BH without ejecta.

    Example; case (i) might be masses sn: [10, 15, 20, 30] and hn: [20, 30].
    Case (ii) might be masses sn: [10, 15, 20, 30] and hn: [20, 30, 40, 60], and the sn
    masses would be extended with two additional models, [10, 15, 20, 30, 40, 60], where
    the 40 and 60 solar mass sn models would had remnant mass == ZAMS mass.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing stellar models.

    Returns
    -------
    pd.DataFrame
        The input dataframe, with high mass sn models added if necessary to match hn masses.

    Raises
    ------
    ValueError: If cc models masses are incompatible with the hn model masses.
    """

    elements = el2z.keys()
    elements = list(set(df.columns).intersection(set(elements)))

    mass_hn = np.unique(df[df.type == "hn"].mass)
    mass_sn = np.unique(df[df.type == "cc"].mass)
    mass_max_sn = np.max(mass_sn)

    hn_xover = mass_hn[mass_hn <= mass_max_sn]

    if not sorted(hn_xover) == sorted(hn_xover):
        raise ValueError(
            f"There must be a solid crossover between SN and "
            f"HN masses.\nSN masses: {sorted(mass_sn)}\n"
            f"HN masses: {sorted(mass_hn)}"
        )

    # Extend SN models to match upper mass of HN models.
    # New SN models have same properties as corresponding HN model, but ejecta mass will be zero.
    if mass_max_sn < np.max(mass_hn):
        extend_sn = pd.DataFrame(df[(df.type == "hn") & (df.mass > mass_max_sn)])
        extend_sn["remnant_mass"] = list(extend_sn.mass)
        extend_sn["mass_final"] = list(extend_sn.mass)
        extend_sn["type"] = "cc"
        extend_sn.loc[:, elements] = 0.0
        df = df.append(extend_sn, ignore_index=True)

    return df
