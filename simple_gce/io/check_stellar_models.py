"""Check stellar model data for completeness, inconsistencies, etc."""
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..gce import approx_winds, approx_lifetime
from ..utils import error_handling
from ..utils.chem_elements import el2z

# TODO:
# check mass limits provided in config.
# def check_winds(df):
#   models with wind and CC, check consistency of mass / mass_final
#   models with CC and no wind, check mass == mass_final

# def check_ia(df):
#   check consistency with config params

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


def check_columns(df: pd.DataFrame):
    """
    Check that all of the necessary columns are present. Where possible, fill
    missing columns appropriately.
    """

    # TODO:
    # - Approximate and fill remnant_mass, mass_presn, if missing.

    # Columns excluding required
    non_req_columns = set(df.columns).difference(set(REQUIRED_COLUMNS))
    # Columns excluding required and optional
    remaining_columns = non_req_columns.difference(set(OPTIONAL_COLUMNS))
    # Columns excluding required, optional, and chemical elements
    unknown_columns = remaining_columns.difference(set(el2z.keys()))

    if not set(REQUIRED_COLUMNS).issubset(set(df.columns)):
        raise error_handling.InsufficientDataError("Stellar data is missing required columns.")
    elif len(unknown_columns) != 0:
        raise error_handling.UnknownCaseError(
            f"Unknown columns/elements in stellar data.\n{unknown_columns}"
        )


def check_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Check that all required data is present."""

    if not df[df["mass_final"].isna()].empty:
        raise error_handling.InsufficientDataError(
            "Missing some entries for mass_final. "
            "If there are no winds/mass loss in a model, set mass_final = mass."
        )
    if not df[df["remnant_mass"].isna()].empty:
        raise error_handling.InsufficientDataError(
            "Missing some entries for remnant_mass. "
            "If there is no remnant (e.g. SNe Ia), set this column to zero. "
            "For AGB stars, set mass_remnant = mass_final"
        )
    if not df[df["Z"].isna()].empty:
        raise error_handling.InsufficientDataError(
            "Missing some entries for Z (metallicity). "
            "If metallicity is not relevant, or you have models with a single value "
            "of metallicity, set this column to zero for those models."
        )
    if not df[df["lifetime"].isna()].empty:
        warnings.warn("Missing stellar lifetime data. Filling with approx values.")
        df = approx_lifetime.fill_lifetimes(df)

    return df


def check_model_types(df: pd.DataFrame) -> pd.DataFrame:
    # Types may include core-collapse, asymptotic giant-branch, thermonuclear,
    # and hypernovae.
    known_types = ["cc", "agb", "hn", "cc_wind", "hn_wind"]

    types = df.type.unique()

    if not set(types).issubset(set(known_types)):
        unknown_types = list(set(types).difference(set(known_types)))
        raise error_handling.UnknownCaseError(f"Unknown model type/s : {unknown_types}")

    if "cc" not in types:
        raise error_handling.InsufficientDataError(
            'Stellar models must include core-collapse (type = "CC")'
        )

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
            message = "HNe are not included."
            warnings.warn(message)
    else:
        if "hn" in types:
            message = "Config is set to exclude HNe. Removing HNe models."
            warnings.warn(message)
            df = df[df.type != "hn"]

    return df


def check_massfracs(df: pd.DataFrame) -> pd.DataFrame:
    elements = el2z.keys()
    elements = list(set(df.columns).intersection(set(elements)))

    check = df[elements].sum(axis=1)
    diff = abs(check - 1.0)

    if diff.max() >= 1.0e-3:
        warnings.warn(
            f"Some stellar models have significant errors ({np.round(diff.max(), 3)}) "
            f"in sum(mass fractions)."
        )
    if diff.max() >= 1.0e-2:
        raise error_handling.ProgramError(
            f"Large errors ({np.round(diff.max(), 3)}) in sum(mass fractions) for stellar models"
        )

    # Allow scaling of mass fractions. Sometimes there are rounding errors in data tables, etc.
    scale = 1 / check

    df.loc[:, elements] = df[elements].mul(scale, axis="rows")

    check2 = abs(df[elements].sum(axis=1) - 1.0)
    if check2.max() >= 1.0e-12:
        raise error_handling.ProgramError("Unable to scale mass fractions.")

    return df


def check_mass_metallicity_consistent(df: pd.DataFrame):
    z_list = df.Z.unique()
    m_list = df.mass.unique()

    for m in m_list:
        z_ = df[df.mass == m].Z.unique()
        if set(z_) != set(z_list):
            raise error_handling.NotImplementedError()


def check_wind_component(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that CC/HN models where mass != mass_final, that a row is added
    to store wind composition
    """
    elements = list(set(df.columns).intersection(set(el2z.keys())))

    for i, model in df[(df.type == "cc")].iterrows():  # & (df.mass != df.mass_final)].iterrows():
        if "wind" not in df[(df.mass == model.mass) & (df.Z == model.Z)].type:
            new_model = model.copy()
            new_model["type"] = "wind"
            new_model[elements] = np.nan
            df = df.append(new_model, ignore_index=True)

    if config.STELLAR_MODELS["include_hn"] is True:
        for i, model in df[
            (df.type == "hn")
        ].iterrows():  # & (df.mass != df.mass_final)].iterrows():
            if "hn_wind" not in df[(df.mass == model.mass) & (df.Z == model.Z)].type:
                new_model = model.copy()
                new_model["type"] = "hn_wind"
                new_model[elements] = np.nan
                df = df.append(new_model, ignore_index=True)

    return df


def check_hn_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    If HNe models are included, there masses must be either; (i) a subset
    of the SNe models, or, (ii) intersect with the upper end of SNe masses.
    The chemical contribution from each mass above the minimum HN mass
    is a weighted average of the HN and SN models. If (ii), then then SNe
    masses will be extended to match the upper mass of HNe models, but added
    sn models added will be assumed to collapse directly to BH.
    E.g., (i) would be masses sn: [10, 15, 20, 30] and hn: [20, 30].
    (ii) would be masses sn: [10, 15, 20, 30] and hn: [20, 30, 40, 60],
    and the sn masses would be extended to [10, 15, 20, 30, 40, 60].
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
    # New SN models have same properties as corresponding HN model, but
    # ejecta mass will be zero. Setting ejecta composition should not be
    # necessary, but do so anyway to be safe.
    if mass_max_sn < np.max(mass_hn):
        extend_sn = pd.DataFrame(df[(df.type == "hn") & (df.mass > mass_max_sn)])
        extend_sn["mass_remnant"] = list(extend_sn.mass)
        extend_sn["mass_final"] = list(extend_sn.mass)
        extend_sn.loc[:, elements] = 0.0
        df = df.append(extend_sn, ignore_index=True)

    return df
