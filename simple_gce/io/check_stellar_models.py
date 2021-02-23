"""Check stellar model data for completeness, inconsistencies, etc."""
from ..utils import chem_elements, error_handling
from ..gce import approx_agb
from .. import config
import warnings

#TODO:
# check mass limits provided in config. 

def check_initial(df):
    """
    Call the relevant checks after loading the initial data.

    Args:
        df: dataframe containing stellar models.

    Returns:

    """
    df = check_columns(df)
    df = check_model_types(df)

    return df

def check_columns(df):
    """
    Check that all of the necessary columns are present. Where possible, fill
    missing columns appropriately.
    """

    # TODO:
    # - Approximate and fill remnant_mass, mass_presn, if missing.

    el2z = chem_elements.el2z

    required_columns = ['mass', # ZAMS mass
                        'mass_final', # Final mass (e.g. presupernova, after wind-loss)
                        'type', # CC, AGB, Ia
                        'remnant_mass', # Mass of the compact remnant.
                        'Z', # Initial metallicity
                        'lifetime', # Stellar lifetime.
                        ]

    optional_columns = []

    non_req_columns = set(df.columns).difference(set(required_columns))
    non_req_opt_columns = non_req_columns.difference(set(optional_columns))    
    unknown_columns = non_req_opt_columns.difference(set(el2z.keys()))

    for col in required_columns:
        if not df[col][df[col].isna()].empty:
            if (df['type'][df[col].isna()] == 'Ia').all():
                if (col in ['mass','mass_final']):
                    df.loc[df[col].isna(), col] = float(config.STELLAR_MODELS['mass_co']) 
                elif col == 'remnant_mass':
                    df.loc[df[col].isna(), col] = 0.0 
            else:
                raise error_handling.InsufficientDataError(f"Missing some entries in stellar data for {col}.")

    if not set(required_columns).issubset(set(df.columns)):
        raise error_handling.InsufficientDataError("Stellar data is missing required columns.")
    elif len(unknown_columns) != 0:
        raise error_handling.UnknownCaseError("Unknown columns/elements in stellar data.")

    return df

def check_model_types(df):
    """
    """

    # Types may include core-collapse, asymptotic giant-branch, thermonuclear,
    # and hypernovae.
    known_types = ['CC','AGB','HN']

    types = df.type.unique()

    if not set(types).issubset(set(known_types)):
        unknown_types = list(set(types).difference(set(known_types)))
        raise error_handling.UnknownCaseError(f'Unknown model type/s : {unknown_types}')
        
    if 'CC' not in types:
        raise error_handling.InsufficientDataError('Stellar models must include core-collapse (type = "CC")')
    
    if 'AGB' not in types:
        mass_min_cc = config.STELLAR_MODELS['mass_min_cc']
        message = f"No AGB models found. Wind mass from stars with mass < {mass_min_cc} Msun will be approximated, and wind chemical composition will be unchanged from formation."
        warnings.warn(message)

        df = approx_agb.fill_agb(df)

    if 'HN' not in types:
        message = 'HNe are not included.'
        warnings.warn(message)

    return df
