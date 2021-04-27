"""Check stellar model data for completeness, inconsistencies, etc."""
from ..utils import chem_elements, error_handling
from ..gce import approx_agb, approx_lifetime
from .. import config
import warnings

#TODO:
# check mass limits provided in config. 
#def check_winds(df):
#   models with wind and CC, check consistency of mass / mass_final 
#   models with CC and no wind, check mass == mass_final

# def check_ia(df):
#   check consistency with config params

REQUIRED_COLUMNS = ['mass', # ZAMS mass
                    'mass_final', # Final mass (e.g. presupernova, after wind-loss)
                    'type', # CC, AGB, Ia
                    'remnant_mass', # Mass of the compact remnant.
                    'Z', # Initial metallicity
                    'lifetime', # Stellar lifetime.
                    ]

OPTIONAL_COLUMNS = ['expl_energy']

def check_initial(df):
    """
    Call the relevant checks after loading the initial data.

    Args:
        df: dataframe containing stellar models.

    Returns:

    """
    df = check_model_types(df)
    check_columns(df)
    df = check_missing_vals(df)
    check_mass_metallicity_consistent(df)

    return df

def check_columns(df):
    """
    Check that all of the necessary columns are present. Where possible, fill
    missing columns appropriately.
    """

    # TODO:
    # - Approximate and fill remnant_mass, mass_presn, if missing.

    el2z = chem_elements.el2z

    # Columns excluding required
    non_req_columns = set(df.columns).difference(set(REQUIRED_COLUMNS))
    # Columns excluding required and optional 
    remaining_columns = non_req_columns.difference(set(OPTIONAL_COLUMNS))    
    # Columns excluding required, optional, and chemical elements
    unknown_columns = remaining_columns.difference(set(el2z.keys()))

    if not set(REQUIRED_COLUMNS).issubset(set(df.columns)):
        raise error_handling.InsufficientDataError("Stellar data is missing required columns.")
    elif len(unknown_columns) != 0:
        raise error_handling.UnknownCaseError(f"Unknown columns/elements in stellar data.\n{unknown_columns}")



def check_missing_vals(df):
    """
    Check that all required data is present.
    """

    if not df[df['mass_final'].isna()].empty:
        raise error_handling.InsufficientDataError(f"Missing some entries for mass_final.\
                        If there are no winds/mass loss in a model, set mass_final = mass.")
    if not df[df['remnant_mass'].isna()].empty:
        raise error_handling.InsufficientDataError(f"Missing some entries for remnant_mass. \
                        If there is no remnant (e.g. SNe Ia), set this column to zero. \
                        For AGB stars, set mass_remnant = mass_final")
    if not df[df['Z'].isna()].empty:
        raise error_handling.InsufficientDataError(f"Missing some entries for Z (metallicity). \
                        If metallicity is not relevant, or you have models with a single value \
                        of metallicity, set this column to zero for those models.")
    if not df[df['lifetime'].isna()].empty:
        warnings.warn('Missing stellar lifetime data. Filling with approx values.')
        df = approx_lifetime.fill_lifetimes(df) 

    return df

def check_model_types(df):
    """
    """

    # Types may include core-collapse, asymptotic giant-branch, thermonuclear,
    # and hypernovae.
    known_types = ['cc','agb','hn','wind']

    types = df.type.unique()

    if not set(types).issubset(set(known_types)):
        unknown_types = list(set(types).difference(set(known_types)))
        raise error_handling.UnknownCaseError(f'Unknown model type/s : {unknown_types}')
        
    if 'cc' not in types:
        raise error_handling.InsufficientDataError('Stellar models must include core-collapse (type = "CC")')
    
    if 'agb' not in types:
        mass_min_cc = config.STELLAR_MODELS['mass_min_cc']
        message = f"No AGB models found. Wind mass from stars with mass < {mass_min_cc} \
             Msun will be approximated, and wind chemical composition will be unchanged from formation."
        warnings.warn(message)

        df = approx_agb.fill_agb(df)

    if 'hn' not in types:
        message = 'HNe are not included.'
        warnings.warn(message)

    return df

def check_massfracs(df):
    """
    """
    elements = chem_elements.elements
    elements = list(set(self.df.columns).intersection(set(elements)))
    
    check = df[elements].sum(axis = 1)
    diff = abs(check - 1.0)

    if diff.max() >= 1.e-3:
        raise error_handling.ProgramError("Large errors in sum(mass fractions) for stellar models")

    scale = 1 / check

    df.loc[:,elements] = df[elements].mul(scale, axis = 'rows')

    check2 = abs(df[elements].sum(axis = 1) - 1.0)
    if check2.max() >= 1.e-12:
        raise error_handling.ProgramError("Unable to scale mass fractions.")

def check_mass_metallicity_consistent(df):
    """
    """
    z_list = df.Z.unique()
    m_list = df.mass.unique()

    for m in m_list:
        z_ = df[df.mass == m].Z.unique()
        if set(z_) != set(z_list):
            raise error_handling.NotImplementedError()