"""
Load stellar models from CSV file
"""

import os
import pandas as pd
from .. import config
from . import check_stellar_models


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
        error_handing.ProgramError("Ia yields do not sum to one.")

    scale = 1 / check

    df = df * scale

    check2 = float(df.sum(axis = 'columns'))
    diff2 = abs(check - 1.0)
    if diff2 >= 1.e-12:
        error_handling.ProgramError("Unable to scale mass fractions.")

    return df