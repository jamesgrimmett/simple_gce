"""
Load stellar models from CSV file
"""

import os
import pandas as pd
from .. import config
from . import check_stellar_models


def read_csv(test = False):
    if test == False:
        filepath = config.FILEPATHS['stellar_models']
    else:
        filepath = config.FILEPATHS['test_stellar_models']
    df = pd.DataFrame()

    for fp in filepath:
        df_ = pd.read_csv(fp)
        df = pd.concat((df,df_))

    check_stellar_models.check_initial(df) 

    return df

# core collapse
# winds
# AGB
# Ia
# mass_initial, mass_presn, mass_core, Z, 