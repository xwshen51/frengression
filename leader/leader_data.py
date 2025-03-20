import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri,numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


import io
import contextlib
import warnings

import math
import os
import sys

warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# Suppress R output
@contextlib.contextmanager
def suppress_r_output():
    r_output = io.StringIO()
    with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
        yield

def generate_bsl():
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r'leader_data.R')
        generate_bsl = robjects.globalenv['generate_bsl']
        r_dataframe = generate_bsl()
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    return df

def generate_outcome():
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r'leader_data.R')
        # generate_mace = robjects.globalenv['generate_mace']
        # r_dataframe_mace = generate_mace()

        # generate_mi = robjects.globalenv['generate_mi']
        # r_dataframe_mi = generate_mi()

        # generate_death = robjects.globalenv['generate_death']
        # r_dataframe_death = generate_death()
        generate_outcome = robjects.globalenv['generate_outcome']
        r_dataframe = generate_outcome()


    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # df_mace = robjects.conversion.rpy2py(r_dataframe_mace)
        # df_mi = robjects.conversion.rpy2py(r_dataframe_mi)
        # df_death = robjects.conversion.rpy2py(r_dataframe_death)
        df = robjects.conversion.rpy2py(r_dataframe)
    return df
    # return df_mace, df_mi, df_death

def generate_longi_cov():
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r'leader_data.R')
        generate_egfr = robjects.globalenv['generate_egfr']
        r_dataframe_egfr = generate_egfr()

        generate_hba1c = robjects.globalenv['generate_hba1c']
        r_dataframe_hba1c = generate_hba1c()

        generate_bmi = robjects.globalenv['generate_bmi']
        r_dataframe_bmi = generate_bmi()

    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_egfr = robjects.conversion.rpy2py(r_dataframe_egfr)
        df_hba1c = robjects.conversion.rpy2py(r_dataframe_hba1c)
        df_bmi = robjects.conversion.rpy2py(r_dataframe_bmi)

    return df_egfr, df_hba1c, df_bmi