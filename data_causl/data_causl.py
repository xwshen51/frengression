import numpy as np
import pandas as pd
from scipy.special import expit
import scipy.stats as stats
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import io
import contextlib
import warnings
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")


def generate_data_causl(n=10000, nI = 3, nX= 1, nO = 1, nS = 1, ate = 2, beta_cov = 0, strength_instr = 3, strength_conf = 1, strength_outcome = 1, binary_intervention=True):
    # Function to suppress R output
    @contextlib.contextmanager
    def suppress_r_output():
        r_output = io.StringIO()
        with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
            yield
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r'data_causl/data_causl.R')
        generate_data = robjects.globalenv['data.causl']
        r_dataframe = generate_data(n=n, nI=nI, nX=nX, nO=nO, nS=nS, ate=ate, beta_cov=beta_cov, strength_instr=strength_instr, strength_conf=strength_conf, strength_outcome=strength_outcome, binary_intervention=binary_intervention)
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    return df
