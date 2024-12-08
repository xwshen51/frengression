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

import math
import os
import sys

from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import diags

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# Suppress R output
@contextlib.contextmanager
def suppress_r_output():
    r_output = io.StringIO()
    with contextlib.redirect_stdout(r_output), contextlib.redirect_stderr(r_output):
        yield

def generate_data_causl(n=10000, nI = 3, nX= 1, nO = 1, nS = 1, ate = 2, beta_cov = 0, strength_instr = 3, strength_conf = 1, strength_outcome = 1, binary_intervention=True):
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

def dr_ate(x_tr,y_tr,z_tr, x_te, y_te, z_te, ps_model = "lr", or_model = "rf"):
    if ps_model == "lr":
        model = LogisticRegression(random_state=42)
        model.fit(z_tr, x_tr)
        hat_propen = model.predict_proba(z_te)[:, 1]

    if or_model == 'rf':
        model_mu0 = RandomForestRegressor(random_state=42)
        model_mu0.fit(z_tr[x_tr == 0], y_tr[x_tr == 0])

        model_mu1 =  RandomForestRegressor(random_state=42)
        model_mu1.fit(z_tr[x_tr == 1], y_tr[x_tr == 1])
    hat_mu0 = model_mu0.predict(z_te)
    hat_mu1 = model_mu1.predict(z_te)
    phi = x_te / hat_propen *(y_te - hat_mu1) + (1-x_te) / (1-hat_propen) *(y_te - hat_mu0) +(hat_mu1 - hat_mu0)
    hat_ate = np.mean(phi)
    hat_sd = np.std(phi)
    return hat_ate, hat_sd

def npcausal_ctseff(y,x,z,bw_seq):
    npcausal = importr('npcausal')
    with suppress_r_output():
        # Convert Python data to R
        y_r = robjects.FloatVector(y) if isinstance(y, np.ndarray) else pandas2ri.py2rpy(y)
        x_r = robjects.FloatVector(x) if isinstance(x, np.ndarray) else pandas2ri.py2rpy(x)
        z_r = pandas2ri.py2rpy(z) if isinstance(z, pd.DataFrame) else robjects.r['as.data.frame'](z)
        bw_seq_r = robjects.FloatVector(bw_seq)
        
        # Call the R ctseff function
        ctseff = robjects.r['ctseff']
        results = ctseff(y=y_r, a=x_r, x=z_r, bw_seq=bw_seq_r)
    # Convert R results back to Python (assuming results is a DataFrame-like structure)
    results_df = pandas2ri.rpy2py(results)
    return results_df

def npcausal_cdensity(y,x,z,kmax=4,gridlen=50,nsplits=5):
    npcausal = importr('npcausal')
    with suppress_r_output():
        # Convert Python data to R
        y_r = robjects.FloatVector(y) if isinstance(y, np.ndarray) else pandas2ri.py2rpy(y)
        x_r = robjects.FloatVector(x) if isinstance(x, np.ndarray) else pandas2ri.py2rpy(x)
        z_r = pandas2ri.py2rpy(z) if isinstance(z, pd.DataFrame) else robjects.r['as.data.frame'](z)
        bw_seq_r = robjects.FloatVector(bw_seq)
        
        # Call the R ctseff function
        cv.cdensity= robjects.r['cdensity']
        results = cdensity(y=y_r, a=x_r, x=z_r, kmax=kmax,gridlen=gridlen, nsplits=nsplits)
    # Convert R results back to Python (assuming results is a DataFrame-like structure)
    results_df = pandas2ri.rpy2py(results)
    return results_df

def rmse(hat_mu, mu):
    return np.sqrt(np.mean((hat_mu-mu)**2))

def mape(hat_mu, mu):
    return np.mean(np.abs((hat_mu-mu)/mu))

