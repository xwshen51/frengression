import numpy as np
import pandas as pd
from scipy.special import expit
import scipy.stats as stats


import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri,numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
# Load the ranger package
ranger = importr("ranger")
npcausal = importr('npcausal')

import io
import contextlib
import warnings

import math
import os
import sys

from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import diags

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

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

def generate_data_survivl(n=10000, T=10, random_seed=1024):
    pandas2ri.activate()
    # Source the ./data.r script for data.causl dgp function
    with suppress_r_output():
        robjects.r['source'](r'data_causl/data_causl.R')
        generate_data = robjects.globalenv['data.survivl']
        r_dataframe = generate_data(n=n, T=T, random_seed=random_seed)
    # Use the localconverter context manager to convert the R dataframe to a Pandas DataFrame
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_dataframe)
    # Drop non-feature columns
    columns_to_drop = ['id', 'status', 'T']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    time_steps = T 
    # Extract baseline  covariates (s)
    s_cols = ['C']
    s = df[s_cols].values  # Shape: (n, s_dim)

    # Initialize lists to hold x, z, y for all time steps
    x_list = []
    z_list = []
    y_list = []

    for t in range(time_steps):
        x_col = f"X_{t}"
        z_col = f"Z_{t}"
        y_col = f"Y_{t}"

        if x_col not in df.columns or z_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Expected columns {x_col}, {z_col}, {y_col} not found in the dataframe.")

        x_list.append(df[x_col].values.reshape(-1, 1))  # Assuming x_dim=1
        z_list.append(df[z_col].values.reshape(-1, 1))  # Assuming z_dim=1
        y_list.append(df[y_col].values.reshape(-1, 1))  # Assuming y_dim=1

    # Concatenate along the second dimension to form [n, T * x_dim], etc.
    x_array = np.concatenate(x_list, axis=1)  # Shape: [n, T * x_dim]
    z_array = np.concatenate(z_list, axis=1)  # Shape: [n, T * z_dim]
    y_array = np.concatenate(y_list, axis=1)  # Shape: [n, T * y_dim]

    return s, x_array, z_array, y_array


def dr_ate(x_tr,y_tr,z_tr, x_te, y_te, z_te, ps_model = "lr", or_model = "rf",random_state = 42):
    if ps_model == "lr":
        model = LogisticRegression(random_state=42)
        model.fit(z_tr, x_tr)
        hat_propen = model.predict_proba(z_te)[:, 1]
    elif ps_model == "rf":
        model = RandomForestClassifier(random_state=42)
        model.fit(z_tr, x_tr)
        hat_propen = model.predict_proba(z_te)[:, 1]

    if or_model == "lr":
        model_mu0 = LinearRegression(random_state=random_state)
        model_mu1 =  LinearRegression(random_state=random_state)
    elif or_model == 'rf':
        model_mu0 = RandomForestRegressor(random_state=random_state)
        model_mu1 =  RandomForestRegressor(random_state=random_state)
    model_mu0.fit(z_tr[x_tr == 0], y_tr[x_tr == 0])
    model_mu1.fit(z_tr[x_tr == 1], y_tr[x_tr == 1])

    hat_mu0 = model_mu0.predict(z_te)
    hat_mu1 = model_mu1.predict(z_te)
    phi = x_te / hat_propen *(y_te - hat_mu1) + (1-x_te) / (1-hat_propen) *(y_te - hat_mu0) +(hat_mu1 - hat_mu0)
    hat_ate = np.mean(phi)
    hat_sd = np.std(phi)
    return hat_ate, hat_sd

def cross_fit_dr_ate(df, p, k_folds=5, ps_model="lr", or_model="rf", random_state=42):
    """
    Perform cross-fitting for doubly robust ATE estimation.

    Parameters:
    - df: DataFrame containing the data.
    - p: Number of covariates in the dataset.
    - k_folds: Number of folds for cross-fitting (default: 5).
    - ps_model: Propensity score model ('lr' for Logistic Regression, 'rf' for Random Forest).
    - or_model: Outcome regression model ('lr' for Linear Regression, 'rf' for Random Forest).
    - random_state: Random state for reproducibility.

    Returns:
    - results: Dictionary with keys 'ATE' and 'SD' containing the estimated ATE and standard deviation.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    phi_values = []

    for train_idx, test_idx in kf.split(df):
        # Split DataFrame into training and test sets
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Extract x, y, z from the training and test sets
        z_tr = df_train[[f"X{i}" for i in range(1, p + 1)]].values
        x_tr = df_train['A'].values
        y_tr = df_train['y'].values

        z_te = df_test[[f"X{i}" for i in range(1, p + 1)]].values
        x_te = df_test['A'].values
        y_te = df_test['y'].values

        # Train propensity score model
        if ps_model == "lr":
            ps_model_instance = LogisticRegression(random_state=random_state)
        elif ps_model == "rf":
            ps_model_instance = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError("Invalid ps_model. Choose 'lr' or 'rf'.")

        ps_model_instance.fit(z_tr, x_tr)
        hat_propen = ps_model_instance.predict_proba(z_te)[:, 1]

        # Train outcome regression models
        if or_model == "lr":
            or_model_mu0 = LinearRegression()
            or_model_mu1 = LinearRegression()
        elif or_model == "rf":
            or_model_mu0 = RandomForestRegressor(random_state=random_state)
            or_model_mu1 = RandomForestRegressor(random_state=random_state)
        else:
            raise ValueError("Invalid or_model. Choose 'lr' or 'rf'.")

        or_model_mu0.fit(z_tr[x_tr == 0], y_tr[x_tr == 0])
        or_model_mu1.fit(z_tr[x_tr == 1], y_tr[x_tr == 1])

        hat_mu0 = or_model_mu0.predict(z_te)
        hat_mu1 = or_model_mu1.predict(z_te)

        # Calculate influence function
        phi = (
            x_te / hat_propen * (y_te - hat_mu1) +
            (1 - x_te) / (1 - hat_propen) * (y_te - hat_mu0) +
            (hat_mu1 - hat_mu0)
        )
        phi_values.append(phi)

    # Combine results across folds
    phi_values = np.concatenate(phi_values)
    hat_ate = np.mean(phi_values)
    hat_sd = np.std(phi_values)

    return {
        'ATE': hat_ate,
        'SD': hat_sd
    }



def npcausal_ctseff(y,x,z,bw_seq=np.linspace(0.2, 2, 100)):
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


def npcausal_cdensity(y, x, z, y_grid, kmax=10, gridlen=100, nsplits=2):
    """
    Python wrapper for R's cdensity function with input validation and debugging.

    Parameters:
    - y: Outcome (1D array).
    - x: Treatment (1D array).
    - z: Covariates (2D array).
    - y_grid: Grid values for y.
    - kmax: Maximum number of splits.
    - gridlen: Grid size.
    - nsplits: Number of splits.

    Returns:
    - res: Output from cdensity.
    """
    # Input validation
    cdensity = robjects.r["cdensity"]

    assert np.all(np.isfinite(y)), "y contains non-finite values!"
    assert np.all(np.isfinite(x)), "x contains non-finite values!"
    assert np.all(np.isfinite(z)), "z contains non-finite values!"
    assert len(y) == len(x) == z.shape[0], "Mismatched dimensions for y, x, z!"

    # Conversion to R
    try:
        y_r = robjects.FloatVector(y)  # Convert y to R vector
        x_r = robjects.FloatVector(x)  # Convert x to R vector
        z_r = robjects.r['as.matrix'](numpy2ri.py2rpy(z))  # Convert z to R matrix
        y_grid_r = robjects.FloatVector(y_grid)
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {e}")

    # Validate R-side objects
    print("R y:", robjects.r['str'](y_r))
    print("R x:", robjects.r['str'](x_r))
    print("R z:", robjects.r['str'](z_r))

    # Call the R cdensity function
    cdensity = robjects.r['cdensity']
    try:
        res = cdensity(y=y_r, a=x_r, x=z_r, kmax=kmax, gridlen=gridlen, nsplits=nsplits)
    except Exception as e:
        raise RuntimeError(f"Error in cdensity: {e}")

    # Extract g1 and g0
    g1_function = res.rx2('g1')
    g0_function = res.rx2('g0')
    g1_res=g1_function(kmax,y_grid_r)
    g0_res=g0_function(kmax,y_grid_r)
    numpy2ri.activate()

    return np.array(g1_res),np.array(g0_res)

def rmse(hat_mu, mu):
    return np.sqrt(np.mean((hat_mu-mu)**2))

def mape(hat_mu, mu):
    return np.mean(np.abs((hat_mu-mu)/mu))

