import torch
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import expit
import urllib
import zipfile
from CausalEGM import *
#twins https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv


# Class reproducing the ACIC 2016 dataset
class ACIC2016(object):
    def __init__(self, setting=1, exp_num=1, one_hot_factors=True, device='cpu', dataset='train', path_data="../data", tensor=True, seed=0, train_size=0.6, val_size=0.2, scale=True):

        self.path_data = path_data
        self.one_hot_factors = one_hot_factors
        self.covariates = pd.read_csv(os.path.join(path_data,'ACIC2016/covariates.csv'))
        self.info = pd.read_csv(os.path.join(path_data,'ACIC2016/info.csv'))
        self.dgp_data = pd.read_csv(os.path.join(path_data,f'ACIC2016/dgp_data/setting{setting}_dataset{exp_num}.csv'))
        self.X_df = self._process_covariates(self.covariates) # turn factor variables into one-hot binary variables

        attrs = {}
        attrs['x'] = np.array(self.X_df)
        attrs['t'] = np.array(self.dgp_data['z']).reshape((-1, 1))
        attrs['y'] = np.array(self.dgp_data['y']).reshape((-1,1))
        attrs['y0'] = np.array(self.dgp_data['y.0']).reshape((-1, 1))
        attrs['y1'] = np.array(self.dgp_data['y.1']).reshape((-1, 1))
        attrs['mu0'] = np.array(self.dgp_data['mu.0']).reshape((-1, 1))
        attrs['mu1'] = np.array(self.dgp_data['mu.1']).reshape((-1, 1))
        attrs['cate_true'] = attrs['mu1'] - attrs['mu0']
        attrs['ps'] = np.array(self.dgp_data['e']).reshape((-1, 1))


        n_samples = self.X_df.shape[0]
        self.rng = np.random.default_rng(seed=seed)
        original_indices = self.rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        itr = original_indices[:n_train] # train set
        iva = original_indices[n_train:n_train + n_val] # val set
        itrva = original_indices[:n_train + n_val] # train+val set
        ite = original_indices[n_train + n_val:]  # test set

        # Check that each factor is in train, val and test sets
        for covariate_name in self.covariates.columns:
            if not pd.to_numeric(self.covariates[covariate_name], errors='coerce').notnull().all():
                factors_tr = pd.unique(self.covariates[covariate_name].iloc[itr])
                factors_va = pd.unique(self.covariates[covariate_name].iloc[iva])
                factors_te = pd.unique(self.covariates[covariate_name].iloc[ite])
                assert set(factors_va).issubset(set(factors_tr))
                assert set(factors_te).issubset(set(factors_tr))

        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva
        elif dataset == "train_val":
            original_indices = itrva
        else:
            original_indices = ite
        self.original_indices = original_indices

        # Find binary covariates
        self.binary = []
        for ind in range(attrs['x'].shape[1]):
            self.binary.append(len(np.unique(attrs['x'][:,ind])) == 2)
        self.binary = np.array(self.binary)

        # Normalise - continuous data
        self.scale = scale
        self.xm = np.zeros(self.binary.shape)
        self.xs = np.ones(self.binary.shape)
        if self.scale:
            self.xm[~self.binary] = np.mean(attrs['x'][itrva][:,~self.binary], axis=0)
            self.xs[~self.binary] = np.std(attrs['x'][itrva][:,~self.binary], axis=0)
        attrs['x'] -= self.xm
        attrs['x'] /= self.xs

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in attrs.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true', 'ps']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)




    def _process_covariates(self, covariates):
        covariates_done = {}
        for ind,covariate_name in enumerate(covariates.columns):
            if not 'x_' in covariate_name:
                continue
            if pd.to_numeric(covariates[covariate_name], errors='coerce').notnull().all():
                covariates_done[covariate_name] = covariates[covariate_name]
            else:
                if self.one_hot_factors:
                    for item in sorted(pd.unique(covariates[covariate_name])):
                       covariates_done[covariate_name + '_' + item] = (covariates[covariate_name] == item).astype(int)
                else:
                    covariates_done[covariate_name] = pd.Series([0]*len(covariates[covariate_name]))
                    for idx, item in sorted(enumerate(pd.unique(covariates[covariate_name]))):
                        covariates_done[covariate_name][covariates[covariate_name] == item] = idx

        return pd.DataFrame(covariates_done)

class News():

    def __init__(self, exp_num, dataset='train', tensor=True, device="cpu", train_size=0.6, val_size=0.2,
                 data_folder=None, scale=True, seed=0):

        if data_folder is None:
            data_folder = '../data'

        # Create data if it does not exist
        if not os.path.isdir(os.path.join(data_folder, 'News/numpy_dicts/')):
            self._create_data(data_folder)

        with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(exp_num)),
                  'rb') as file:
            data = pickle.load(file)
        data['cate_true'] = data['mu1'] - data['mu0']

        # Create and store indices
        x = data['x']
        n_samples = x.shape[0]
        rng = np.random.default_rng(seed=seed)
        original_indices = rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        n_val = int(val_size * n_samples)
        itr = original_indices[:n_train] # train set
        iva = original_indices[n_train:n_train + n_val] # val set
        idxtrain = original_indices[:n_train + n_val] # train + val set
        ite = original_indices[n_train + n_val:]  # test set

        if dataset == 'train':
            original_indices = itr
        elif dataset == 'val':
            original_indices = iva
        elif dataset == "train_val":
            original_indices = idxtrain
        else:
            original_indices = ite
        self.original_indices = original_indices

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in data.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    @staticmethod
    def _create_data(data_folder):

        print('News : no data, creating it')
        print('Downloading zipped csvs')
        urllib.request.urlretrieve('http://www.fredjo.com/files/NEWS_csv.zip', os.path.join(data_folder, 'News/csv.zip'))

        print('Unzipping csvs with sparse data')
        with zipfile.ZipFile(os.path.join(data_folder, 'News/csv.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_folder, 'News'))

        print('Densifying the sparse data')
        os.mkdir(os.path.join(data_folder, 'News/numpy_dicts/'))

        for f_index in range(1, 50 + 1):
            mat = pd.read_csv(os.path.join(data_folder,'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.x'.format(f_index)))
            n_rows, n_cols = int(mat.columns[0]), int(mat.columns[1])
            x = np.zeros((n_rows, n_cols)).astype(int)
            for i, j, val in zip(mat.iloc[:, 0], mat.iloc[:, 1], mat.iloc[:, 2]):
                x[i - 1, j - 1] = val
            data = {}
            data['x'] = x
            meta = pd.read_csv(
                os.path.join(data_folder, 'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.y'.format(f_index)),
                names=['t', 'y', 'y_cf', 'mu0', 'mu1'])
            for col in ['t', 'y', 'y_cf', 'mu0', 'mu1']:
                data[col] = np.array(meta[col]).reshape((-1, 1))
            with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(f_index)), 'wb') as file:
                pickle.dump(data, file)

        print('Done!')

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)

class IHDP(object):
    def __init__(
            self,
            path,
            trial,
            center=False,
            exclude_population=False
    ):
        self.trial = trial
        train_dataset = np.load(
            os.path.join(path, 'ihdp_npci_1-1000.train.npz')
            # os.path.join(path, 'ihdp_npci_1-100.train.npz')
        )
        test_dataset = np.load(
            os.path.join(path, 'ihdp_npci_1-1000.test.npz')
            # os.path.join(path, 'ihdp_npci_1-100.train.npz')
        )
        self.train_data = get_trial(
            dataset=train_dataset,
            trial=trial,
            training=True,
            exclude_population=exclude_population
        )
        # self.y_mean = self.train_data['y'].mean(dtype='float32')
        # self.y_std = self.train_data['y'].std(dtype='float32')
        self.test_data = get_trial(
            dataset=test_dataset,
            trial=trial,
            training=False,
            exclude_population=exclude_population
        )
        # self.dim_x_cont = self.train_data['x_cont'].shape[-1]
        # self.dim_x_bin = self.train_data['x_bin'].shape[-1]
        # self.dim_x = self.dim_x_cont + self.dim_x_bin

    def get_training_data(self):
        x, y, t = self.preprocess(self.train_data)
        mu0, mu1 = self.get_mu(test_set = False)
        examples_per_treatment = t.sum(0)
        return x, y, t, mu0, mu1, examples_per_treatment

    def get_test_data(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        x, y, t = self.preprocess(_data)
        examples_per_treatment = t.sum(0)
        mu1 = _data['mu1'].astype('float32')
        mu0 = _data['mu0'].astype('float32')
        #cate = mu1 - mu0
        return x, y, t, mu0, mu1, examples_per_treatment

    def get_subpop(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['ind_subpop']

    def get_t(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['t']

    def preprocess(self, dataset):
        x = np.hstack([dataset['x_cont'], dataset['x_bin']])
        #y = (dataset['y'].astype('float32') - self.y_mean) / self.y_std
        y = dataset['y_factual'].astype('float32')
        t = dataset['t'].astype('float32')
        return x, y, t
    def get_mu(self, test_set=True):
        _data = self.test_data if test_set else self.train_data
        return _data['mu0'], _data['mu1']

def get_arm_idx(t):
    return np.where(t==0), np.where(t==1)

def get_trial(
        dataset,
        trial,
        training=True,
        exclude_population=False
):
    bin_feats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    cont_feats = [i for i in range(25) if i not in bin_feats]
    ind_subpop = dataset['x'][:, bin_feats[2], trial].astype('bool')
    x = dataset['x'][:, :, trial]
    if exclude_population:
        x = np.delete(x, bin_feats[2], axis=-1)
        bin_feats.pop(2)
        if training:
            idx_included = np.where(ind_subpop)[0]
        else:
            idx_included = np.arange(dataset['x'].shape[0], dtype='int32')
    else:
        idx_included = np.arange(dataset['x'].shape[0], dtype='int32')
    x_bin = dataset['x'][:, bin_feats, trial][idx_included]
    # x_bin[:, 7] -= 1.
    t = dataset['t'][:, trial]
    # t_in = np.zeros((len(t), 2), 'float32')
    # t_in[:, 0] = 1 - t
    # t_in[:, 1] = t
    trial_data = {
        'x_bin': x_bin.astype('float32'),
        'x_cont': dataset['x'][:, cont_feats, trial][idx_included].astype('float32'),
        'y_factual': dataset['yf'][:, trial][idx_included],
        'treatment': t[idx_included],
        #'t': t_in[idx_included],
        'y_cfactual': dataset['ycf'][:, trial][idx_included],
        'mu0': dataset['mu0'][:, trial][idx_included],
        'mu1': dataset['mu1'][:, trial][idx_included],
        # 'ate': dataset['ate'],
        # 'yadd': dataset['yadd'],
        # 'ymul': dataset['ymul'],
        # 'ind_subpop': ind_subpop[idx_included]
    }

    return trial_data

def process_data(path='', trial=4):
    data = IHDP(path=path, trial = trial)

    x_combined = np.hstack([data.train_data['x_bin'], data.train_data['x_cont']])

    # Create column names for the x covariates
    # Assuming `x_bin` has 19 columns and `x_cont` has 6 columns
    x_bin_dim = data.train_data['x_bin'].shape[1]
    x_cont_dim = data.train_data['x_cont'].shape[1]
    x_columns = [f"X{i+1}" for i in range(x_bin_dim + x_cont_dim)]

    # Create a DataFrame for the combined x covariates
    df_x = pd.DataFrame(x_combined, columns=x_columns)

    # Now, extract the remaining arrays (y, t, etc.) and add them to the DataFrame
    df_other = pd.DataFrame({key: data.train_data[key] for key in data.train_data if key not in ['x_cont', 'x_bin']})

    # Concatenate the x DataFrame with the other columns
    df_train = pd.concat([df_x, df_other], axis=1)

    x_combined = np.hstack([data.test_data['x_bin'],data.test_data['x_cont']])

    # Create column names for the x covariates
    # Assuming `x_bin` has 19 columns and `x_cont` has 6 columns
    x_bin_dim = data.test_data['x_bin'].shape[1]
    x_cont_dim = data.test_data['x_cont'].shape[1]
    x_columns = [f"X{i+1}" for i in range(x_bin_dim + x_cont_dim)]

    # Create a DataFrame for the combined x covariates
    df_x = pd.DataFrame(x_combined, columns=x_columns)

    # Now, extract the remaining arrays (y, t, etc.) and add them to the DataFrame
    df_other = pd.DataFrame({key: data.test_data[key] for key in data.test_data if key not in ['x_cont', 'x_bin']})

    # Concatenate the x DataFrame with the other columns
    df_test = pd.concat([df_x, df_other], axis=1)

    # df = pd.concat([df_train, df_test],axis=0)
    return df_train, df_test


def Sim_Hirano_Imbens_adrf(x):
    return x + 2/(1+x)**3

def Sim_Sun_adrf(x):
    return x+0.5+np.exp(-0.5)

def Sim_Colangelo_adrf(x):
    return 1.2*x + (x**3)

def Semi_Twins_adrf(x):
    x_tr,y,z=Semi_Twins_sampler(path = 'data_causl').load_all()
    auxiliary_constant = np.mean(y+2 * 1/(1 + np.exp(-3 * x_tr)))
    return -2 * 1/(1 + np.exp(-3 * x)) + auxiliary_constant