import torch
from engression.models import StoNet
from engression.loss_func import energy_loss_two_sample
import numpy as np
import copy
sigmoid = torch.nn.Sigmoid()

# cross-fitting
from sklearn.model_selection import KFold
def cross_fit_frengression(df, binary_intervention, p, outcome_reg=True, k_folds=5, num_iters=1000, lr=1e-4, sample_size=1000):
    """
    Perform cross-fitting for the Frengression model.

    Parameters:
    - df: DataFrame containing the data.
    - binary_intervention: Boolean indicating if the intervention variable is binary.
    - p: Number of covariates in the dataset.
    - k_folds: Number of folds for cross-fitting (default: 5).
    - outcome_reg: Whether to use conditional outcome regression (True) or marginal outcome (False).
    - num_iters: Number of iterations for training the model.
    - lr: Learning rate for training.
    - n_p: Sample size for predictions.

    Returns:
    - predictions: Dictionary with keys 'P0', 'P1', and 'ATE' containing the predictions.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    predictions_P0 = np.zeros(len(df), dtype=float)
    predictions_P1 = np.zeros(len(df), dtype=float)
    ate_marginal = 0

    for train_idx, test_idx in kf.split(df):
        # Split DataFrame into training and test sets
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Extract x, y, z from the training and test sets
        z_tr = torch.tensor(df_train[[f"X{i}" for i in range(1, p + 1)]].values, dtype=torch.float32)
        x_tr= torch.tensor(df_train['A'].values, dtype=torch.int32).view(-1, 1) if binary_intervention else \
                  torch.tensor(df_train['A'].values, dtype=torch.float32).view(-1, 1)
        y_tr = torch.tensor(df_train['y'].values, dtype=torch.float32).view(-1, 1)

        z_te = torch.tensor(df_test[[f"X{i}" for i in range(1, p + 1)]].values, dtype=torch.float32)
        x_te = torch.tensor(df_test['A'].values, dtype=torch.int32).view(-1, 1) if binary_intervention else \
                 torch.tensor(df_test['A'].values, dtype=torch.float32).view(-1, 1)

        # Initialize and train the Frengression model
        model = Frengression(x_dim = x_tr.shape[1], y_dim = 1, z_dim =z_tr.shape[1], 
                             noise_dim=1, num_layer=3, hidden_dim=100, 
                             device = torch.device('cpu'), x_binary=binary_intervention, z_binary_dims=0)

        model.train_xz(x_tr, z_tr, num_iters=num_iters, lr=lr, print_every_iter=400)

        model.train_y(x_tr, z_tr, y_tr, num_iters=num_iters, lr=lr, print_every_iter=400)

        # Prepare input for predictions
        x0 = torch.zeros(z_te.shape[0], dtype=torch.int32).reshape(-1, 1) if binary_intervention else \
            torch.zeros(z_te.shape[0], dtype=torch.float32).reshape(-1, 1)
        x1 = torch.ones(z_te.shape[0], dtype=torch.int32).reshape(-1, 1) if binary_intervention else \
            torch.ones(z_te.shape[0], dtype=torch.float32).reshape(-1, 1)

        xz0 = torch.cat([x0, z_te], dim=1)
        xz1 = torch.cat([x1, z_te], dim=1)

        # Predict conditional distributions
        P0 = model.predict_conditional(x0, xz0, sample_size=sample_size).numpy().reshape(-1, 1)
        P1 = model.predict_conditional(x1, xz1, sample_size=sample_size).numpy().reshape(-1, 1)
        # Store predictions
        predictions_P0[test_idx] = P0.mean(axis=1)
        predictions_P1[test_idx] = P1.mean(axis=1)


        if outcome_reg == False:
            P0_marginal = model.sample_causal_margin(torch.tensor([0], dtype=torch.int32), sample_size=sample_size).numpy().reshape(-1, 1)
            P1_marginal = model.sample_causal_margin(torch.tensor([1], dtype=torch.int32), sample_size=sample_size).numpy().reshape(-1, 1)
            ate_marginal += np.mean(P1_marginal) - np.mean(P0_marginal)

    # Calculate ATE
    ate_predictions = predictions_P1.mean() - predictions_P0.mean()

    result = {
        'P0': predictions_P0,
        'P1': predictions_P1,
        'ATE': ate_predictions
    }
    if outcome_reg == False:
        result['ATE_marginal'] = ate_marginal / k_folds
    return result

# Example usage:
# Assuming `FrengressionModel` is the Frengression model class
# result = cross_fit_frengression(FrengressionModel, df, binary_intervention=True, p=10, k_folds=5, num_iters=1000, lr=1e-4, n_p=100)

class Frengression(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, 
                 #z_binary=False,
                 z_binary_dims = 0, # number of binary confounders.
                 y_binary=False,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        # self.z_binary = z_binary
        self.z_binary_dims = z_binary_dims
        self.y_binary = y_binary
        self.device = device
        self.model_xz = StoNet(0, x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim, noise_dim), add_bn=False, noise_all_layer=False).to(device)
        out_act = 'sigmoid' if y_binary else None
        self.model_y = StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
        self.model_eta = StoNet(x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False).to(device)
        
    def train_xz(self, x, z, num_iters=100, lr=1e-3, print_every_iter=10):
        self.model_xz.train()
        self.optimizer_xz = torch.optim.Adam(self.model_xz.parameters(), lr=lr)
        xz = torch.cat([x, z], dim=1)
        xz = xz.to(self.device)
        for i in range(num_iters):
            self.optimizer_xz.zero_grad()
            sample1 = self.model_xz(x.size(0))
            sample2 = self.model_xz(x.size(0))
            if self.x_binary:
                sample1[:, :self.x_dim] = sigmoid(sample1[:, :self.x_dim])
                sample2[:, :self.x_dim] = sigmoid(sample2[:, :self.x_dim])
            if self.z_binary_dims > 0:
                sample1[:, self.x_dim:(self.x_dim+self.z_binary_dims)] = sigmoid(sample1[:, self.x_dim:(self.x_dim+self.z_binary_dims)])
                sample2[:, self.x_dim:(self.x_dim+self.z_binary_dims)] = sigmoid(sample2[:, self.x_dim:(self.x_dim+self.z_binary_dims)])
            loss, loss1, loss2 = energy_loss_two_sample(xz, sample1, sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')
    
    def train_y(self, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        self.model_y.train()
        self.model_eta.train()
        self.optimizer_y = torch.optim.Adam(list(self.model_y.parameters()) + list(self.model_eta.parameters()), lr=lr)
        x = x.to(self.device)
        y = y.to(self.device)
        xz = torch.cat([x, z], dim=1)
        xz = xz.to(self.device)
        for i in range(num_iters):
            self.optimizer_y.zero_grad()
            eta1 = self.model_eta(xz)
            eta2 = self.model_eta(xz)
            y_sample1 = self.model_y(torch.cat([x, eta1], dim=1))
            y_sample2 = self.model_y(torch.cat([x, eta2], dim=1))
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y, y_sample1, y_sample2)
            eta_true = torch.randn(y.size(), device=self.device)
            eta1 = self.model_eta(xz)
            eta2 = self.model_eta(xz[torch.randperm(x.size(0))])
            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1, eta2)
            loss = loss_y + loss_eta
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}')
    
    @torch.no_grad()
    def predict_causal(self, x, target="mean", sample_size=100):
        self.eval()
        x = x.to(self.device)
        return self.model_y.predict(x, target, sample_size)

        
    @torch.no_grad()
    def sample_joint(self, sample_size=100):
        self.eval()
        xz = self.model_xz(sample_size)
        x = xz[:, :self.x_dim]
        z = xz[:, self.x_dim:]
        if self.x_binary:
            x = (x > 0).float()
        if self.z_binary_dims>0:
            z[:, :self.z_binary_dims] = (z[:, :self.z_binary_dims] > 0).float()
        xz = torch.cat([x, z], dim=1)
        eta = self.model_eta(xz)
        y = self.model_y(torch.cat([x, eta], dim=1))
        return x, y, z

    @torch.no_grad()
    def sample_causal_margin(self, x, sample_size=100):
        self.eval()
        y = self.model_y.sample(x, sample_size = sample_size)
        return y
    
    @torch.no_grad()
    def predict_conditional(self, x, xz, sample_size=100):
        self.eval()
        xz = xz.to(self.device)
        eta = self.model_eta(xz)
        y = self.model_y.predict(torch.cat([x, eta], dim=1), sample_size = sample_size)
        return y
    

    def specify_causal(self, causal_margin):
        def causal_margin1(x_eta):
            x = x_eta[:, :self.x_dim]
            eta = x_eta[:, self.x_dim:]
            return causal_margin(x, eta)
        self.model_y = causal_margin1
    
    def reset_y_models(self):
        self.model_y = StoNet(self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)
        self.model_eta = StoNet(self.x_dim + self.z_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)


class FrengressionSeq(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, T, s_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, z_binary=False, y_binary=False,
                 s_in_predict=True,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.T = T
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        self.z_binary = z_binary
        self.y_binary = y_binary
        self.s_in_predict = s_in_predict
        self.device = device
        # for xz:
        # generate x0z0
        self.model_xz = [
            StoNet(s_dim, x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]
        #generate x1z1 to xTzT
        for t in range(T-1):
            self.model_xz.append(
                StoNet(s_dim + (x_dim + z_dim + y_dim) * (t + 1), x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim+y_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
            )
        
        out_act = 'sigmoid' if y_binary else None
       
        # for y
        # generate y0
        if self.s_in_predict:
            self.model_y = [
                StoNet(s_dim + x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(s_dim + x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                    # StoNet((x_dim+y_dim) * (t+1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
                )
        else:
            self.model_y = [
                StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                )
        
        # for eta:
        self.model_eta = [
            StoNet(s_dim + x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]

        for t in range(1,T):
            self.model_eta.append(
                StoNet(s_dim+(x_dim + z_dim)*(t + 1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            )
        
    
    def sample_xz(self, s=None, x=None, z=None, y = None):
        xz = self.model_xz[0](s)
        x0 = xz[:, :self.x_dim]
        z0 = xz[:, self.x_dim:]
        if self.x_binary:
            x0 = (x0 > 0).float()
        if self.z_binary:
            z0 = (z0 > 0).float()
        x_all = [x0]
        z_all = [z0]
        for t in range(1,self.T):
            sxzy_p = torch.cat([s, x[:,:(t*self.x_dim)], z[:,:(t*self.z_dim)], y[:, :(t*self.y_dim)]], dim=1)
            xz = self.model_xz[t](sxzy_p)
            xt = xz[:, :self.x_dim]
            zt = xz[:, self.x_dim:]
            if self.x_binary:
                xt = (xt > 0).float()
                zt = (zt > 0).float()
            x_all.append(xt)
            z_all.append(zt)
        return torch.cat(x_all, dim=1), torch.cat(z_all, dim=1)

    def sample_eta(self, s = None, x=None, z=None):
        eta_all = []
        for t in range(self.T):
            sxz_p = torch.cat([s, x[:, :((t+1)*self.x_dim)], z[:,:((t+1)*self.z_dim)]], dim=1)
            etat = self.model_eta[t](sxz_p)
            eta_all.append(etat)
        return torch.cat(eta_all, dim=1)
    
    def sample_y(self, s = None, x=None, eta=None):
        y_all = []
        for t in range(self.T):
            if self.s_in_predict:
                sxeta_p = torch.cat([s, x[:,:((t+1)*self.x_dim)], eta[:, (t*self.y_dim):((t+1)*self.y_dim)]], dim=1)
                yt = self.model_y[t](sxeta_p)
            else:
                xeta_p = torch.cat([x[:,:((t+1)*self.x_dim)], eta[:, (t*self.y_dim):((t+1)*self.y_dim)]], dim=1)
                yt = self.model_y[t](xeta_p)
            y_all.append(yt)
        return torch.cat(y_all, dim=1)

        
    def train_xz(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        for model in self.model_xz:
            model.train()
        all_parameters = []
        for t in range(self.T):
            all_parameters += list(self.model_xz[t].parameters())
        self.optimizer_xz = torch.optim.Adam(all_parameters, lr=lr)
        xz = torch.cat([x, z], dim=1)
        for i in range(num_iters):
            self.optimizer_xz.zero_grad()
            sample1_x, sample1_z = self.sample_xz(s=s, x=x, z=z,y=y)
            sample1 = torch.cat([sample1_x, sample1_z], dim=1)
            sample2_x, sample2_z = self.sample_xz(s=s, x=x, z=z,y=y)
            sample2 = torch.cat([sample2_x, sample2_z], dim=1)
            if self.x_binary:
                sample1[:, :self.x_dim] = sigmoid(sample1[:, :self.x_dim])
                sample2[:, :self.x_dim] = sigmoid(sample2[:, :self.x_dim])
            if self.z_binary:
                sample1[:, self.x_dim:] = sigmoid(sample1[:, self.x_dim:])
                sample2[:, self.x_dim:] = sigmoid(sample2[:, self.x_dim:])
            loss, loss1, loss2 = energy_loss_two_sample(xz, sample1, sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')
    
    def train_y(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        all_parameters = []
        for t in range(self.T):
            self.model_y[t].train()
            self.model_eta[t].train()
            all_parameters += list(self.model_y[t].parameters())+ list(self.model_eta[t].parameters())

        self.optimizer_y = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)
        for i in range(num_iters):
            self.optimizer_y.zero_grad()
            eta1 = self.sample_eta(s=s, x=x,z=z)
            eta2 = self.sample_eta(s=s, x=x,z=z)
            y_sample1 = self.sample_y(s=s,x=x, eta = eta1)
            y_sample2 = self.sample_y(s=s,x=x, eta = eta2)
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y, y_sample1, y_sample2)
            
            eta_true = torch.randn(y.size(), device=self.device)
            eta1 = self.sample_eta(s=s, x=x, z=z)
            perm = torch.randperm(x.size(0))
            eta2 = self.sample_eta(s=s[perm], x=x[perm],z=z[perm])
            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1, eta2)
            loss = loss_y + loss_eta
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}')
    
    @torch.no_grad()
    def predict_causal(self, s, x, target="mean", sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = self.model_y[t].predict(torch.cat([s,x[:,:(t+1)*self.x_dim]],dim=1), target, sample_size)
            else:
                yt = self.model_y[t].predict(x[:,:(t+1)*self.x_dim], target, sample_size)
            all_y.append(yt)
        return all_y
    
    @torch.no_grad()
    def sample_causal_margin(self,s, x, sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = self.model_y[t].sample(torch.cat([s,x[:,:(t+1)*self.x_dim]], dim=1), sample_size = sample_size)
            else:
                yt = self.model_y[t].sample(x[:,:(t+1)*self.x_dim], sample_size = sample_size)
            all_y.append(yt)
        return all_y



class FrengressionSurv(torch.nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, T, s_dim,
                 num_layer=3, hidden_dim=100, noise_dim=10,
                 x_binary=False, z_binary=False, y_binary=True,
                 s_in_predict = True,
                 device=torch.device('cuda')):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.T = T
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.x_binary = x_binary
        self.z_binary = z_binary
        self.y_binary = y_binary # Default setting of survivl data : binary outcome
        self.device = device
        self.s_in_predict=s_in_predict
        # for xz:
        # generate x0z0
        self.model_xz = [
            StoNet(s_dim, x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]
        #generate x1z1 to xTzT
        for t in range(T-1):
            self.model_xz.append(
                StoNet(s_dim + (x_dim + z_dim + y_dim) * (t + 1), x_dim + z_dim, num_layer, hidden_dim, max(x_dim + z_dim+y_dim, noise_dim), add_bn=False, noise_all_layer=False,verbose=False).to(device)
            )
        
        out_act = 'sigmoid' if y_binary else None
       
        # for y
        # generate y0
        if self.s_in_predict:
            self.model_y = [
                StoNet(s_dim + x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(s_dim + x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                    # StoNet((x_dim+y_dim) * (t+1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act).to(device)
                )
        else:
            self.model_y = [
                StoNet(x_dim + y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
            ]
            # generate y1 onwards
            for t in range(1,T):
                self.model_y.append(
                    StoNet(x_dim * (t+1)+ y_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, out_act=out_act,verbose=False).to(device)
                )
        
        # for eta:
        self.model_eta = [
            StoNet(s_dim + x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False,verbose=False).to(device)
        ]

        for t in range(1,T):
            self.model_eta.append(
                StoNet(s_dim+(x_dim + z_dim)*(t + 1), y_dim, num_layer, hidden_dim, noise_dim, add_bn=False, noise_all_layer=False, verbose=False).to(device)
            )
        
    
    def sample_xz(self, s=None, x=None, z=None, y = None):
        xz = self.model_xz[0](s)
        x0 = xz[:, :self.x_dim]
        z0 = xz[:, self.x_dim:]
        if self.x_binary:
            x0 = (x0 > 0).float()
        if self.z_binary:
            z0 = (z0 > 0).float()
        x_all = [x0]
        z_all = [z0]
        for t in range(1,self.T):
            sxzy_p = torch.cat([s, x[:,:(t*self.x_dim)], z[:,:(t*self.z_dim)], y[:, :(t*self.y_dim)]], dim=1)
            xz = self.model_xz[t](sxzy_p)
            xt = xz[:, :self.x_dim]
            zt = xz[:, self.x_dim:]
            if self.x_binary:
                xt = (xt > 0).float()
                zt = (zt > 0).float()
            x_all.append(xt)
            z_all.append(zt)
        

        return torch.cat(x_all, dim=1), torch.cat(z_all, dim=1)

        
    def train_xz(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        for model in self.model_xz:
            model.train()
        all_parameters = []
        for t in range(self.T):
            all_parameters += list(self.model_xz[t].parameters())
        self.optimizer_xz = torch.optim.Adam(all_parameters, lr=lr)
        xz = torch.cat([x, z], dim=1)
        for i in range(num_iters):
            self.optimizer_xz.zero_grad()
            sample1_x, sample1_z = self.sample_xz(s=s, x=x, z=z,y=y)
            sample1 = torch.cat([sample1_x, sample1_z], dim=1)
            sample2_x, sample2_z = self.sample_xz(s=s, x=x, z=z,y=y)
            sample2 = torch.cat([sample2_x, sample2_z], dim=1)
            if self.x_binary:
                sample1[:, :self.x_dim] = sigmoid(sample1[:, :self.x_dim])
                sample2[:, :self.x_dim] = sigmoid(sample2[:, :self.x_dim])
            if self.z_binary:
                sample1[:, self.x_dim:] = sigmoid(sample1[:, self.x_dim:])
                sample2[:, self.x_dim:] = sigmoid(sample2[:, self.x_dim:])
            loss, loss1, loss2 = energy_loss_two_sample(xz, sample1, sample2)
            loss.backward()
            self.optimizer_xz.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f}, loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}')
    
    def train_y(self, s, x, z, y, num_iters=100, lr=1e-3, print_every_iter=10):
        all_parameters = []
        for t in range(self.T):
            self.model_y[t].train()
            self.model_eta[t].train()
            all_parameters += list(self.model_y[t].parameters())+ list(self.model_eta[t].parameters())

        self.optimizer_y = torch.optim.Adam(all_parameters, lr=lr)
        s = s.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        n = x.shape[0]
        event_indicator = (y>0).float()
        c = torch.cumsum(event_indicator, dim=1)
        c_shifted = torch.zeros_like(c)
        c_shifted[:, 1:] = c[:, :-1]
        mask = (c_shifted > 0)
        y_masked = y.clone()
        y_masked[mask] = -1
        
        for i in range(num_iters):
            y_list = [y[:,:self.y_dim]]
            x_list = [x[:,:self.x_dim]]
            s_list = [s[:,:self.s_dim]]
            z_list = [z[:,:self.z_dim]]

            # resample from data
            for t in range(1, self.T):
                valid_idx = (y_masked[:,t] >=-0.5).nonzero(as_tuple=True)[0]
                sample_idx = valid_idx[torch.randint(0, len(valid_idx), (n,))]
                y_list.append(y[sample_idx, t*self.y_dim:((t+1)*self.y_dim)])
                x_list.append(x[sample_idx, :((t+1)*self.x_dim)])
                z_list.append(z[sample_idx, :((t+1)*self.z_dim)])
                s_list.append(s[sample_idx, :self.s_dim])
            y_sample = torch.cat(y_list, dim=1)

            self.optimizer_y.zero_grad()
            y_sample1 = []
            y_sample2 = []
            for t in range(self.T):
                sxz_p = torch.cat([s_list[t], x_list[t], z_list[t]], dim=1)
                etat1 = self.model_eta[t](sxz_p)
                etat2 = self.model_eta[t](sxz_p)
                if self.s_in_predict:
                    sxeta_p1 = torch.cat([s_list[t], x_list[t], etat1], dim=1)
                    yt1 = self.model_y[t](sxeta_p1)
                    sxeta_p2 = torch.cat([s_list[t], x_list[t], etat2], dim=1)
                    yt2 = self.model_y[t](sxeta_p2)
                else:
                    xeta_p1 = torch.cat([x_list[t], etat1], dim=1)
                    xeta_p2 = torch.cat([x_list[t], etat2], dim=1)
                    yt1 = self.model_y[t](xeta_p1)
                    yt2 = self.model_y[t](xeta_p2)
                y_sample1.append(yt1)
                y_sample2.append(yt2)

            y_sample1_cat = torch.cat(y_sample1,dim=1)
            y_sample2_cat = torch.cat(y_sample2,dim=1)
            loss_y, loss1_y, loss2_y = energy_loss_two_sample(y_sample, y_sample1_cat, y_sample2_cat)
            
            eta_true = torch.randn(y.size(), device=self.device)

            eta1 = []
            eta2 = []
            perm = torch.randperm(x.size(0))
            for t in range(self.T):
                sxz_p1 = torch.cat([s_list[t], x_list[t], z_list[t]], dim=1)
                sxz_p2 = torch.cat([s_list[t][perm], x_list[t][perm], z_list[t][perm]], dim=1)
                etat1 = self.model_eta[t](sxz_p1)
                etat2 = self.model_eta[t](sxz_p2)
 
                eta1.append(etat1)
                eta2.append(etat2)
            eta1_cat = torch.cat(eta1,dim=1)
            eta2_cat = torch.cat(eta2,dim=1)

            loss_eta, loss1_eta, loss2_eta = energy_loss_two_sample(eta_true, eta1_cat, eta2_cat)
            loss = loss_y + loss_eta
            loss.backward()
            self.optimizer_y.step()
            if (i == 0) or ((i + 1) % print_every_iter == 0):
                print(f'Epoch {i + 1}: loss {loss.item():.4f},\tloss_y {loss_y.item():.4f}, {loss1_y.item():.4f}, {loss2_y.item():.4f},\tloss_eta {loss_eta.item():.4f}, {loss1_eta.item():.4f}, {loss2_eta.item():.4f}')
    
    
    @torch.no_grad()
    def sample_causal_margin(self,s, x, sample_size=100):
        self.eval()
        x = x.to(self.device)
        s = s.to(self.device)
        all_y = []
        for t in range(self.T):
            if self.s_in_predict:
                yt = ((self.model_y[t].sample(torch.cat([s,x[:,:(t+1)*self.x_dim]], dim=1), sample_size = sample_size))>0.6).float()
            else:
                yt = ((self.model_y[t].sample(x[:,:(t+1)*self.x_dim], sample_size = sample_size))>0.5).float()
            all_y.append(yt)

        return torch.cat(all_y,dim=1).squeeze(0).permute(1, 0)
   
    @torch.no_grad()
    def predict_causal(self, s, x, sample_size=100):
        y_causal_margin = self.sample_causal_margin(s, x, sample_size)
        event_indicator = (y_causal_margin>0).float()
        
        return event_indicator
        