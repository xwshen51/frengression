from dragonnet.dragonnet import DragonNet # https://github.com/farazmah/dragonnet-pytorch
from pyro.contrib.cevae import CEVAE


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
import optuna
from pyro.contrib.cevae import CEVAE
from dragonnet.dragonnet import DragonNet


def mmd_rbf(x, y, gamma=None):
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    Z = torch.cat([x_flat, y_flat], dim=0)
    dist = (
        Z.pow(2).sum(1, keepdim=True)
        - 2 * Z @ Z.t()
        + Z.pow(2).sum(1, keepdim=True).t()
    )
    if gamma is None:
        d = dist.detach().cpu().numpy()
        gamma = 1.0 / (0.5 * np.median(d[d > 0]))
    K = torch.exp(-gamma * dist)
    n = x_flat.size(0)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()

class TARNetModel(nn.Module):
    def __init__(self, input_dim, rep_dims, head_dims, dropout):
        super().__init__()
        layers, last = [], input_dim
        for h in rep_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.repr_net = nn.Sequential(*layers)
        def make_head(indim):
            hl, cur = [], indim
            for h in head_dims:
                hl += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
                cur = h
            hl += [nn.Linear(cur, 1)]
            return nn.Sequential(*hl)
        self.h0 = make_head(last)
        self.h1 = make_head(last)
    def forward(self, x):
        z = self.repr_net(x)
        y0 = self.h0(z).squeeze(-1)
        y1 = self.h1(z).squeeze(-1)
        return y0, y1

class TARNetTrainer:
    def __init__(self, input_dim, rep_dims, head_dims, dropout):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TARNetModel(input_dim, rep_dims, head_dims, dropout).to(self.device)
    def fit(self, X_train, t_train, y_train,
            X_val, t_val, y_val,
            lr, weight_decay, batch_size, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32, device=self.device),
            torch.tensor(t_train, dtype=torch.float32, device=self.device),
            torch.tensor(y_train, dtype=torch.float32, device=self.device)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        best = float('inf')
        for _ in range(epochs):
            self.model.train()
            for xb, tb, yb in dl:
                optimizer.zero_grad()
                y0, y1 = self.model(xb)
                y_pred = torch.where(tb.unsqueeze(1)==1, y1, y0).squeeze(-1)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                tv = torch.tensor(t_val, dtype=torch.float32, device=self.device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                y0v, y1v = self.model(Xv)
                ypv = torch.where(tv.unsqueeze(1)==1, y1v, y0v).squeeze(-1)
                vloss = criterion(ypv, yv).item()
                best = min(best, vloss)
        return best
    def predict(self, X):
        self.model.eval()
        Xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y0, y1 = self.model(Xb)
        return y0.cpu().numpy(), y1.cpu().numpy()

class CFRNetTrainer:
    def __init__(self, input_dim, rep_dims, head_dims, dropout, ipm_weight=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TARNetModel(input_dim, rep_dims, head_dims, dropout).to(self.device)
        self.ipm_weight = ipm_weight
    def fit(self, X_train, t_train, y_train,
            X_val, t_val, y_val,
            lr, weight_decay, batch_size, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32, device=self.device),
            torch.tensor(t_train, dtype=torch.float32, device=self.device),
            torch.tensor(y_train, dtype=torch.float32, device=self.device)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        best = float('inf')
        for _ in range(epochs):
            self.model.train()
            for xb, tb, yb in dl:
                optimizer.zero_grad()
                z = self.model.repr_net(xb)
                y0 = self.model.h0(z).squeeze(-1)
                y1 = self.model.h1(z).squeeze(-1)
                y_pred = torch.where(tb.unsqueeze(1)==1, y1, y0).squeeze(-1)
                loss = criterion(y_pred, yb)
                z0 = z[tb.squeeze()==0]
                z1 = z[tb.squeeze()==1]
                if z0.size(0)>1 and z1.size(0)>1:
                    ipm = mmd_rbf(z0, z1)
                    loss = loss + self.ipm_weight*ipm
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                tv = torch.tensor(t_val, dtype=torch.float32, device=self.device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                z = self.model.repr_net(Xv)
                y0v = self.model.h0(z).squeeze(-1)
                y1v = self.model.h1(z).squeeze(-1)
                ypv = torch.where(tv.unsqueeze(1)==1, y1v, y0v).squeeze(-1)
                vloss = criterion(ypv, yv).item()
                best = min(best, vloss)
        return best
    def predict(self, X):
        self.model.eval()
        Xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.model.repr_net(Xb)
            y0 = self.model.h0(z).squeeze(-1)
            y1 = self.model.h1(z).squeeze(-1)
        return y0.cpu().numpy(), y1.cpu().numpy()

class CEVAETrainer:
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers, num_samples):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CEVAE(
            feature_dim=input_dim, latent_dim=latent_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_samples=num_samples, outcome_dist="normal"
        ).to(self.device)
    def fit(self, X_train, t_train, y_train,
            X_val, t_val, y_val,
            lr, weight_decay, batch_size, epochs):
        Xb = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        tb = torch.tensor(t_train, dtype=torch.float32, device=self.device)
        yb = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        elbo = self.model.fit(
            Xb, tb, yb,
            num_epochs=epochs, batch_size=batch_size,
            learning_rate=lr, weight_decay=weight_decay
        )
        return elbo[-1]
    def predict(self, X):
        Xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        ite = self.model.ite(Xb).cpu().numpy()
        return ite.mean(0)

class DragonNetTrainer:
    def __init__(self, input_dim, shared_hidden, outcome_hidden):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DragonNet(input_dim, shared_hidden, outcome_hidden)
        self.net.model.to(self.device)
    def fit(self, X_train, t_train, y_train,
            X_val, t_val, y_val,
            lr, weight_decay, batch_size, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.model.parameters(), lr=lr, weight_decay=weight_decay)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32, device=self.device),
            torch.tensor(t_train, dtype=torch.float32, device=self.device),
            torch.tensor(y_train, dtype=torch.float32, device=self.device)
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        best = float('inf')
        for _ in range(epochs):
            self.net.model.train()
            for xb, tb, yb in dl:
                optimizer.zero_grad()
                outs = self.net.model(xb)
                y0, y1 = outs[0], outs[1]
                y_pred = torch.where(tb.unsqueeze(1)==1, y1, y0).squeeze(-1)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
            self.net.model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                tv = torch.tensor(t_val, dtype=torch.float32, device=self.device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                outs_v = self.net.model(Xv)
                y0v, y1v = outs_v[0], outs_v[1]
                ypv = torch.where(tv.unsqueeze(1)==1, y1v, y0v).squeeze(-1)
                vloss = criterion(ypv, yv).item()
                best = min(best, vloss)
        return best
    def predict(self, X):
        self.net.model.eval()
        Xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outs = self.net.model(Xb)
            y0, y1 = outs[0], outs[1]
        return y0.cpu().numpy(), y1.cpu().numpy()
