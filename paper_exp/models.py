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

## Example

# def make_synthetic_data(N=1000, D=5, seed=0):
#     torch.manual_seed(seed)
#     X = torch.randn(N, D)
#     w_p, b_p = torch.randn(D), 0.1
#     p = torch.sigmoid(X @ w_p + b_p)
#     t = torch.bernoulli(p)
#     beta0 = torch.randn(D)
#     y0 = X @ beta0 + 0.1 * torch.randn(N)
#     tau = (X[:,0] * 2.0).clamp(min=0)
#     y1 = y0 + tau + 0.1 * torch.randn(N)
#     y = y0 * (1 - t) + y1 * t
#     return X.numpy(), t.numpy(), y.numpy(), y0.numpy(), y1.numpy()

# X, t, y, y0, y1 = make_synthetic_data(N=400, D=2)
# X_train, X_tmp, t_train, t_tmp, y_train, y_tmp = train_test_split(X, t, y, test_size=0.4, random_state=42)
# X_val, X_test, t_val, t_test, y_val, y_test = train_test_split(X_tmp, t_tmp, y_tmp, test_size=0.5, random_state=42)

# def tune_and_eval(model_name):
#     def objective(trial):
#         lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
#         wd = trial.suggest_loguniform("wd", 1e-7, 1e-4)
#         bs = trial.suggest_categorical("bs", [32, 64, 128])
#         epochs = trial.suggest_int("epochs", 50, 200)
#         if model_name == "tarnet":
#             rep1 = trial.suggest_int("rep1", 50, 200)
#             rep2 = trial.suggest_int("rep2", 50, 200)
#             head = trial.suggest_int("head", 50, 200)
#             drop = trial.suggest_uniform("drop", 0.0, 0.5)
#             trainer = TARNetTrainer(X_train.shape[1], [rep1,rep2], [head], drop)
#         elif model_name == "cfrnet":
#             rep1 = trial.suggest_int("rep1", 50, 200)
#             rep2 = trial.suggest_int("rep2", 50, 200)
#             head = trial.suggest_int("head", 50, 200)
#             drop = trial.suggest_uniform("drop", 0.0, 0.5)
#             ipm_w = trial.suggest_loguniform("ipm_weight", 0.01, 10.0)
#             trainer = CFRNetTrainer(X_train.shape[1], [rep1,rep2], [head], drop, ipm_w)
#         elif model_name == "cevae":
#             ld = trial.suggest_int("latent_dim", 10, 200)
#             hd = trial.suggest_int("hidden_dim", 20, 400)
#             nl = trial.suggest_int("num_layers", 2, 5)
#             ns = trial.suggest_categorical("num_samples", [10,50,100,200])
#             trainer = CEVAETrainer(X_train.shape[1], ld, hd, nl, ns)
#         else:  # dragonnet
#             sh = trial.suggest_int("shared_hidden", 50, 200)
#             oh = trial.suggest_int("outcome_hidden", 50, 200)
#             trainer = DragonNetTrainer(X_train.shape[1], sh, oh)
#         return trainer.fit(X_train, t_train, y_train, X_val, t_val, y_val,
#                             lr=lr, weight_decay=wd, batch_size=bs, epochs=epochs)
#     study = optuna.create_study(direction="minimize", study_name=f"{model_name}_tune")
#     study.optimize(objective, n_trials=50)
#     best = study.best_params
#     print(f"Best params for {model_name}: {best}")
#     # retrain
#     X_trn = np.vstack([X_train, X_val])
#     t_trn = np.concatenate([t_train, t_val])
#     y_trn = np.concatenate([y_train, y_val])
#     if model_name == "tarnet":
#         trainer = TARNetTrainer(X_trn.shape[1], [best['rep1'],best['rep2']], [best['head']], best['drop'])
#     elif model_name == "cfrnet":
#         trainer = CFRNetTrainer(X_trn.shape[1], [best['rep1'],best['rep2']], [best['head']], best['drop'], best['ipm_weight'])
#     elif model_name == "cevae":
#         trainer = CEVAETrainer(X_trn.shape[1], best['latent_dim'], best['hidden_dim'], best['num_layers'], best['num_samples'])
#     else:
#         trainer = DragonNetTrainer(X_trn.shape[1], best['shared_hidden'], best['outcome_hidden'])
#     trainer.fit(X_trn, t_trn, y_trn, X_test, t_test, y_test,
#                 lr=best['lr'], weight_decay=best['wd'], batch_size=best['bs'], epochs=best['epochs'])
#     if model_name == "cevae":
#         return trainer.predict(X_test)
#     else:
#         y0p, y1p = trainer.predict(X_test)
#         return y1p - y0p

# # Tune and train each model separately, storing ITEs
# print("Tuning and training TARNet...")
# ite_tarnet = tune_and_eval("tarnet")
# print("TARNet ITE shape:", ite_tarnet.shape)

# print("Tuning and training CFRNet...")
# ite_cfrnet = tune_and_eval("cfrnet")
# print("CFRNet ITE shape:", ite_cfrnet.shape)

# print("Tuning and training CEVAE...")
# ite_cevae = tune_and_eval("cevae")
# print("CEVAE ITE shape:", ite_cevae.shape)

# print("Tuning and training DragonNet...")
# ite_dragonnet = tune_and_eval("dragonnet")
# print("DragonNet ITE shape:", ite_dragonnet.shape)

# # Now you have separate ITE arrays:
# # ite_tarnet, ite_cfrnet, ite_cevae, ite_dragonnet
# print("All models complete.")