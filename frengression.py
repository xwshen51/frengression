import torch
from engression.models import StoNet
from engression.loss_func import energy_loss_two_sample
sigmoid = torch.nn.Sigmoid()

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
    
    def specify_causal(self, causal_margin):
        def causal_margin1(x_eta):
            x = x_eta[:, :self.x_dim]
            eta = x_eta[:, self.x_dim:]
            return causal_margin(x, eta)
        self.model_y = causal_margin1
    
    def reset_y_models(self):
        self.model_y = StoNet(self.x_dim + self.y_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)
        self.model_eta = StoNet(self.x_dim + self.z_dim, self.y_dim, self.num_layer, self.hidden_dim, self.noise_dim, add_bn=False, noise_all_layer=False).to(self.device)

