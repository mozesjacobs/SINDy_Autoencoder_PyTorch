import torch
import torch.nn as nn
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):
    
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.z_dim = args.z_dim
        self.u_dim = args.u_dim
        self.hidden_dim = args.hidden_dim
        self.poly_order = args.poly_order
        self.use_sine = args.use_sine
        self.include_constant = args.include_constant
        self.library_dim = library_size(self.z_dim, self.poly_order, use_sine=self.use_sine, include_constant=self.include_constant)
        self.mse = nn.MSELoss(reduction='mean')
        
        self.encoder = nn.Sequential(
            nn.Linear(self.u_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.u_dim)
        )
        
        self.sindy_coefficients = nn.Parameter(torch.randn(self.library_dim, self.z_dim, requires_grad=True))
        nn.init.xavier_normal_(self.sindy_coefficients)
        
    def forward(self, x, dx, lambdas):
        batch_size, T, _ = x.shape
        device = self.sindy_coefficients.device
        
        # reshape data to be (b * t) x u
        x = x.view(-1, self.u_dim).type(torch.FloatTensor).to(device)
        dx = dx.view(-1, self.u_dim).type(torch.FloatTensor).to(device)
        
        # encode and decode
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        # build the SINDy library using the latent vector
        theta = sindy_library(z, self.poly_order, device, self.use_sine, self.include_constant) # (b * T) x 20
        
        # calculate the z derivative
        dz = self.get_derivative(x, dx, self.encoder)
        
        # predict the z derivative using the library
        dz_pred = self.predict(theta)
        
        # predict the derivative of dx by using the predicted z derivative
        dx_pred = self.get_derivative(z, dz_pred, self.decoder)
                
        # calculate loss
        loss = self.loss_func(x, x_recon, dx_pred, dz_pred, dx, dz, lambdas)
        
        return loss
        
    def predict(self, theta):
        # sindy_coefficients: L x z
        theta = theta.unsqueeze(1) # (b * T) x L  --->   (b * T) x 1 x L
        return torch.matmul(theta, self.sindy_coefficients).squeeze() # (b x T) x z
    
    # Returns the first order time derivative of z (dz/dt) or the reconstructed x (dx/dt)
    # assumes only sigmoid activation
    def get_derivative(self, layer_output, dL, net):
        dz = dL
        for i in range(len(net) - 1):
            curr_layer = net[i]
            if isinstance(curr_layer, nn.Linear):
                w, b = curr_layer.weight, curr_layer.bias
                wT = w.T
                layer_output = torch.sigmoid(torch.matmul(layer_output, wT) + b)
                d_layer_output = layer_output * (1 - layer_output)  # derivative of sigmoid: σ(x) * (1−σ(x))
                dz = d_layer_output * (torch.matmul(dz, wT))
        dz = torch.matmul(dz, net[-1].weight.T)
        return dz
    
    def loss_func(self, x, x_recon, dx_pred, dz_pred, dx, dz, lambdas):
        # reconstruction loss
        l_recon = self.mse(x_recon, x)
        
        # SINDy loss in dx
        l_dx = lambdas[0] * self.mse(dx_pred, dx)
        
        # SINDy loss in dz
        l_dz = lambdas[1] * self.mse(dz_pred, dz)
        
        # SINDy regularization
        l_reg = lambdas[2] * torch.abs(self.sindy_coefficients).mean()
        
        return l_recon, l_dx, l_dz, l_reg