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
        self.nonlinearity = args.nonlinearity
        
        self.encoder = self.build_net(self.u_dim, self.hidden_dim, self.z_dim)
        self.decoder = self.build_net(self.z_dim, self.hidden_dim, self.u_dim)
        
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
            # if linear layer, do the matrix multiply and add bias
            if isinstance(curr_layer, nn.Linear):
                wT, b = curr_layer.weight.T, curr_layer.bias
                output_before_act = torch.matmul(layer_output, wT) + b
            else: # if its the activation function, skip to next layer
                continue
            # calculate derivative for each type of nonlinearity (or no nonlinearity)
            if self.nonlinearity == 'sig':
                # derivative of sigmoid(x): σ(x) * (1−σ(x))
                layer_output = torch.sigmoid(output_before_act)
                d_layer_output = layer_output * (1 - layer_output)  
            elif self.nonlinearity == 'relu':
                # derivative of relu(x): 1 if x > 0, else 0
                layer_output = nn.functional.relu(output_before_act)
                d_layer_output = (output_before_act > 1).float()
            elif self.nonlinearity == 'elu':
                # derivative of elu(x): 1 if x > 0, else alpha * exp(x). we use alpha = 1.0 by default
                layer_output = nn.functional.elu(output_before_act)
                d_layer_output = torch.min(torch.exp(output_before_act), torch.ones_like(output_before_act))[0]
            else: # no nonlinearity
                d_layer_output = 1
            dz = d_layer_output * torch.matmul(dz, wT)
        return torch.matmul(dz, net[-1].weight.T)
    
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

    def build_net(self, in_dim, hidden_dim, out_dim):
        if self.nonlinearity == 'elu':
            nonlinearity = nn.ELU
        elif self.nonlinearity == 'sig':
            nonlinearity = nn.Sigmoid
        elif self.nonlinearity == 'relu':
            nonlinearity =  nn.ReLU
        else:  # no nonlinearity 
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, out_dim)
            )
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, out_dim)
        )