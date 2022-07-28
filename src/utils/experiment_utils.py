import torch
import numpy as np
from src.utils.model_utils import init_weights, equation_sindy_library, get_equation

def print_gov_eqs(net):
    library = equation_sindy_library(net.z_dim, net.poly_order)
    coefs = (net.threshold_mask * net.sindy_coefficients).detach().cpu().numpy()
    print(get_equation(library, coefs[:,0], "X' = "))
    print(get_equation(library, coefs[:,1], "Y' = "))
    print(get_equation(library, coefs[:,2], "Z' = "))