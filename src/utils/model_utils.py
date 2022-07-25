import torch
import torch.nn as nn
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint


# Code taken from:
# https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order, device, include_sine=False, include_constant=True):
    # timesteps x latent dim
    m, n = X.shape
    l = library_size(n, poly_order, include_sine, include_constant)
    library = torch.ones((m,l), device=device)
    index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i] * X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i] * X[:,j] * X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q] * X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1

    return library


def equation_sindy_library(n=3, poly_order=3, device=1, include_sine=False, include_constant=True):
    # timesteps x latent dim
    l = library_size(n, poly_order, include_sine, include_constant)
    index = 1
    X = ['x', 'y', 'z']
    str_lib = ['1']
    
    for i in range(n):
        str_lib.append(X[i])
    
    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                str_lib.append(X[i] + X[j])
    
    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    str_lib.append(X[i] + X[j] + X[k])

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        str_lib.append(X[i] + X[j] + X[k] + X[q])
    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            str_lib.append(X[i] + X[j] + X[k] + X[q] + X[r])

    if include_sine:
        for i in range(n):
            str_lib.append('sin(' + X[i] + ')')

    return str_lib


def sindy_library_tf_order2(z, dz, poly_order, include_sine=False, include_constant=True):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    # timesteps x latent dim
    X = torch.cat((z, dz), dim=1)
    m, n = X.shape
    l = library_size(n, poly_order, include_sine, include_constant)
    library = torch.ones((m,l), device=device)
    index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i] * X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i] * X[:,j] * X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q] * X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1

    return library


def get_equation(lib, coef, start):
    res = start
    for i in range(len(coef)):
        if coef[i] != 0:
            res += str(coef[i]) + lib[i] + ' + '
    return res[:-2]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)