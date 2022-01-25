# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 01:15:07 2022

@author: maout
"""

import torch
import numpy as np
from typing import Union
from .kernels.RBF_pytorch import RBF

__all__ = ["torched_score_function_multid_seperate_all_dims"]



def torched_score_function_multid_seperate_all_dims(X: np.ndarray, Z: np.ndarray,
                                                    l: Union[float, torch.tensor, np.ndarray]=1.0,
                                                    func_out: Union[bool, None]=False,
                                                    C: Union[float, None]=0.001,
                                                    kern: Union[None, str]  ='RBF',
                                                    device: Union[bool,str]=None) -> torch.tensor:
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented
    by samples X for all dimensions simultaneously.
    Implemented with pytorch.

    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log
                               in the vicinity of the samples.

    Parameters
    -----------
            X: N x dim array,
               N samples from the density (N x dim), where dim>=2 the
               dimensionality of the system.
            Z: M x dim array,
              inducing points points (M x dim).
            func_out : Boolean,
                      True returns function,
                      if False returns grad-log-p evaluated on samples X.
            l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
            C: float,
              weighting constant
              (leave it at default value to avoid unreasonable contraction
              of deterministic trajectories).
            kern: string,
                options:
                    - 'RBF': radial basis function/Gaussian kernel
                    - 'periodic': periodic, not functional yet.
            device: device selection for exacution either on cpu or gpu.

    Returns
    -------
        res1: array with logarithmic gradient of the density  N_s x dim or function
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.

    """



    M, dim = Z.shape
    C = 0.001

    if not torch.is_tensor(X):
          # convert inputs to pytorch tensors if not already pytorched
          X = torch.tensor(X, dtype=torch.float64, device=device)

          if Z is None:
              Z = X
          elif not torch.is_tensor(Z):
              Z = torch.tensor(Z, dtype=torch.float64, device=device)
    N, D = X.size()
    if isinstance(l, (list, tuple, np.ndarray)) or (torch.is_tensor(l) and l.size(dim=0)>1):
          multil = True

          ### for different lengthscales for each dimension

          # pytorched
          K_instance = RBF(length_scale=l, multil=True, device=device)
          K_xz = K_instance.Kernel(X, Z)#.detach().numpy()

          K_instancez = RBF(length_scale=l, multil=True, device=device) ##instance of kernel object - non-evaluated
          K_s = K_instancez.Kernel(Z, Z)#.detach().numpy()

    else:
          multil = False
          K_instance = RBF(length_scale=l, multil=False, device=device)
          K_xz = K_instance.Kernel(X, Z)#.detach().numpy()

          K_instancez = RBF(length_scale=l, multil=False, device=device) ##instance of kernel object - non-evaluated
          K_s = K_instancez.Kernel(Z, Z)#.detach().numpy()



    Ksinv = torch.linalg.inv(K_s+ 1e-3 * torch.eye(M, device=device))
    A = torch.t(K_xz) @ K_xz ##matrix multilication
    #compute the gradient of the X x Z kernel
    gradx_K = -K_instance.gradient_X(X, Z) #shape: (N,M,dim)
    ##last axis will have the gradient for each dimension ### shape (M, dim)
    sumgradx_K = torch.sum(gradx_K ,axis=0)

    if func_out==False: #if output wanted is evaluation at data points

        res1 = torch.zeros(N, dim, dtype=torch.float64, device=device)
        # ### evaluatiion at data points
        # for di in range(dim):
        #     res1[:,di] = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) +\
        #                 Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K[:,di]

        for di in range(dim):
            res1[:,di] = -K_xz @ torch.linalg.inv( C*torch.eye(M, M, device=device) + \
                                            Ksinv @ A + 1e-3 * torch.eye(M, device=device))\
                                            @ Ksinv @ sumgradx_K[:,di]

        #res1 = np.einsum('ik,kj->ij', -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv, sumgradx_K)


    else:
        #### for function output

        if multil:

            if kern == 'RBF':

                def K_sz(x):
                    if not torch.is_tensor(x):
                          # convert inputs to pytorch tensors if not already pytorched
                          x = torch.tensor(x, dtype=torch.float64, device=device)
                    X_i = x[:, None, :] # shape (n, D) -> (n, 1, D)
                    Z_j = Z[None, :, :] # shape (M, D) -> (1, M, D)
                    sqd1     = torch.div( (X_i - Z_j)**2, l**2)
                    sqd     = torch.sum( sqd1, 2)
                    K_sz    = torch.exp( -0.5* sqd )

                    #K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
                    return K_sz


        elif not multil:
            if kern=='RBF':

                  def K_sz(y):
                      if not torch.is_tensor(y):
                          # convert inputs to pytorch tensors if not already pytorched
                          xu = torch.tensor(y, dtype=torch.float64, device=device)
                      X_i = xu[:, None, :] # shape (n, D) -> (n, 1, D)
                      Z_j = Z[None, :, :] # shape (M, D) -> (1, M, D)
                      sqd     = torch.sum( (X_i - Z_j)**2, 2)         # |X_i - Y_j|^2 # (N, M, D)
                      # Divide by length scale
                      sqd  = torch.div(sqd, l**2)
                      K_sz    = torch.exp( -0.5* sqd )

                      return K_sz




        res1 = lambda x: K_sz(x) @ (-torch.linalg.inv(C*torch.eye(M, M, device=device) + \
                                                      Ksinv @ A + 1e-3 * torch.eye(M, device=device))) @ Ksinv@sumgradx_K


    return res1   ### shape out N x dim