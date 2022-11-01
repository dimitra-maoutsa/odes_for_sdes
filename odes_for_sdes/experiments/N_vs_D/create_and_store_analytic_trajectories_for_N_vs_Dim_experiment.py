#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:55:46 2020

@author: dimitra
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
"""
###############################################
##Create analytic trajectory samples for N-dim OU
#################################################
"""


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from odeintw import odeintw
import joblib
save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'

dims = [1,2,3,4,5,6]
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
for dim in dims:
    
    

    def f(x,t):        
        ret = np.ones((dim,dim))
        np.fill_diagonal(ret, -4)        
        return ret@x
            
    def f_var(C,t):    
        A = np.ones((dim,dim))
        np.fill_diagonal(A, -4)    
        return A@C + C@A.T + 1*np.eye(dim,dim)

    #initial conditions
    x0 = np.ones(dim)*0.5
    C_0 = np.zeros((dim,dim))
    np.fill_diagonal(C_0,0.25**2)    
    #integrate
    m_t = odeint(f, x0, timegrid)
    C_t = odeintw(f_var, C_0,timegrid)


    n_sampls = 1000*dim

    AF = np.zeros((dim,n_sampls,timegrid.size))
    for ti,t in enumerate(timegrid):
        # Define epsilon.
        epsilon = 0.0001
        # Add small pertturbation. 
        K = C_t[ti] + epsilon*np.identity(dim)  
        AF[:,:,ti] = np.random.multivariate_normal(mean=m_t[ti].reshape(dim,), cov=K, size=n_sampls).T
        
    joblib.dump(AF,filename=save_file+'OU_%d_D_samples_from_analytic_trajectories_N_%d'%(dim,n_sampls)) 








