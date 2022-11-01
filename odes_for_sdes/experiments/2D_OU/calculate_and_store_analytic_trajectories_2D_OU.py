# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:27:27 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from odeintw import odeintw
import joblib
save_file='C:/Users/maout/Data_Assimilation_stuff/codes/results_otto/'
x0 = np.array([0.5, 0.5])
f =  lambda x,t: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
#xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N) 
sigma = 1
def f_var(C,t):
    
    A = np.array([[-4, 1],[1,-4]])
    
    return A@C + C@A.T + sigma**2*np.eye(2,2)   #this should have been be A@C + C@A.T + A@np.eye(2,2)*sigma**2

h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
C_0 = np.array([[0.25**2,0],[0,0.25**2]])

m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_0,timegrid)


n_sampls = 5000#2000

AF = np.zeros((2,n_sampls,timegrid.size))
for ti,t in enumerate(timegrid):
    # Define epsilon.
    epsilon = 0.0001
    # Add small pertturbation. 
    K = C_t[ti] + epsilon*np.identity(2)  
    AF[:,:,ti] = np.random.multivariate_normal(mean=m_t[ti].reshape(2,), cov=K, size=n_sampls).T
    
joblib.dump(AF,filename=save_file+'OU_2D_samples_from_analytic_trajectories_fortiming_N_%d'%(5000)) 