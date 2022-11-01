# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 05:14:23 2020

@author: Dimi
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

import copy
h = 0.001
t_start = 0
T = 3
dim = 2
x0 = np.array([0.5, 0.5])
f =  lambda x: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
timegrid = np.arange(0,T,h)
g = 1
Ns = [1500]#500, 1000, 1500, 2000, 2500]
foldername = 'fewer_n_vs_d/ou2d/'
for ni,N in enumerate(Ns):
    
    F = np.zeros((2,N,timegrid.size,20))
    for i in range(20): 
        #simulate stochastic#
        np.random.seed(i)
        
        xs = np.zeros((2,N))
        for ii in range(2):
            xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N)
        for ti,tt in enumerate(timegrid): 
            if ti == 0:                
                F[:,:,ti,i] = copy.deepcopy(xs)                   
            else:                    
                F[:,:,ti,i] = F[:,:,ti-1,i] + h* f(F[:,:,ti-1,i]) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(2,N))
                    
    joblib.dump(F,filename=foldername+'OU_2D_stochastic_trajectories_N%d'%(N))
        