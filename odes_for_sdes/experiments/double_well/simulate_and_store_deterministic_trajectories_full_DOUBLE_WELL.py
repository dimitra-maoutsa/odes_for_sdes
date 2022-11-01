#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:13:37 2019

@author: dimitra
"""

import numpy as np
import copy
import joblib
from grad_log_p import score_functionD


h = 0.001
t_start = 0
T = 10
x0 = 0.
f = lambda x,t: 4*x-4*x*x*x
timegrid = np.arange(0,T,h)
g = 1
Ns = [ 1500, 2000, 2500]#500, 1000,
reps = 20



C = 0.001
def f_effD(x,t):#GP prior on derivative

    gpsi= score_functionD(x,'None',g**2,C=C,which=1,l=0.5)    
    return (f(x,t)-0.5*g**2 * gpsi)
    


for N in Ns:
    D = dict()
    for repi in range(reps):
        print('Deterministic derivative prior: N: %d, repetition: %d'%(N, repi))
        se = repi #setting a different seed for each repetition
        np.random.seed(se)
        D[repi] = np.zeros((N,timegrid.size))
        xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=N))
        for ti,tt in enumerate(timegrid):    
            if ti == 0:                
                D[repi][:,ti] = copy.deepcopy(xs)                
            else:                 
                D[repi][:,ti] = D[repi][:,ti-1] + h* f_effD(D[repi][:,ti-1],ti)
                
                
    joblib.dump(D,filename='DOUBLE_WELL_deterministic_trajectories_nonsparse_derivative_prior_N_%d'%N)