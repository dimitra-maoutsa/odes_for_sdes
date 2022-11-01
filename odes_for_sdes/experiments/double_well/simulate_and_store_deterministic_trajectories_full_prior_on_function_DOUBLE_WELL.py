#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:54:04 2019

@author: dimitra
"""

import numpy as np
import copy
import joblib
from score_fucntion2 import score_function


h = 0.001
t_start = 0
T = 10
x0 = 0.
f = lambda x,t: 4*x-4*x*x*x
timegrid = np.arange(0,T,h)
g = 1
Ns = [ 1000, 1500, 2000, 2500]#500
reps = 20



C = 0.001
def f_eff(x,t):#plain GP prior    
    gpsi= score_function(x,'None',g**2,C=C,which=1,l=0.5)
    return (f(x,t)-0.5*g**2 * gpsi.reshape(-1,))
    


for N in Ns:
    DF = dict()
    for repi in range(reps):
        print('Deterministic function prior: N: %d, repetition: %d'%(N, repi))
        se = repi #setting a different seed for each repetition
        np.random.seed(se)
        DF[repi] = np.zeros((N,timegrid.size))
        xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=N))
        for ti,tt in enumerate(timegrid):    
            if ti == 0:                
                DF[repi][:,ti] = copy.deepcopy(xs)                
            else:                 
                DF[repi][:,ti] = DF[repi][:,ti-1] + h* f_eff(DF[repi][:,ti-1],ti)
                
                
    joblib.dump(DF,filename='DOUBLE_WELL_deterministic_trajectories_nonsparse_function_prior_N_%d'%N)