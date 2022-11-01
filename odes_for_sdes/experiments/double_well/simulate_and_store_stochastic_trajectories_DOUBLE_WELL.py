#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:20:31 2019

@author: dimitra
"""

import numpy as np
import copy
import joblib


h = 0.001
t_start = 0
T = 10
x0 = 0.
f = lambda x,t: 4*x-4*x*x*x
timegrid = np.arange(0,T,h)
g = 1
Ns = [500, 1000, 1500, 2000, 2500]
reps = 20


for N in Ns:
    F = dict()
    for repi in range(reps):
        print('N: %d, repetition: %d'%(N, repi))
        se = repi #setting a different seed for each repetition
        np.random.seed(se)
        F[repi] = np.zeros((N,timegrid.size))
        xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=N))
        for ti,tt in enumerate(timegrid):    
            if ti == 0:                
                F[repi][:,ti] = copy.deepcopy(xs)                
            else:
                for j in range(N):   
                    F[repi][j,ti] = F[repi][j,ti-1] + h* f(F[repi][j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h))
                
                
    joblib.dump(F,filename='DOUBLE_WELL_stochastic_trajectories_N_%d'%N)