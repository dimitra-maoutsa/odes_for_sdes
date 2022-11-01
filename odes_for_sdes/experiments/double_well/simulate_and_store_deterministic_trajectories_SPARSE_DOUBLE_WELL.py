#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 01:00:31 2019

@author: dimitra
"""

import numpy as np
import copy
import joblib
from score_fucntion2 import score_function
from score_function_sparse import score_function_sparse
save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/DOUBLE_WELL_1D_USED/'
h = 0.001
t_start = 0
T = 5
x0 = 0.
f = lambda x,t: 4*x-4*x*x*x
timegrid = np.arange(0,T,h)
g = 1
Ns = [   2000]#,1000,1500,,2000, 2500
#Ms = [ 20, 40,60,80,100,120,140,160,180,200]
Ms = [150 ]#50,100,150,200
reps = 20



C = 0.001
def f_sparse(x,t,Z=None):
    if (not Z.all):        
        Z = np.linspace(np.min(x),np.max(x),round(x.size/10))
    gpsi= score_function_sparse(x,Z,C=1/x.size,l=0.5,funct_out=False) 
    return (f(x,t)-0.5*g**2 * gpsi.reshape(-1,))


for N in Ns:
    #S = dict()
    for M in Ms:
        #S = dict()
        print('M: %d' %M )            
        #inducing_pos = np.linspace(-2.5,2.5,M)
        S = np.zeros((N,timegrid.size,reps))
        for repi in range(reps):
            print('Deterministic function prior: N: %d, repetition: %d'%(N, repi))
            se = repi #setting a different seed for each repetition
            np.random.seed(se)
            xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=N))            
            for ti,tt in enumerate(timegrid): 
                #inducing_pos = np.linspace(np.min(S[:,ti-1]),np.max(S[:,ti-1]),M)
                inducing_pos = np.linspace(np.min(S[:,ti-1,repi]),np.max(S[:,ti-1,repi]),M)
                if ti == 0:                
                    S[:,ti,repi] = copy.deepcopy(xs)                
                else:                 
                    S[:,ti,repi] = S[:,ti-1,repi] + h* f_sparse(S[:,ti-1,repi],ti,inducing_pos)
                    
                
        joblib.dump(S,filename=save_file+'DOUBLE_WELL_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))
        
        #joblib.dump(S,filename=save_file+'DOUBLE_WELL_deterministic_trajectories_SPARSE_N_%d_fixed_inducing_M_%d'%(N,M))