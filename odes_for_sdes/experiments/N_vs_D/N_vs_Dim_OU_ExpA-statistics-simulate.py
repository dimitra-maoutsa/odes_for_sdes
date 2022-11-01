#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:30:07 2020

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import ot
from score_function_multid_seperate import score_function_multid_seperate

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'




dim = 1
M = 100
seeds = np.arange(10, 30,1)
#dims = [1,2,3,4,5,6]
Ns = [1000,2000,3000,4000,5000,6000]
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
g = 1
x0 = 0.5

####### f # multidim OU
def f(x,t):        
    ret = np.ones((dim,dim))
    np.fill_diagonal(ret, -4)        
    return ret@x


########## Otto dynamics
def f_seperate(x,t,N_sparse=100):
    dimi, N = x.shape    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]    
    Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )    
    gpsi = np.zeros((dimi, N))
    lnthsc = 2*np.std(x,axis=1)    
    for ii in range(dimi):        
        gpsi[ii,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=ii+1)[0] #C=0.1 !!!!!
    return (f(x,t)-0.5*g**2* gpsi)
##############



for N in Ns:
    for se in range(19):
        np.random.seed(se+101)
        print('Dimension: %d, particles: %d, seed:%d'%(dim,N,se))
        D = np.zeros((dim,N,timegrid.size))
        for ti,t in enumerate(timegrid):
            if ti==0: 
                for di in range(dim): 
                    D[di,:,0] = np.random.normal(loc=x0, scale=0.25,size=N)
            else:
                D[:,:,ti] = D[:,:,ti-1] + h* f_seperate(D[:,:,ti-1],t,M)
                    
        joblib.dump(D,filename=save_file+'OU_%d_D_Deterministic_Trajectories_N_%d_M_%d_seed_%d'%(dim,N,M,se))
        