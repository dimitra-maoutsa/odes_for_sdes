# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:29:54 2020

@author: Dimi
"""

import numpy as np
import copy
import joblib
from score_function_multid_seperate import score_function_multid_seperate

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
h = 0.001
t_start = 0
T = 3
dim = 2
x0 = np.array([0.5, 0.5])
f =  lambda x: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
timegrid = np.arange(0,T,h)
g = 1
Ns = [  1500]#,1000,1500,,2000, 2500
#Ms = [ 20, 40,60,80,100,120,140,160,180,200]
Ms = [50,100,150,200]
reps = 20

C = 0.001
gii = np.multiply(np.ones(dim),g)

def f_seperate(x,t=0):#plain GP prior
    N_sparse = 100
    dimi, N = x.shape    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]    
    Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )    
    gpsi = np.zeros((dimi, N))
    lnthsc = 2*np.std(x,axis=1)
    for ii in range(dimi):        
        gpsi[ii,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=ii+1)[0] #C=0.1 !!!!!
    
    gi_term = np.tile(gii**2 ,(N,1)).T
    return (f(x)-0.5* np.multiply( gi_term, gpsi))



for N in Ns:    
    for M in Ms:        
        print('M: %d' %M )   
        G = np.zeros((dim,N,timegrid.size,reps))
        for repi in range(reps):
            print('Deterministic 2D OU: N: %d, repetition: %d'%(N, repi))
            se = repi #setting a different seed for each repetition
            np.random.seed(se)
            xs = np.zeros((2,N))
            for ii in range(2):
                xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N)            
            for ti,tt in enumerate(timegrid): 
                
                if ti == 0:                
                    G[:,:,ti,repi] = copy.deepcopy(xs)                
                else:                 
                    G[:,:,ti,repi] = G[:,:,ti-1,repi] + h* f_seperate(G[:,:,ti-1,repi])
                    
                
        joblib.dump(G,filename=save_file+'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))
        
  