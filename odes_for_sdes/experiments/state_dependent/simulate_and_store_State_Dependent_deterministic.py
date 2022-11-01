# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 05:35:16 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
from score_function_sparse import score_function_sparse
import copy
import joblib
#from scipy.stats import wasserstein_distance as wd
foldername = 'Otto_dynamics_paper_data/state_dependent/'
N = 2500#[500,1000,1500,2000,2500]
Ms = [50]#50,100,150,200
sim_prec = 0.001
t_start = 0.
x0 = 1.0
num_ind = 100
T=4.5

f = lambda x,t: 4*x-4*x*x*x
g = 1.
gii = lambda x:g*np.sin(x)#x**(po)
C = 0.001

def f_sparse(x,t,Z=None):
    if (not Z.all):        
        Z = np.linspace(np.min(x),np.max(x),round(x.size/10))
    gpsi= score_function_sparse(x,Z,C=C,l=0.25,funct_out=False) #C=1/x.size

    return (f(x,t)-0.5*gii(x)**2 * gpsi.reshape(-1,) - 0.5*g**2*2*np.sin(x)*np.cos(x))#(2*po)*x**(2*po-1))
 
h = sim_prec #step
timegrid = np.arange(0,T,h)

for M in Ms:
    print('Particle number: %d, sparsity: %d'%(N,M))
    F = np.zeros((N,timegrid.size,20))
    
    for repi in range(20):  
        print(repi)
        np.random.seed(20+repi)
        xs=np.random.normal(loc=x0, scale=0.05,size=N)
        for ti,tt in enumerate(timegrid[:]):
            #print(tt)
            if ti == 0:            
                F[:,ti,repi] = copy.deepcopy(xs)            
            else:           
                inducing = np.linspace(np.min(F[:,ti-1,repi]),np.max(F[:,ti-1,repi]),M)           
                
                F[:,ti,repi] = F[:,ti-1,repi] + h*f_sparse(F[:,ti-1,repi],tt,inducing)#+ 0.001*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=N)
            
    joblib.dump(F, filename='State_dependent_DW_sin_deterministic_trajectories_N_%d_M_%d'%(N,M))   
    
####calculate Wasserstein
#Ninf_ws[repi,repii,ti] = wd(F_inf[repi][:,ti],F_inf[repii][:,ti])