# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 05:59:57 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt

import joblib

foldername = 'Otto_dynamics_paper_data/state_dependent/'
Ns = [500,1000,1500,2000,2500]
Ninf = 35000
Ms = [50,100,150,200]
sim_prec = 0.001
t_start = 0.
x0 = 1.0
num_ind = 100
T=4.5

f = lambda x,t: 4*x-4*x*x*x
g = 1.
gii = lambda x:g*np.sin(x)#x**(po)
C = 0.001
h = sim_prec #step
timegrid = np.arange(0,T,h)

xs=np.random.normal(loc=x0, scale=0.05,size=Ninf)
#Z = np.zeros((Ninf,timegrid.size))
#for ti,tt in enumerate(timegrid[:]):
#    #print(tt)
#    if ti == 0:
#        Z[:,ti] = xs        
#    else:       
#       
#        Z[:,ti] = Z[:,ti-1] + h* f(Z[:,ti-1],ti) + np.multiply(gii(Z[:,ti-1]),np.random.normal(loc = 0.0, scale = np.sqrt(h),size=Ninf))
#joblib.dump(Z,filename=foldername+'State_dependent_DW_sin_stochastic_trajectory_Ninf_%d'%(Ninf) ) 
#del Z
print('Ninf trajectories ready!')

for N in Ns:
    print('Particle number: %d' %N)
    Z = np.zeros((N,timegrid.size,20))
    for repi in range(20):   
        np.random.seed(20+repi)
        xs=np.random.normal(loc=x0, scale=0.05,size=N)
        for ti,tt in enumerate(timegrid[:]):
            #print(tt)
            if ti == 0:
                Z[:,ti,repi] = xs 
                
            else:       
               
                Z[:,ti,repi] = Z[:,ti-1,repi] + h* f(Z[:,ti-1,repi],ti) + np.multiply(gii(Z[:,ti-1,repi]),np.random.normal(loc = 0.0, scale = np.sqrt(h),size=N))
                
                
    joblib.dump(Z,filename=foldername+'State_dependent_DW_sin_stochastic_trajectories_N_%d'%(N) ) 