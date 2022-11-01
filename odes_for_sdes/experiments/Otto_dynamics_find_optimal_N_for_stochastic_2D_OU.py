#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 03:43:21 2020

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from scipy.stats import ks_2samp as ks
import copy
from scipy.stats import wasserstein_distance as wd
import ot
"""
 Calculate N^{inf} for 2D OU
 """

print('Calculate N^{inf} for 2D OU')
h = 0.001
t_start = 0
T = 3
x0 = np.array([0.5, 0.5])

f = lambda x,t: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
version = 2
g = 1.
N_inf = np.zeros(5)
for repi in range(5):
    N = 2000
    se = repi+version*5 #setting a different seed for each repetition
    np.random.seed(se)
    xs = np.zeros((2,9000))
    for ii in range(2):
        xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=9000)
    timegrid = np.arange(0,T,h)
    F = np.zeros((2,N,timegrid.size))
    
    flag = 0
    iteration = 0
    while not flag:
        M = np.zeros((2,N+200,timegrid.size))
        print('Iteration: %d' %iteration)
        for ti,tt in enumerate(timegrid):    
            if ti == 0:
                if iteration == 0:
                    F[:,:,ti] = copy.deepcopy(xs[:,:N])
                M[:,:,ti] = copy.deepcopy(xs[:,:N+200])
            else:
                for j in range(N):     
                    if iteration == 0:
                        F[:,j,ti] = F[:,j,ti-1] + h* f(F[:,j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=2)
                for j in range(N+200):
                    M[:,j,ti] = M[:,j,ti-1] + h* f(M[:,j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=2)
        print('Calculating wasserstein distance for 2D OU' )
        #Ks = np.zeros(F.shape[1])
        #Ksp = np.zeros(F.shape[1])
        Was = np.zeros(int(F.shape[-1]/10))
        for ti,tt in enumerate(timegrid[::10]):
            #Ks[ti], Ksp[ti] = ks(F[:,ti],M[:,ti])
            print(ti)
            xs = M[:,:,ti]
            xt = F[:,:,ti]
            # loss matrix
#            Mdist = ot.dist(xs.T, xt.T)
#            Mdist /= Mdist.max()
#            a, b = np.ones((N+200,)) / (N+200), np.ones((N,)) / N  # uniform distribution on samples
            #Was[ti] = ot.emd2(a,b,Mdist)
            lambd = 1e-3
            Was[ti] = ot.bregman.empirical_sinkhorn2(xs.T, xt.T ,lambd)
            #Was[ti] = ot.sinkhorn2(a, b, Mdist ,lambd)
            
#        plt.figure(),plt.plot(timegrid,Was),plt.xlabel('time'),plt.ylabel('Wasserstein distance')
#        plt.title('Iteration %d: ' %iteration)
        print(np.mean(Was))
        print(np.max(Was))
        print(np.min(Was))
        if np.all(Was < 0.05):
            N_inf[repi] = N
            break;
            flag = 1
        else:
            iteration += 1
            del F
            F = copy.deepcopy(M)
            N = F.shape[1]
            print('Particle number: %d' %N)
            del M            
    
            
import joblib

joblib.dump(N_inf,filename='N_infinite_for_2Dim_OU_dt_0_001_T_10_x0_0_version%d'%version)  
#%%

#N_inf = joblib.load('/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/N_infinite_for_DOUBLE_WELL_dt_0_001_T_10_x0_0')          
mean_N_inf = 3000#np.mean(N_inf)
#std_N_inf = np.std(N_inf)
reps = 20

F_inf = dict()
for repi in range(reps):
    print('N_inf for 2dim OU: %d, repetition: %d'%(mean_N_inf, repi))
    se = repi #setting a different seed for each repetition
    np.random.seed(se)
    F_inf[repi] = np.zeros((2,mean_N_inf,timegrid.size))
    xs = np.zeros((2,mean_N_inf))
    for ii in range(2):
        xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=mean_N_inf)
    for ti,tt in enumerate(timegrid):    
        if ti == 0:                
            F_inf[repi][:,:,ti] = copy.deepcopy(xs)                
        else:
            for j in range(mean_N_inf):   
                F_inf[repi][:,j,ti] = F_inf[repi][:,j,ti-1] + h* f(F_inf[repi][:,j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=2)
            
            
joblib.dump(F_inf,filename='2Dim_OU_stochastic_trajectories_N_inf%d'%mean_N_inf)