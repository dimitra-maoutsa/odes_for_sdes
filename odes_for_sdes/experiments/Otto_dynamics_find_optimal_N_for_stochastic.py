#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:24:22 2019

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp as ks
import copy
from scipy.stats import wasserstein_distance as wd

"""
 Calculate N^{inf} for double well system
 """


h = 0.001
t_start = 0
T = 10
x0 = 0.
f = lambda x,t: 4*x-4*x*x*x


g = 1.
N_inf = np.zeros(20)
for repi in range(6,20):
    N = 2000
    se = repi #setting a different seed for each repetition
    np.random.seed(se)
    xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=9000))
    timegrid = np.arange(0,T,h)
    F = np.zeros((N,timegrid.size))
    
    flag = 0
    iteration = 0
    while not flag:
        M = np.zeros((N+200,timegrid.size))
        print('Iteration: %d' %iteration)
        for ti,tt in enumerate(timegrid):    
            if ti == 0:
                if iteration == 0:
                    F[:,ti] = copy.deepcopy(xs[:N])
                M[:,ti] = copy.deepcopy(xs[:N+200])
            else:
                for j in range(N):     
                    if iteration == 0:
                        F[j,ti] = F[j,ti-1] + h* f(F[j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h))
                for j in range(N+200):
                    M[j,ti] = M[j,ti-1] + h* f(M[j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h))
        print('Calculating wasserstein distance' )
        #Ks = np.zeros(F.shape[1])
        #Ksp = np.zeros(F.shape[1])
        Was = np.zeros(F.shape[1])
        for ti,tt in enumerate(timegrid):
            #Ks[ti], Ksp[ti] = ks(F[:,ti],M[:,ti])
            Was[ti] = wd(M[:,ti],F[:,ti])
            
#        plt.figure(),plt.plot(timegrid,Was),plt.xlabel('time'),plt.ylabel('Wasserstein distance')
#        plt.title('Iteration %d: ' %iteration)
        print(np.mean(Was))
        print(np.max(Was))
        if np.all(Was < 0.05):
            N_inf[repi] = N
            break;
            flag = 1
        else:
            iteration += 1
            del F
            F = copy.deepcopy(M)
            N = F.shape[0]
            print('Particle number: %d' %N)
            del M            
    
            
import joblib

joblib.dump(N_inf,filename='N_infinite_for_DOUBLE_WELL_dt_0_001_T_10_x0_0')  


#N_inf = joblib.load('/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/N_infinite_for_DOUBLE_WELL_dt_0_001_T_10_x0_0')          
mean_N_inf = np.mean(N_inf)
std_N_inf = np.std(N_inf)
reps = 20
F_inf = dict()
for repi in range(reps):
    print('N_inf: %d, repetition: %d'%(mean_N_inf, repi))
    se = repi #setting a different seed for each repetition
    np.random.seed(se)
    F_inf[repi] = np.zeros((mean_N_inf,timegrid.size))
    xs=np.sort(np.random.normal(loc=x0, scale=0.05,size=mean_N_inf))
    for ti,tt in enumerate(timegrid):    
        if ti == 0:                
            F[repi][:,ti] = copy.deepcopy(xs)                
        else:
            for j in range(mean_N_inf):   
                F_inf[repi][j,ti] = F_inf[repi][j,ti-1] + h* f(F_inf[repi][j,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h))
            
            
joblib.dump(F_inf,filename='DOUBLE_WELL_stochastic_trajectories_N_inf%d'%mean_N_inf)