# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:53:20 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import ot
import copy





def KL(m1,m2,S1, S2):
    """
    Calculates KL divergence between two gaussiandistributions with mean m1, m2 
    and covariance matrices S1 and S2
    Expectation taken over m1, S1
    
    """
    d = m1.size
    S2inv = np.linalg.inv(S2)
    KL = 0.5*( np.log(np.linalg.det(S2)/np.linalg.det(S1))- d + np.trace( S2inv @S1) + (m2-m1).T @S2inv@ (m2-m1) )
    
    
    return KL


foldername = 'Otto_dynamics_paper_data/OU2d/'#Figure_2_1D_Double_well_vs_N_and_M/'
N_inf = 2000
F_inf = joblib.load(foldername+'OU_2D_samples_from_analytic_trajectories_N_%d'%2000)
##stachastic trajectories
Ns = [2000]#500, 1000, 1500, 2000, 2500]



h = 0.001
t_start = 0
T = 3
dim = 2
x0 = np.array([0.5, 0.5])
f =  lambda x: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
timegrid = np.arange(0,T,h)
g = 1


for ni,N in enumerate(Ns):
    Wasdis = np.zeros((20,timegrid.size))
    KLdis = np.zeros((20,timegrid.size))
    
    for i in range(18,20): 
        #simulate stochastic#
        np.random.seed(i)
        F = np.zeros((2,N,timegrid.size))
        xs = np.zeros((2,N))
        for ii in range(2):
            xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N)
        for ti,tt in enumerate(timegrid): 
            if ti == 0:                
                F[:,:,ti] = copy.deepcopy(xs)                   
            else:                    
                F[:,:,ti] = F[:,:,ti-1] + h* f(F[:,:,ti-1]) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(2,N))
                
        print('Particle number: %d'%N)
        
        print('Repetition: i= %d'%(i))
        for ti in range(3000):
            #print(ti)
            xs = F_inf[:,:,ti] 
            xt = F[:,:,ti]  #index i denotes repetition
            # loss matrix
            Mdist = ot.dist(xs.T, xt.T)
            Mdist /= Mdist.max()
            a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
            Wasdis[i,ti] = ot.emd2(a,b,Mdist)
            #Wasdis[N][i*20+j,ti] = ot.wasserstein_1d(F[i][:,ti],F_inf[j][:,ti])
            
            #### calculate KL
            means = np.mean(F[:,:,ti],axis=1)
            covs = np.cov(F[:,:,ti])
            meansinf = np.mean(F_inf[:,:,ti],axis=1)
            covsinf = np.cov(F_inf[:,:,ti])
            KLdis[i,ti] =  KL(meansinf , means , covsinf , covs )
            
            
        joblib.dump(Wasdis,filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_stochastic_N%d_all_to_allmissing'%(N_inf,N))
        joblib.dump(KLdis,filename=foldername+'OU_2D_KL_between_analytic%d_vs_stochastic_N%d_all_to_allmissing'%(N_inf,N))
        