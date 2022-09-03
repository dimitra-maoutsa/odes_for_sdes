# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:41:19 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import ot

foldername = 'Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
N_inf = 2000
F_inf = joblib.load(foldername+'OU_2D_samples_from_analytic_trajectories_N_%d'%2000)
##stachastic trajectories
Ns = [2500]#500, 1000, 1500, 2000, 2500]
Ms = [50]# 50,100,150,200]
Wasdis = dict()
for ni,N in enumerate(Ns):
    
    for M in Ms:
        
        Wasdis = np.zeros((20,3000))
        F = joblib.load(foldername+'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))
        for i in range(20): 
            
            print('Particle number: %d'%N)
            print('Sparse number: %d'%M)
            print('Repetition: i= %d'%(i))
            for ti in range(3000):
                print(ti)
                xs = F_inf[:,:,ti] 
                xt = F[:,:,ti,i]  #index i denotes repetition
                # loss matrix
                Mdist = ot.dist(xs.T, xt.T)
                Mdist /= Mdist.max()
                a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
                Wasdis[i,ti] = ot.emd2(a,b,Mdist)
                #Wasdis[N][i*20+j,ti] = ot.wasserstein_1d(F[i][:,ti],F_inf[j][:,ti])
        
            joblib.dump(Wasdis,filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_deterministic_N%d_M_%d_all_to_all'%(N_inf,N,M))
        