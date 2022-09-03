# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 01:21:34 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import ot

foldername = 'Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
N_inf = 2000
F_inf = joblib.load(foldername+'2Dim_OU_stochastic_trajectories_N_inf%d'%N_inf)
##stachastic trajectories
Ns = [500, 1000, 1500, 2000, 2500]
Wasdis = dict()
for ni,N in enumerate(Ns):
    print(ni)
    Wasdis = np.zeros((20*20,10000))
    F = joblib.load(foldername+'2Dim_OU_stochastic_trajectories_N%d'%N)
    for i in range(20): 
        for j in range(20):
            print(i*20+j)
            for ti in range(10000):
                xs = F_inf[j][:,:,ti]
                xt = F[i][:,:,ti]
                # loss matrix
                Mdist = ot.dist(xs.T, xt.T)
                Mdist /= Mdist.max()
                a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
                Wasdis[i*20+j,ti] = ot.emd2(a,b,Mdist)
                #Wasdis[N][i*20+j,ti] = ot.wasserstein_1d(F[i][:,ti],F_inf[j][:,ti])
        
    joblib.dump(Wasdis,filename=foldername+'OU_2D_Wasserstein_between_stochastic%d_vs_stochastic%d_all_to_all'%(N_inf,N))
        
#plt.figure()    
#for ni,N in enumerate(Ns):
#    plt.plot(Wasdis[N][0],label=N)
#    
#plt.legend()


#means = np.zeros((20*20,len(Ns)))
#maxwas =   np.zeros((20*20,len(Ns))) 
#for ni,N in enumerate(Ns):
#    for i in range(20*20):
#        
#        means[i,ni] = np.mean(Wasdis[N][i])
#        maxwas[i,ni] = np.max(Wasdis[N][i])
#        
#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(Ns,np.mean(means,axis=0),'o')
#plt.subplot(1,2,2)
#plt.plot(Ns,np.mean(maxwas,axis=0),'o')



