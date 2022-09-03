# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:55:25 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import ot
from scipy.stats import wasserstein_distance as wd

foldername = 'DOUBLE_WELL_1D_USED/'

#N_inf = joblib.load(foldername+'N_infinite_for_DOUBLE_WELL_dt_0_001_T_10_x0_0')
#
#mean_N_inf = np.mean(N_inf)
#std_N_inf = np.std(N_inf)
N_inf = 26000
#F_inf = joblib.load(foldername+'DOUBLE_WELL_stochastic_trajectories_N_inf%d'%N_inf)
Dict = joblib.load(foldername+'Data_for_1d_DW_for_plot_statistics')
F_inf = Dict['Z']
print(F_inf.size)

print(np.mean(F_inf[:,0]))
print(np.std(F_inf[:,0]))


##stachastic trajectories
Ns = [2500]#[500, 1000, 1500, 2000, 2500]



Ms = [ 50,100,150,200]
timelength = 5000
WasdisDS = dict()
for ni,N in enumerate(Ns):
    print(ni)
#    if N==500:
#        F = joblib.load(foldername+'DOUBLE_WELL_deterministic_trajectories_SPARSE_N_%d_moving_inducing'%N)
    for M in Ms:
        
        F = joblib.load(foldername+'DOUBLE_WELL_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M)) 
        WasdisDS = np.zeros((20,timelength))
        for i in range(20): 
        
            for ti in range(timelength):
                if False:#N== 500:
                    WasdisDS[i*20+j,ti] = ot.wasserstein_1d(F[i][M][:,ti],F_inf[:,ti])
                else:
                    WasdisDS[i,ti] = ot.wasserstein_1d(F[:,ti,i],F_inf[:,ti])
                        
                    
                
        joblib.dump(WasdisDS,filename=foldername+'DOUBLE_WELL_Wasserstein_between_stochastic%d_vs_deterministic%d__M_%d'%(N_inf,N,M))           
                
                