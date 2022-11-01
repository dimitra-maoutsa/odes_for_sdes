# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:08:41 2020

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

for ni,N in enumerate(Ns):
    print(ni)
    Wasdis = np.zeros((20,5000))
    F = joblib.load(foldername+'DOUBLE_WELL_stochastic_trajectories_N_%d'%N)
    print(np.mean(F[0][:,0]))
    print(np.std(F[0][:,0]))
    for i in range(20):             
        for ti in range(5000):
            Wasdis[i,ti] = ot.wasserstein_1d(F[i][:,ti],F_inf[:,ti])    



    joblib.dump(Wasdis,filename=foldername+'DOUBLE_WELL_Wasserstein_between_stochastic%d_vs_stochastic_%d'%(N,N_inf))