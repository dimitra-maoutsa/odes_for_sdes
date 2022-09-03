# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:31:32 2019

@author: Dimi
"""

import numpy as np
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance as wd

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'



#### load stohastic paths N_inf

F_inf = joblib.load(save_file+'DOUBLE_WELL_stochastic_trajectories_N_inf2630')

reps = 20
fig1 = plt.figure()
for repi in range(reps):
    ax_1 = fig1.add_subplot(5,4,repi+1)
    ax_1.plot(F_inf[repi].T, alpha=0.2)
    
Ninf_ws = np.zeros((reps,reps,F_inf[0].size[1]))
for repi in range(reps):
    for repii in range(repi,reps):
        for ti in range(F_inf[0].size[1]): #for every timepoint
            #compare the Ninf distributions among themselves
            Ninf_ws[repi,repii,ti] = wd(F_inf[repi][:,ti],F_inf[repii][:,ti])

for ti in range(1,F_inf[0].size[1],step=100): #for every timepoint
    plt.figure(),
    plt.imshow(Ninf_ws[:,:,ti]) 
    plt.colorbar()
    
#### load stochastic paths for various Ns

    

