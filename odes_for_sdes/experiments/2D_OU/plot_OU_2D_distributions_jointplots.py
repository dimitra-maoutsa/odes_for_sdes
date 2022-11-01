# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 04:27:29 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import joblib
from scipy.stats import skew, kurtosis  
import pandas as pd
#from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mplt
from plot_2D_distrib import multivariateGrid

from plot_statistics import plot_statistics
foldername = 'fewer_n_vs_d/ou2d/'


Ns = [ 1000, 1500, 2000, 2500]
N= 1500
filename=foldername+'OU_2D_KDE_deterministic_trajectories%d'%(N)  


D = joblib.load(filename)   


#plt.figure(),
#plt.hist(D[0,:,-2,0],100) 
timegrid = np.linspace(0,3,3000)
plot_statistics(timegrid,[D[:,:,:,0],D[:,:,:,1]])
#D=F_KDE1
#%%
for ti in [ -2]:
    
    #xss = np.zeros(-2,2,100)
    V = lambda x,y: (0.5*x**2 -0.5*x*y+0.5*y**2)
    #fv = lambda x,y: np.exp(-(2/g**2)* V(x,y))
    Zx1 = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100), sparse=False, indexing='ij')
    Zx = np.array([Zx1[0].reshape(-1,), Zx1[1].reshape(-1,)])
    
    dfG = pd.DataFrame(D[:,:,ti].T, columns=['x','y'])
    #dfF = pd.DataFrame(F[:,:,ti].T, columns=['x','y'])
    #dfM = pd.DataFrame(M[:,:,ti].T, columns=['x','y'])
    dfG['kind'] = 'deter_s'
    #dfF['kind'] = 'stoch'
    #dfM['kind'] = 'deter'
    #df=pd.concat([dfG,dfF])
    
    level_sets = [0.2,0.4, 0.6, 0.8, 1]
    ax_lims = [[np.min(D[0,:,ti])-0.2,np.max(D[0,:,ti])+0.2], [np.min(D[1,:,ti])-0.2,np.max(D[1,:,ti])+0.2]]
    fig0 = multivariateGrid('x', 'y', 'kind', df=dfG,legend= ti==0, levels = level_sets,ax_lims=ax_lims)
    
    #analyt = fv(Zx[0],Zx[1])
    
    
    
#    CS = plt.contour(Zx1[0], Zx1[1], analyt.reshape(100,100), level_sets,colors='r')    
#    plt.clabel(CS, inline=1, fontsize=10)