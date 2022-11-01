# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 02:33:19 2020

@author: Dimi
"""

from plot_2D_distrib import multivariateGrid
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #
col_gr2 = '#272e2d'
col_grn = '#208f33'
params = {'backend': 'ps',
  'axes.labelsize': 26,
  'font.size': 20,
  'legend.fontsize': 20,
  'xtick.labelsize': 20,
  'ytick.labelsize': 20,
  'text.usetex': True,
   'xtick.top': True,
   'ytick.right': True          }
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})
#%%
pairs = [ (u'$x$', u'$y$'), (u'$x$', u'$z$'), (u'$y$', u'$z$') ]
pairsind = [ (0, 1), (0, 2), (1, 2) ]

for ti in [400,800, 900]:

    dfG = pd.DataFrame(F[:,:,ti].T, columns=[u'$x$',u'$y$',u'$z$'])   
    dfG['kind'] = 'S'
    for pi,pair in enumerate(pairs):
        idx1 = pairsind[pi][0]
        idx2 = pairsind[pi][1]
        ax_lims = [[np.min(F[idx1,:,ti])-0.2,np.max(F[idx1,:,ti])+0.2], [np.min(F[idx2,:,ti])-0.2,np.max(F[idx2,:,ti])+0.2]]
        fig0 = multivariateGrid(pair[0], pair[1], 'kind',k_is_color=col_ye, df=dfG,legend= ti==0,ax_lims=ax_lims, scatter_pnts=False)
        
        plt.savefig('Lorenz_S_4000'+pair[0]+'vs'+pair[1]+'_t_%d.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0)     
        plt.savefig('Lorenz_S_4000'+pair[0]+'vs'+pair[1]+'_t_%d.png'%ti,  bbox_inches = 'tight', pad_inches = 0)  
        plt.close()
#%%    
    
for ti in [400,800, 900]:



    dfG = pd.DataFrame(G[:,:,ti].T, columns=[u'$x$',u'$y$',u'$z$'])
    
    dfG['kind'] = 'D'
    for pi,pair in enumerate(pairs):
        idx1 = pairsind[pi][0]
        idx2 = pairsind[pi][1]
        ax_lims = [[np.min(F[idx1,:,ti])-0.2,np.max(F[idx1,:,ti])+0.2], [np.min(F[idx2,:,ti])-0.2,np.max(F[idx2,:,ti])+0.2]]
    
        fig0 = multivariateGrid(pair[0], pair[1], 'kind',k_is_color=col_grn, df=dfG,legend= ti==0,ax_lims=ax_lims, scatter_pnts=False)
        plt.savefig('Lorenz_D_4000'+pair[0]+'vs'+pair[1]+'_t_%d.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0)     
        plt.savefig('Lorenz_D_4000'+pair[0]+'vs'+pair[1]+'_t_%d.png'%ti,  bbox_inches = 'tight', pad_inches = 0)  
        plt.close()
        
#%%
    
for ti in [400,800, 900]:



    dfG = pd.DataFrame(M[:,:,ti].T, columns=[u'$x$',u'$y$',u'$z$'])
    
    dfG['kind'] = 'Si'
    for pi,pair in enumerate(pairs):
        idx1 = pairsind[pi][0]
        idx2 = pairsind[pi][1]
        ax_lims = [[np.min(F[idx1,:,ti])-0.2,np.max(F[idx1,:,ti])+0.2], [np.min(F[idx2,:,ti])-0.2,np.max(F[idx2,:,ti])+0.2]]


        level_sets = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7, 0.8,0.9, 1]
        
        fig0 = multivariateGrid(pair[0], pair[1], 'kind',k_is_color=col_gr2, df=dfG,legend= ti==0, levels = level_sets,ax_lims=ax_lims, scatter_pnts=False)
        plt.savefig('Lorenz_Sinf_150000'+pair[0]+'vs'+pair[1]+'_t_%d.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0)     
        plt.savefig('Lorenz_D_150000'+pair[0]+'vs'+pair[1]+'_t_%d.png'%ti,  bbox_inches = 'tight', pad_inches = 0)  
        plt.close()
    