# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:25:47 2023

@author: maout
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: dimitra
"""


import numpy as np
from matplotlib import pyplot as plt

#from score_fucntion2_multid import score_function_multid
from score_function_multid_seperate import score_function_multid_seperate
#import sdeint
import seaborn as sns
#from score_function_lite import dens_est_lite

import pandas as pd
from copy import deepcopy

import joblib
save_folder = 'Otto_results/nonlinear'
folder_name = 'Lorenz63/'
PLOT_MARGINALS = False
col_gr = '#3D4A49'
col_ro = '#c010d0'

h = 0.001 #sim_prec
t_start = 0.
t_end = 1
T =2#.5 #t_end
#initial condition
dim = 6

g = 10#np.sqrt(0.0025)#np.array([0.5,2,0.5,1])


f = lambda x,t: x*0

#f = lambda x,t: np.array([-x[0]**1+0.5*x[1] , -x[1]**1+0.3*x[0]])



x0 = [-0,-0,0,0,0,0]#,0,0,0,0 ]
                 


Ns = 2000 # number of stochastic particles
N = Ns #number of particles

#diffission coefficient/fucntion
#gii = lambda x,t:np.array([1*g,1* g])#np.multiply(np.ones(dim),g)
xs = np.zeros((dim,N))
for ii in range(dim):
    xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N)
#xs=np.array([(),(np.random.normal(loc=x02, scale=0.25,size=N)),(np.random.normal(loc=x03, scale=0.25,size=N))])

opt_l = []
opt_s = []

# def f_eff(x,t):#plain GP prior
#     lnthsc = 40#np.max(x) - np.min(x)
#     gpsi= score_function_multid(x.T,'None',g**2,C=0.01,which=1,l=lnthsc)
#     gpsi2 = np.array([gpsi[:N],gpsi[N:]]).reshape(x.shape)

#     return (f(x,t)-0.5*g**2 * gpsi2)





def f_seperate(x,t):#plain GP prior
    N_sparse = 500
    dimi, N = x.shape
    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
    
    Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )
    
    gpsi = np.zeros((dimi, N))
    lnthsc = 2*np.std(x,axis=1)
    #print(lnthsc)
    for ii in range(dimi):
        #print('here')
        gpsi[ii,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=ii+1)[0] #C=0.1 !!!!!
    
    
    
    
    
    return (f(x,t)-0.5* g**2* gpsi)





  

timegrid = np.arange(0,T,h)
#Z = np.zeros((2,N,timegrid.size))
F = np.zeros((dim,Ns,timegrid.size))
G = np.zeros((dim,N,timegrid.size))
#M = np.zeros((dim,Ninf,timegrid.size))
#w = np.zeros((N,timegrid.size))
#v = np.zeros((N,N,timegrid.size))




#Z = sdeint.itoint(f_eff, gi, xs, timegrid).T



for ti,t in enumerate(timegrid):
    
    print(ti)
    if ti==0: 
        #Z[:,:,ti] = deepcopy(xs) 
        G[:,:,ti] = deepcopy(xs[:,:N]) 
        F[:,:,ti] = deepcopy(xs[:,:Ns])
        #M[:,:,ti] = deepcopy(xs) 
    else:
        #feff_t = f_eff_lite(Z[:,:,ti-1],ti)
        G[:,:,ti] = G[:,:,ti-1] + h* f_seperate(G[:,:,ti-1],t)

            
            #F[:,i,ti] = F[:,i,ti-1] + h* f(F[:,i,ti-1],ti) + (g)*np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=dim) ]).reshape(dim,)
        F[:,:,ti] = F[:,:,ti-1] + h* f(F[:,:,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,Ns))




#%%
from scipy.stats import norm
import seaborn as sns
rv = norm(loc = 0., scale = np.sqrt(g**2*T)) #2*D*T = 2*(g**2)/2*T
x = np.arange(-40, 40, .1)
#x = np.arange(-10, 10, .1)
plt.figure()
for i in range(dim):
    plt.subplot(dim,1, 1+i)
    #plot the pdfs of these normal distributions 
    plt.plot(x, rv.pdf(x), 'k--', lw=3, label='true')
    sns.distplot(G[i,:,-1], 100, hist=False, label='D') ##G deterministic
    sns.distplot(F[i,:,-1], 100, hist=False, label='S')
    # plt.hist(F[i,:,-1], 100, histtype='step', density=True)
    # plt.hist(G[i,:, -1], 100,  histtype='step', density=True)
    plt.ylabel(r'$p_{%d}(x)$'%(i+1))
    plt.xlim([-45,45])
    #plt.xlim([-5,5])
    if i==0:
        plt.legend(ncol=3, bbox_to_anchor=[0.3, 0.9] ,framealpha =0)

    

    
plt.xlabel('x')
savepic = "C://Users//maout//Data_Assimilation_stuff//my_papers//"
plt.savefig(savepic+"Brownian_motion_%dD_T_2_histogram_sigma%d_N_%d.png"%(dim, g, N), bbox_inches='tight', transparent='False',  facecolor='white',dpi=300)
#%%


CeF = np.cov(F[:,:,-1])
CeG = np.cov(G[:,:,-1])#determin
CeAn = np.diag([g**2*T]*dim)


plt.figure()
plt.subplot(1,3,1)
im = plt.imshow(CeG, vmin=0, vmax=210)#,  aspect='auto')
ax1 = plt.gca()

# Loop over data dimensions and create text annotations.
for i in range(dim):
    for j in range(dim):
        if i==j:
            text = ax1.text(j, i, int(CeG[i, j]),
                           ha="center", va="center", color="k", size=16-dim)

plt.title('deterministic')
plt.subplot(1,3,2)
plt.imshow(CeF, vmin=0, vmax=210)#,  aspect='auto')
ax2 = plt.gca()
# Loop over data dimensions and create text annotations.
for i in range(dim):
    for j in range(dim):
        if i==j:
            text = ax2.text(j, i, int(CeF[i, j]),
                           ha="center", va="center", color="k", size=16-dim)
plt.title('stochastic')
plt.subplot(1,3,3)
plt.imshow(CeAn, vmin=0, vmax=210)#,  aspect='auto')
ax3 = plt.gca()
# Loop over data dimensions and create text annotations.
for i in range(dim):
    for j in range(dim):
        if i==j:
            text = ax3.text(j, i, int(CeAn[i, j]),
                           ha="center", va="center", color="k", size=16-dim)
plt.title('analytic')

import matplotlib
cax,kw = matplotlib.colorbar.make_axes([ax1, ax2, ax3], location='bottom', orientation='horizontal', fraction=0.15, shrink=1.0, aspect=20)
plt.colorbar(im, cax=cax, **kw)
plt.savefig(savepic+"Brownian_motion_covariance_%dD_T_2_histogram_sigma%d_N_%d.png"%(dim, g, N), bbox_inches='tight', transparent='False',  facecolor='white',dpi=300)
#%%