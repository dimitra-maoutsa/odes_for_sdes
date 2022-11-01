#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:18:54 2020

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from scipy.integrate import odeint
from odeintw import odeintw
from sklearn.metrics import mean_squared_error
from math import sqrt

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'
M= 200
seeds = np.arange(10, 30,1)
dims = [1,2,3,4,5,6]
Ns = [1000,2000,3000,4000,5000,6000]
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)

mean_norm = np.zeros((len(dims),len(Ns),timegrid.size))
C_norm =  np.zeros((len(dims),len(Ns),timegrid.size))
mean_norm2 = np.zeros((len(dims),len(Ns)))
C_norm2 =  np.zeros((len(dims),len(Ns)))
x0 = 0.5
for di,dim in enumerate(dims):
    

    ####### f # multidim OU
    def f(x,t):        
        ret = np.ones((dim,dim))
        np.fill_diagonal(ret, -4)        
        return ret@x
    
    
    def f_var(C,t):    
        A = np.ones((dim,dim))
        np.fill_diagonal(A, -4)    
        return A@C + C@A.T + 1*np.eye(dim,dim)

    #initial conditions
    x0 = np.ones(dim)*0.5
    C_0 = np.zeros((dim,dim))
    np.fill_diagonal(C_0,0.25**2) 
    
    m_t = odeint(f, x0, timegrid)
    C_t = odeintw(f_var, C_0,timegrid)
    
    for ni,N in enumerate(Ns):
    
        D = joblib.load(save_file+'OU_%d_D_Deterministic_Trajectories_N_%d_M_%d'%(dim,N,M))
        cov_t = np.zeros((dim,dim,timegrid.size))
        for ti,t in enumerate(timegrid):
            mean_norm[di,ni,ti] = np.linalg.norm(np.mean(D[:,:,ti],axis=1) - m_t[ti])
            C_norm[di,ni,ti] = np.linalg.norm(np.cov(D[:,:,ti])-C_t[ti])
            cov_t[:,:,ti]  = np.cov(D[:,:,ti])
        mean_norm2[di,ni] = sqrt(mean_squared_error(np.mean(D,axis=1),m_t.T))   
        C_norm2[di,ni] = sqrt(mean_squared_error(cov_t.flatten(),C_t.T.flatten())) 
            
            
            
            
plt.figure(),

for ni,N in enumerate(Ns):
    if N in [1000,3000,5000]:
        plt.subplot(2,1,1)
        plt.plot(dims,np.mean(mean_norm[:,ni],axis=1)/np.sqrt(dims),'o',label=N)
        #plt.plot(sampls,np.mean(mean_norm,axis=1)+np.std(mean_norm,axis=1),'k' )
        #plt.plot(sampls,np.mean(mean_norm,axis=1)-np.std(mean_norm,axis=1) ,'k')
        plt.ylabel(r'$\| \hat{m_t}-m_t  \|$')
        plt.yscale('log')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(dims,np.mean(C_norm[:,ni],axis=1),'o',label=N)
        #plt.plot(sampls,np.mean(C_norm,axis=1)+np.std(C_norm,axis=1) ,'k')
        #plt.plot(sampls,np.mean(C_norm,axis=1)-np.std(C_norm,axis=1),'k' )
        plt.ylabel(r'$\| \hat{C_t}-C_t  \|$')
        plt.xlabel(r'$N$')
        plt.yscale('log')
        plt.legend()
plt.tight_layout()


plt.figure(),

for ni,N in enumerate(Ns):
    
    plt.subplot(2,1,1)
    plt.plot(dims,mean_norm2[:,ni],'o',label=N)
    #plt.plot(sampls,np.mean(mean_norm,axis=1)+np.std(mean_norm,axis=1),'k' )
    #plt.plot(sampls,np.mean(mean_norm,axis=1)-np.std(mean_norm,axis=1) ,'k')
    plt.ylabel(r'$\| \hat{m_t}-m_t  \|$')
    plt.yscale('log')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(dims,C_norm2[:,ni],'o',label=N)
    #plt.plot(sampls,np.mean(C_norm,axis=1)+np.std(C_norm,axis=1) ,'k')
    #plt.plot(sampls,np.mean(C_norm,axis=1)-np.std(C_norm,axis=1),'k' )
    plt.ylabel(r'$\| \hat{C_t}-C_t  \|$')
    plt.xlabel(r'$N$')
    plt.yscale('log')
    plt.legend()
plt.tight_layout()
