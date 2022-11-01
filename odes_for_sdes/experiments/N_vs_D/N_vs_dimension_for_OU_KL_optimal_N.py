# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:34:09 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
#import ot
from score_function_multid_seperate import score_function_multid_seperate
import scipy as sc
import pandas as pd
from scipy.integrate import odeint
from odeintw import odeintw

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'


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


dim =6#[2,3,4,5,6]
Ns = np.arange(200,7000,200)
trials = 20
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
g = 1
M = 100



kll = np.zeros((len(timegrid),trials,Ns.size))
threshold = 0.005
be_th = np.zeros((len(timegrid),trials))
be_th.fill(np.nan)

print(dim)
x0 = np.ones(dim)*0.5
C_0 = np.zeros((dim,dim))
np.fill_diagonal(C_0,0.25**2)

means = np.zeros((dim,len(timegrid),trials,Ns.size))
covs = np.zeros((dim,dim,len(timegrid),trials,Ns.size))
####### f # multidim OU
def f(x,t):        
    ret = np.ones((dim,dim))
    np.fill_diagonal(ret, -4)        
    return ret@x

########## Otto dynamics
def f_seperate(x,t,N_sparse=100):
    dimi, N = x.shape    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]    
    Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )    
    gpsi = np.zeros((dimi, N))
    lnthsc = 2*np.std(x,axis=1)    
    for ii in range(dimi):        
        gpsi[ii,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=ii+1)[0] #C=0.1 !!!!!
    return (f(x,t)-0.5*g**2* gpsi)
##############
    
def f_var(C,t):    
        A = np.ones((dim,dim))
        np.fill_diagonal(A, -4)    
        return A@C + C@A.T + 1*np.eye(dim,dim)

#m1 = np.array(x0)
#S1 = np.array(C_0)    
flag = np.zeros(trials)

ite = reversed(list(enumerate(Ns)))
for ss,N in ite:
    
    print('Sample size: %d'%N)
    
    ### simulate analytical moments
    #integrate
    m_t = odeint(f, x0, timegrid)
    C_t = odeintw(f_var, C_0,timegrid)
    
    
    for i in range(trials):
        print('Dimension: %d, particles: %d, seed:%d'%(dim,N,i))
        np.random.seed(i)
        ##simulated trajectory
        initial = np.random.multivariate_normal(x0, C_0, N)    
        D = np.zeros((dim,N,timegrid.size))
        for ti,t in enumerate(timegrid):
            if ti==0: 
                
                D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
            else:
                D[:,:,ti] = D[:,:,ti-1] + h* f_seperate(D[:,:,ti-1],t,M)
                    
            ### calculate KL for every timepoint
            means[:,ti,i,ss] = np.mean(D[:,:,ti],axis=1)
            covs[:,:,ti,i,ss] = np.cov(D[:,:,ti])
            kll[ti,i,ss] = KL(m_t[ti], means[:,ti,i,ss] , C_t[ti], covs[:,:,ti,i,ss] )
        
        
        
        
        
        
        
    
        if flag[i]==0 and (kll[:,i,ss]<threshold).all():
            
            flag[i] = 1
            be_th[i] = N

    

    filenm = 'N_vs_dims_for_OU_with_KL_opt_N_for_M_%d_for_dim_%d_end'%(M,dim)
    
    Dic = dict()
    Dic['Ns'] = Ns
    Dic['dim'] = dim
    Dic['M'] = M
    Dic['means'] = means
    Dic['covs'] = covs
    Dic['kll'] = kll
    Dic['m_t'] = m_t
    Dic['C_t'] = C_t
    Dic['timegrid'] = timegrid
    Dic['T'] = T
    Dic['h'] = h
    Dic['g'] = g
    Dic['trials'] = trials
    Dic['be_th'] = be_th
    Dic['threshold'] = threshold
    
    joblib.dump(Dic, filename= save_file+filenm)

    print(np.sum(flag))
    print('-------')