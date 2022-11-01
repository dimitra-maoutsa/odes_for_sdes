# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:14:37 2020

@author: Dimi
"""

import numpy as np
import copy
import joblib
from try_analytic_ou_dims import analytic_OU
from scipy.spatial.distance import cdist

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
h = 0.001
t_start = 0
T = 1#3
dim = 2
x0 = np.array([0.5, 0.5])
#f =  lambda x: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
f =  lambda x: np.array([-4*x[0]**1 , -4*x[1]**1])
timegrid = np.arange(0,T,h)
g = 1
Ns = [ 1000]# 500,1000,1500,2000, 2500]
#Ms = [ 20, 40,60,80,100,120,140,160,180,200]
#Ms = [50,100,150,200]
reps = 1#20

C = 0.001
gii = np.multiply(np.ones(dim),g)
C_0 = np.zeros((dim,dim))
np.fill_diagonal(C_0,0.25**2)
m_t,C_t = analytic_OU(x0,C_0,timegrid)

def grad_station(x):
    return -np.linalg.inv(C_t[-1])@(x.T-m_t[-1]).T

#def my_kde(s,multil=False):
#    
#    ##### Kernel
#    def K(x,y,l,multil=False):
#        if multil:                
#            res = np.ones((x.shape[0],y.shape[0]))                
#            for ii in range(len(l)): 
#                res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
#            return res
#        else:
#            return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
#            
#    def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
#            N,dim = x.shape            
#            diffs = x[:,None]-y   
#            
#            redifs = np.zeros((1*N,N))
#            ii = which_dim -1
#            #print(ii)
#            if multil:
#                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
#            else:
#                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
#            return redifs
#    
#    ###########
#    
#    l = 2*np.std(s,axis=1)
#    
#    di,N = s.shape
#    gradlog = np.zeros(s.shape)
#    #print(gradlog.shape)
#    if multil:
#        #fx = np.average(K(s,s,l=l,multil=multil),weights=l,axis=1)
#        for ii in range(di):
#            
#            #print(np.average(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil)/K(s.T,s.T,l=l,multil=multil),weights=l[ii],axis=1).shape)
#            gradlog[ii,:] = np.average(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil)/K(s.T,s.T,l=l,multil=multil),weights=[l[ii]]*N,axis=1)
#    else:
#        #fx = np.mean(K(s,s,l=l,multil=multil),axis=1)/l
#        
#        ltot = np.copy(l)
#        for ii in range(di):
#            l = ltot[ii]
#            gradlog[ii,:]=(1/l)* np.mean(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil),axis=1)/np.mean(K(s.T,s.T,l=l,multil=multil),axis=1)
#        
#    return (f(s)-0.5*  g*gradlog)  

######
    
def kde_S(s,multil=False):
    
    
    ##### Kernel
    def K(x,y,l,multil=False):
        if multil:                
            res = np.ones((x.shape[0],y.shape[0]))                
            for ii in range(len(l)): 
                res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
            return res
        else:
            return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
            
    def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape            
            diffs = x[:,None]-y   
            
            redifs = np.zeros((1*N,N))
            ii = which_dim -1
            #print(ii)
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
    
    #l = 2*np.std(s,axis=1)
    l = np.std(s,axis=1)*(4/3/(s.shape[1]))**(1/5) #Bandwidth estimated by Silverman's Rule
    #l=[0.07, 0.07]#[1,1]
    di,N = s.shape
    gradlog = np.zeros(s.shape)
    gradlogsum = np.zeros(s.shape)
    #print(gradlog.shape)
    if multil:
        #fx = np.average(K(s,s,l=l,multil=multil),weights=l,axis=1)
        sumK = np.sum(K(s.T,s.T,l=l,multil=multil),axis=0)
        for ii in range(di):
            
            gradlog[ii,:]= np.mean(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil),axis=1)/np.mean(K(s.T,s.T,l=l,multil=multil),axis=1)
            gradlogsum[ii,:] = np.sum(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil)/sumK,axis=1)


    else:
        #fx = np.mean(K(s,s,l=l,multil=multil),axis=1)/l
        
        #ltot = np.copy(l)
        sumK = np.sum(K(s.T,s.T,l=l[0],multil=multil),axis=0)
        for ii in range(di):
            #l = ltot[ii]
            gradlog[ii,:]= np.mean(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil),axis=1)/np.mean(K(s.T,s.T,l=l,multil=multil),axis=1)
            gradlogsum[ii,:] = np.sum(grdx_K(s.T,s.T,l=l,which_dim=ii+1,multil=multil)/sumK,axis=1)


    
    return (-  gradlog-gradlogsum+grad_station(s))  

###################################################################################
##start simulating   
    

for N in Ns:    
    
    S = np.zeros((dim,N,timegrid.size,reps))
    for repi in range(reps):
        print('Deterministic KDE 2D OU: N: %d, repetition: %d'%(N, repi))
        se = repi #setting a different seed for each repetition
        np.random.seed(se)
        xs = np.zeros((2,N))
        for ii in range(2):
            xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N)            
        for ti,tt in enumerate(timegrid): 
            
            if ti == 0:                
                S[:,:,ti,repi] = copy.deepcopy(xs)                
            else:                 
                #S[:,:,ti,repi] = S[:,:,ti-1,repi] + h* my_kde(S[:,:,ti-1,repi],False)#,True)
                S[:,:,ti,repi] = S[:,:,ti-1,repi] + h* kde_S(S[:,:,ti-1,repi],True)
                
            
    #joblib.dump(S,filename=save_file+'OU_2D_KDE_deterministic_trajectories%d_UPDATED_Sl'%(N))
from plot_statistics import plot_statistics
plot_statistics(timegrid,[S[:,:,:,0],S[:,:,:,0]],labelss=['x','y','z'])    
#  