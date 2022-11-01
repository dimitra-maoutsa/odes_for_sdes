#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:40:09 2020

@author: dimitra
"""


import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import pinv
from functools import reduce
from scipy.stats import gamma,norm,dweibull,tukeylambda,skewnorm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import time
### calculate score function from empirical distribution
### uses RBF kernel
### follows description of [Batz, Ruttor, Opper, 2016]

#Ktestsp = pdist2(xtrain',xsparse');
#Ktestsp= Ktestsp.^2/L^2;
#Ktestsp = exp(-Ktestsp);

def score_function_multid_seperate(X,Z,Test_p='None',T=1, C=0.25,kern ='RBF',p=4,l=1.,which=1,figs=False,which_dim=1):
    """
    returns function psi(z)
    Input: X: N observations
            Z: sparse points
            Test_p : If 'None', the density is evaluated at the observation points,
                    if array of N_s elements then the density is evaluatedat those N_s points
           f: function of known drift part
           C: weighting constant           
           which: return 1: grad log p(x) , 2: log p(x), 3:both
           which_dim: which gradient of log density we want to compute
    Output: psi: array with density along the given dimension N or N_s x 1
    
    """
    if kern=='RBF':
        #l = 1 # lengthscale of RBF kernel
        #@njit
        def K(x,y,l,multil=False):
            if multil:                
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
            #return np.exp(-(x-y.T)**2/(2*l*l))
            #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
        #@njit
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape            
            diffs = x[:,None]-y   
            #print(diffs.shape)
            redifs = np.zeros((1*N,N))
            ii = which_dim -1
            #print(ii)
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
            #return -(1./(l*l))*(x-y.T)*K(x,y)
            
        
            
            
        
        def grdy_K(x,y): # gradient with respect to the second argument
            N,dim = x.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            ii = which_dim -1              
            redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
            return -redifs
            #return (1./(l*l))*(x-y.T)*K(x,y)
        #@njit        
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs
            #return np.multiply((K(x,y)),(np.power(x[:,None]-y,2)-l**2))/l**4
        
        #@njit
        def ggrdxx_K(x,y,l):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            ii = which_dim-1
            jj = which_dim-1
            redifs = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y,l))/(l**4)
            return redifs
            
            #@njit        
        def ggrdyy_K(x,y):            
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4)
            return redifs
        #@njit
#        def gggrdxxy_K(x,y):
#            return (1./(l**6))*np.multiply(np.multiply((x[:,None]-y),( np.power( x[:,None]-y,2) -3*l*l )) , (K(x,y)))
        #@njit
        def gggrdxyy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,1)) 
            redifs[0:N,0] = (np.sum(np.multiply(np.multiply(diffs[:,:,0],(3*l*l+np.power(diffs[:,:,0],2))),K(x,y))/(2*l**6),axis=1) + \
                  np.sum(np.multiply(np.multiply(diffs[:,:,0],(l*l+np.power(diffs[:,:,1],2))),K(x,y))/(2*l**6),axis=1) ) 
#            redifs[N:2*N,0] = (np.sum(np.multiply(np.multiply(diffs[:,:,1],(3*l*l+np.power(diffs[:,:,1],2))),K(x,y))/(2*l**6),axis=1) + \
#                  np.sum(np.multiply(np.multiply(diffs[:,:,1],(l*l+np.power(diffs[:,:,0],2))),K(x,y))/(2*l**6),axis=1) )
#                
#            return redifs
        #@njit
#        def ggggrdxxyy_K(x,y):
#            #return K(x,y)*(1./l**6)*(3-6*(x-y)**2/l**2+(x-y)**4/l**4)
#            return (1/l**8)*np.multiply((K(x,y)),(3*l**4-6*l**2*np.power(x[:,None]-y,2) + np.power(x[:,None]-y,4)))

            
        
    #ii = which_dim -1
    
    
    if isinstance(l, (list, tuple, np.ndarray)):
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True)
       if not(Test_p == 'None'):
           K_sz = K(Test_p,Z,l,multil=True)
        
    else:
    
        K_xz = K(X,Z,l) 
        Ks = K(Z,Z,l)    
        #print(Z.shape)
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        gradx_K = -grdx_K(X,Z,l,which_dim=which_dim)
    sumgradx_K = np.sum(gradx_K ,axis=0)
    if Test_p == 'None':
        
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:   
        print('///////')
#        print(X.shape)
#        print(Z.shape)
#        print(Test_p.shape)
        res1 = -K_sz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
#    print(res1)
#    A2 = gradx_K.T @ gradx_K
#    gradxx_K = ggrdxx_K(Z,X,l)
#    gKs = grdx_K(Z,Z,l,which_dim=which_dim)
#    #print(gradxx_K.shape)
#    sumgradxx_K = np.sum(gradxx_K ,axis=1)
#    res2 =  -gradx_K @ np.linalg.inv( C*gKs +  A2 + 1e-3 * np.eye(Z.shape[0])) @sumgradxx_K 
    res2 = 0
    return(res1, res2)