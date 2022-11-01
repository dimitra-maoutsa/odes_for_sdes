#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:32:54 2019

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from numpy.linalg import cholesky, det, lstsq


def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

def score_function_sparse(X, Z,C=1,l=1,funct_out=False,kernel ='RBF'):
    
    """
    Calculates the gradient log density of the density samples X, by employing a sparse approximation
    Input: X: training points
           Z: inducing points
           l: kernel lengthscale
           funct_out: True/False = output function/ evaluation of function at training points
           kernel: 'RBF' or 'polyRBF'
           
           
           
   Output: 
    
    """
    X = np.atleast_2d(X).T
    Z = np.atleast_2d(Z).T
    if kernel ==  'polyRBF':
        def K(x,y,l=1,b=0.001):
            r = cdist(x, y,'sqeuclidean')
            diffs = x[:,None]-y 
            
            return np.exp(-r/(2*l*l)) + b*np.power(diffs,3)[:,:,0]
        
        def grdx_K(x,y,l=1,b=0.001):
            N,dim = x.shape
            M,dim = y.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((dim*N,M))
            for ii in range(dim):  
                redifs[ii*N:(ii+1)*N] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs + 3*b*np.power(diffs,2)[:,:,0]
            
    elif kernel == 'RBF':        
        def K(x,y,l):
            return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
        
        def grdx_K(x,y,l):
            N,dim = x.shape
            M,dim = y.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((dim*N,M))
            for ii in range(dim):  
                redifs[ii*N:(ii+1)*N] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
        
        def grdx_K2(x,y,l):
            N,dim = x.shape 
            M,dim = y.shape                     
            redifs = np.zeros((dim*N,M))
            Ki = K(x,y)
            for i in range(N):
                for j in range(N):                                    
                    redifs[i,j] = (x[i,0]-y[j,0])*Ki[i,j] /(l*l)  
                    redifs[N+i,j] = (x[i,1]-y[j,1])*Ki[i,j]/(l*l)  
            return redifs
    
    Ks = K(Z,Z,l)
    
    #print(np.linalg.cond(Ks))
    #plt.figure(),sns.distplot(X,norm_hist =True)
    #Ks = low_rank_approx(A=Ks,r=5)
    
    #L = cholesky(Ks+ 1e-8 * np.eye(Z.size))
    
    Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.size))
    K_xz = K(X,Z,l)    
    A =K_xz.T @ K_xz    
    #A = low_rank_approx(A=A,r=5)
    gradx_K = np.sum(grdx_K(X,Z,l),axis=0)#.T
    
    if funct_out:
        ks = lambda x: K(x,Z,l)
        
        #result =lambda x: -ks(x) @ np.linalg.inv( C*np.eye(Z.shape, Z.shape) + Ks.T )@ np.linalg.inv(Ks)@gradx_K
        #print(np.linalg.cond( C*np.eye(Z.size, Z.size) + Ksinv @ A))
        result =lambda x: -ks(x) @ np.linalg.inv( C*np.eye(Z.size, Z.size) + Ksinv @ A + 1e-3 * np.eye(Z.size))@ Ksinv @ gradx_K
    
    else:
        Ksx = K(X,Z,l)
        #result =-Ksx @ np.linalg.inv( C*np.eye(Z.size, Z.size) + Ks.T )@ np.linalg.inv(Ks)@gradx_K
        #KsA = lstsq(L.T, lstsq(L,A,rcond=None)[0],rcond=None)[0]
#        KsB = Ksinv @ A
##        print(np.all(np.linalg.eigvals(KsB) > 0))
##        print(np.allclose(KsB,KsB.T))
#        e_vals, e_vect = np.linalg.eig(KsA)
#        indx = np.where(e_vals<0)[0]
#        if len(indx)>0:
#            e_vals[indx] = 1e-8
#        #print(e_vals)
#        KsA = e_vect@np.diag(e_vals)@np.linalg.inv(e_vect)
#        print(np.all(np.linalg.eigvals(KsA) > 0))
        #print(np.allclose(KsA,KsA.T))
##        print('there')
#        #print(Ks)
#        plt.imshow(KsA)
#        cholesky(KsA)
#        #print(np.linalg.eig(C*np.eye(Z.size, Z.size) + KsA+ 1e-7 * np.eye(Z.size)))
#        L2 = cholesky(C*np.eye(Z.size, Z.size) + KsA+ 1e-8 * np.eye(Z.size))
#        a = lstsq(L2.T, lstsq(L2,Ksinv@gradx_K ,rcond=None)[0],rcond=None)[0]
#        print(a.shape)
#        result2 = -Ksx @ a
#        print((np.linalg.inv( C*np.eye(Z.size, Z.size) + Ksinv @ A )@ Ksinv@gradx_K).shape)
#        print(Ksx.shape)
        #print(np.linalg.cond(C*np.eye(Z.size, Z.size) + Ksinv @ A + 1e-8 * np.eye(Z.size)))
        result =Ksx @ np.linalg.inv( C*np.eye(Z.size, Z.size) + Ksinv @ A + 1e-3 * np.eye(Z.size))@ Ksinv@gradx_K #full
        
        ##result = gradx_K.T @ np.linalg.inv( C*np.eye(Z.size, Z.size) + Ksinv@A  + 1e-8 * np.eye(Z.size))
        #result =-Ksx @ np.linalg.inv( C*np.eye(Z.size, Z.size) + Ks.T )@ Ksinv@gradx_K
        
        
    return (result)



if __name__ == '__main__':
    
    X2 = np.sort(np.random.normal(loc=1,scale=1,size=1500))
    plt.figure(),
    zi = 500
    #for ii,zi in enumerate([10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300]):
    for ii,Ci in enumerate([1 ]):
        Z = np.linspace(np.min(X2), np.max(X2), zi )
        
        
        ln0 = score_function_sparse(X2.T, Z, C=Ci,l=5,funct_out=False ,kernel='RBF')
        
        plt.subplot(4,5,ii+1)
        plt.plot(X2, -(X2-1)/1**2,'k',lw=1)
        plt.plot(X2,ln0,'r.',label='sparse',markersize=5)
    