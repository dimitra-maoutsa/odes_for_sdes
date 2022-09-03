# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:10:54 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from odeintw import odeintw

x0 = np.array([0.5, 0.5])
f =  lambda x,t: np.array([-4*x[0]**1+1*x[1] , -4*x[1]**1+1*x[0]])
#xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N) 
def f_var(C,t):
    
    A = np.array([[-4, 1],[1,-4]])
    
    return A@C + C@A.T + 1*np.eye(2,2)

h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
C_0 = np.array([[0.25**2,0],[0,0.25**2]])

m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_0,timegrid)

plt.figure(),
plt.subplot(2,1,1)
plt.plot(timegrid, m_t[:,0],'r')
plt.plot(timegrid, m_t[:,0]+ np.sqrt(C_t[:,0,0]),'k--')
plt.plot(timegrid, m_t[:,0]- np.sqrt(C_t[:,0,0]),'k--')

plt.subplot(2,1,2)
plt.plot(timegrid, m_t[:,1])
plt.plot(timegrid, m_t[:,1]+ np.sqrt(C_t[:,1,1]),'k--')
plt.plot(timegrid, m_t[:,1]- np.sqrt(C_t[:,1,1]),'k--')
plt.xlabel('time')
#plt.ylabel(u'$\langle x \rangle$')



####Sample from 2d gaussian
sampls = np.arange(100,3000,200)
mean_norm = np.zeros((len(sampls),len(timegrid) ))
C_norm = np.zeros((len(sampls),len(timegrid) ))
for ni,n_sampls in enumerate(sampls):
    print(n_sampls)
    #n_sampls = 500
    AF = np.zeros((2,n_sampls,timegrid.size))
    for ti,t in enumerate(timegrid):
        # Define epsilon.
        epsilon = 0.0001
        # Add small pertturbation. 
        K = C_t[ti] + epsilon*np.identity(2)    
        ### or 
        AF[:,:,ti] = np.random.multivariate_normal(mean=m_t[ti].reshape(2,), cov=K, size=n_sampls).T
        mean_norm[ni,ti] = np.linalg.norm(np.mean(AF[:,:,ti],axis=1) - m_t[ti])
        C_norm[ni,ti] = np.linalg.norm(np.cov(AF[:,:,ti])-C_t[ti])
        
     
          
    
plt.figure()
plt.subplot(2,1,1)
plt.boxplot(mean_norm.T)
plt.subplot(2,1,2)
plt.boxplot(C_norm.T)

plt.figure(),
plt.subplot(2,1,1)
plt.plot(sampls,np.mean(mean_norm,axis=1))
plt.plot(sampls,np.mean(mean_norm,axis=1)+np.std(mean_norm,axis=1) )
plt.plot(sampls,np.mean(mean_norm,axis=1)-np.std(mean_norm,axis=1) )
plt.subplot(2,1,2)
plt.plot(sampls,np.mean(C_norm,axis=1))
plt.plot(sampls,np.mean(C_norm,axis=1)+np.std(C_norm,axis=1) )
plt.plot(sampls,np.mean(C_norm,axis=1)-np.std(C_norm,axis=1) )


###for the 2dim OU the optimal sampling is 2000








##  Cholesky decomposition.
#L = np.linalg.cholesky(K)
##random normal samples
## Number of samples. 
#n = 1000
#u = np.random.normal(loc=0, scale=1, size=2*n).reshape(2, n)
#
#x = mean_2 + np.dot(L, u)
