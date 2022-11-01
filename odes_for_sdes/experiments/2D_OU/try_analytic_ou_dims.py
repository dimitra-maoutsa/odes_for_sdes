# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:27:11 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

from odeintw import odeintw

#plt.figure()
#for dim in [2,3,4,5,6]:
#    Ns = np.arange(500,7000,500)
#    trials = 20
#    h = 0.001
#    t_start = 0
#    T = 15
#    timegrid = np.arange(0,T,h)
#    g = 1
#    M = 100
#    
#    print(dim)
#    x0 = np.ones(dim)*0.5
#    C_0 = np.zeros((dim,dim))
#    np.fill_diagonal(C_0,0.25**2)
#    
#    def f(x,t):        
#        ret = np.ones((dim,dim))*1
#        np.fill_diagonal(ret, -4)        
#        return ret@x
#    
#    
#    def f_var(C,t):    
#        A = np.ones((dim,dim))*1
#        np.fill_diagonal(A, -4)    
#        return A@C + C@A.T + 1*np.eye(dim,dim)
#    
#    
#    
#    m_t = odeint(f, x0, timegrid)
#    
#    
#    plt.plot(timegrid[:-1],np.diff(m_t[:,0]),label=dim)
#    print(timegrid[np.where(np.abs(m_t[:,0])<0.00001)[0][0]])
#plt.legend()
#




def analytic_OU(x0,C_0,timegrid):
    dim = x0.shape[0]
    def f(x,t):        
        ret = np.ones((dim,dim))
        np.fill_diagonal(ret, -4)        
        return ret@x
            
    def f_var(C,t):    
        A = np.ones((dim,dim))
        np.fill_diagonal(A, -4)    
        return A@C + C@A.T + 1*np.eye(dim,dim)

    #initial conditions
    #x0 = np.ones(dim)*0.5
    #C_0 = np.zeros((dim,dim))
    #np.fill_diagonal(C_0,0.25**2)    
    #integrate
    m_t = odeint(f, x0, timegrid)
    C_t = odeintw(f_var, C_0,timegrid)
    
    return(m_t,C_t)

