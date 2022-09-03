# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:54:59 2020

@author: Dimi
"""
import numpy as np
from matplotlib import pyplot as plt
from score_function_sparse import score_function_sparse
import copy
#simulation precision    
sim_prec = 0.001
t_start = 0.
#t_end = 1
#initial condition
x0 = -1.0
se = 21
np.random.seed(se)
#sampling precision
obs_dens = sim_prec
#observation noise amplitude
obs_noise = 0.000000001
num_ind = 50
T=3

f = lambda x,t: 4*x-4*x*x*x#-np.cos(x)#
#f = lambda x,t: 9*x-4*x*x*x
N = 2000#50  #number of particles
#diffission coefficient/fucntion
Ninf = 6000
g = 1.
gii = lambda x:g*np.sin(x)
xs=np.random.normal(loc=x0, scale=0.05,size=Ninf)
grads = [0]

C = 0.001

#def f_kde(x,t):
#    gpsi = score_kde(x)
#    
#    return (f(x,t)-0.5*g**2 * gpsi.reshape(-1,)+np.random.normal(loc=0,scale=0.0001,size=x.size))

def f_sparse(x,t,Z=None):
    if (not Z.all):        
        Z = np.linspace(np.min(x),np.max(x),round(x.size/10))
    gpsi= score_function_sparse(x,Z,C=C,l=0.25,funct_out=False) #C=1/x.size
    grads.append(np.linalg.norm(gpsi)/N)
    return (f(x,t)-0.5*g**2 * gpsi.reshape(-1,) -0.5*g**2*2*np.sin(x)*np.cos(x))



h = sim_prec #step
timegrid = np.arange(0,T,h)
Z = np.zeros((Ninf,timegrid.size))
F = np.zeros((N,timegrid.size))
#G = np.zeros((N,timegrid.size))
M = np.zeros((N,timegrid.size))

inducing = np.random.uniform(low=-1,high=1,size=num_ind)
for ti,tt in enumerate(timegrid[:]):
    print(tt)
    if ti == 0:
        Z[:,ti] = copy.deepcopy(xs)
        F[:,ti] = copy.deepcopy(xs[:N])
        M[:,ti] = copy.deepcopy(xs[:N])
    else:
        #Z[:,ti] = Z[:,ti-1] + h*f_effD(Z[:,ti-1],tt)
        #inducing = np.random.uniform(low=np.min(M[:,ti-1]),high=np.max(M[:,ti-1]),size=num_ind)
        
        
        M[:,ti] = M[:,ti-1] + h*f_sparse(M[:,ti-1],tt,inducing)#+ 0.001*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=N)
        
        F[:,ti] = F[:,ti-1] + h* f(F[:,ti-1],ti) + np.multiply(gii(F[:,ti-1]),np.random.normal(loc = 0.0, scale = np.sqrt(h),size=N))
        Z[:,ti] = Z[:,ti-1] + h* f(Z[:,ti-1],ti) + np.multiply(gii(Z[:,ti-1]),np.random.normal(loc = 0.0, scale = np.sqrt(h),size=Ninf))
         
    #F[j,:] = sdeint.itoint(f, gii, xs[j], timegrid)[:,0]
from plot_statistics import plot_statistics
plot_statistics(timegrid, [Z,F,M])

#def KL(a, b):
#    """
#    KL divergence between two distributions/histograms
#    """
#    a = np.asarray(a, dtype=np.float)
#    b = np.asarray(b, dtype=np.float)
#
#    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
#
#
#def KL_gauss(P,Q):
#    """
#    P and Q are samples from gaussian distributions
#    """
#    m1 = np.mean(P)
#    s1 = np.std(P)
#    
#    m2 = np.mean(Q)
#    s2 = np.std(Q)
#    
#    return np.log(s2/s1) + (s1**2+(m1-m2)**2)/(2*s2**2)-0.5
#from scipy.stats  import entropy
#min_F = min(np.min(M[:,0]),np.min(M[:,-1]))
#max_F = max(np.max(M[:,0]),np.max(M[:,-1]))
#p_inf,bins = np.histogram(M[:,-1],100,range=(min_F,max_F))
#p_0,bins = np.histogram(M[:,0],bins=bins)
#KL_0_inf = KL(p_0,p_inf)
#KL_0_infa = KL_gauss(M[:,0],M[:,-1])
#print(KL_0_inf)
#print(KL_0_infa)
#plt.figure(),
#plt.plot(timegrid[1:],KL_0_infa-(2/g**2)*np.array(grads[1:]))    
#plt.yscale('log')
#plt.xscale('log')
#
#
#
#plt.figure(),
#plt.plot(timegrid[1:],(2/g**2)*np.array(grads[1:]))    
#plt.yscale('log')
#plt.xscale('log')
#
#plt.figure(),
#plt.plot(timegrid[1:],KL_0_infa-(2/g**2)*np.cumsum(np.array(grads[1:])) )   
#plt.yscale('log')
#plt.xscale('log')
