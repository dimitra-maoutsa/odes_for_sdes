<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:01:04 2020

@author: Dimi
"""


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

#from score_function_multid_seperate import score_function_multid_seperate
import scipy as sc
#import pandas as pd
from scipy.integrate import odeint
from odeintw import odeintw
import time
#save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/'


dim =2
thresholds = {1:10**(-2), 2:5*10**(-3), 3:10**(-3),4:5*10**(-4),5:10**(-4) }
Nstart = {1:100,2:500,3:5000,4:10000,5:50000}
run_id = 4


i_start = 0
i_end = 20
trials = 20#20
h = 0.001
t_start = 0
T = 3.1
timegrid = np.arange(0,T,h)
g = 1
M = 100

x0 = np.ones(dim)*0.5
C_0 = np.zeros((dim,dim))
np.fill_diagonal(C_0,0.25**2)


####### f # multidim OU
def f(x,t):        
    ret = np.ones((dim,dim))*0.5
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
    A = np.ones((dim,dim))*0.5
    np.fill_diagonal(A, -4)    
    return A@C + C@A.T + 1*np.eye(dim,dim)

thr = thresholds[run_id]
Ns_start = Nstart[run_id]
Ns = np.arange(Ns_start,20000,10)

means = np.zeros((dim,len(timegrid)))
covs = np.zeros((dim,dim,len(timegrid)))
kll = np.zeros((len(timegrid)))
#Wasdis = np.zeros((len(timegrid),trials,Ns.size))
N_inf = 2000
### simulate analytical moments
#integrate
m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_0,timegrid)
#F_inf = np.zeros((dim,N_inf, timegrid.size))
#for ti,tt in enumerate(timegrid):
#    
#    F_inf[:,:,ti] = np.random.multivariate_normal(m_t[ti], C_t[ti], size=N_inf).T
flags1 = np.zeros(20)
flags2 = np.zeros(20)
flags3 = np.zeros(20)
N_star = np.zeros(20)

#N2 = dict()    
    
Ns = np.arange(Ns_start,5500000,1000)    
    
for i in range(i_start,i_end):
    print('Entering trial %d: '%i)
    
    for ss,N in enumerate(Ns):
        if flags1[i] ==0:
    
            print('Sample size 1st loop: %d'%N)
            startime = time.time()
            #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
            np.random.seed(i)
            ##simulated trajectory
            initial = np.random.multivariate_normal(x0, C_0, N)    
            D = np.zeros((dim,N,2))
            for ti,t in enumerate(timegrid):
                if ti==0: 
                    
                    D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                else:
                    D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N))
                    D[:,:,0] = D[:,:,1]   
                    ### calculate KL for every timepoint
                means[:,ti] = np.mean(D[:,:,0],axis=1)
                #covs[:,:,ti] = np.cov(D[:,:,0])
                kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
    #            ##wasserstein distance
    #            xs = F_inf[:,:,ti] 
    #            xt = D[:,:,ti]  #index i denotes repetition
    #            # loss matrix
    #            Mdist = ot.dist(xs.T, xt.T)
    #            Mdist /= Mdist.max()
    #            a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
    #            Wasdis[ti,i,ss] = ot.emd2(a,b,Mdist)
                
                #print('KL: %.7f and Wasserstein %.7f' %(kll[ti,i,ss],Wasdis[ti,i,ss]))
            if np.std(kll)<thr:
                flags1[i] = 1
                N_star[i] = N
                
                Ns2 = np.arange(N-1000,N+100,100)   
                
                for ss,N2 in enumerate(Ns2):
                    if flags2[i] ==0:
                
                        print('Sample size 2nd loop: %d'%N2)
                        startime = time.time()
                        #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
                        np.random.seed(i)
                        ##simulated trajectory
                        initial = np.random.multivariate_normal(x0, C_0, N2)    
                        D = np.zeros((dim,N2,2))
                        for ti,t in enumerate(timegrid):
                            if ti==0: 
                                
                                D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                            else:
                                D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N2))
                                D[:,:,0] = D[:,:,1]    
                            ### calculate KL for every timepoint
                            means[:,ti] = np.mean(D[:,:,0],axis=1)
                            #covs[:,:,ti] = np.cov(D[:,:,0])
                            kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
                        if np.std(kll)<thr:
                            flags2[i] = 1
                            N_star[i] = N2
                            Ns3 = np.arange(N2-100,N2+10,10)
                            
                            for ss,N3 in enumerate(Ns3):
                                if flags3[i] ==0:
                            
                                    print('Sample size: %d'%N3)
                                    startime = time.time()
                                    #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
                                    np.random.seed(i)
                                    ##simulated trajectory
                                    initial = np.random.multivariate_normal(x0, C_0, N3)    
                                    D = np.zeros((dim,N3,2))
                                    for ti,t in enumerate(timegrid):
                                        if ti==0: 
                                            
                                            D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                                        else:
                                            D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N3))
                                            D[:,:,0] = D[:,:,1]    
                                        ### calculate KL for every timepoint
                                        means[:,ti] = np.mean(D[:,:,0],axis=1)
                                        #covs[:,:,ti] = np.cov(D[:,:,0])
                                        kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
                                    if np.std(kll)<thr:
                                        flags3[i] = 1
                                        N_star[i] = N3
                            
                                        filenm = 'Comp_complexity_particle_numberStoch_MEAN_FLUCTUATIONS_OU%d_trial_%d'%(run_id, i)

                                        Dic = dict()
                                        Dic['Ns'] = Ns
                                        Dic['dim'] = dim
                                        Dic['M'] = M
                                        
                                        Dic['N_star'] = N_star
                                        #Dic['kll'] = kll            
                                        Dic['timegrid'] = timegrid
                                        Dic['T'] = T
                                        Dic['h'] = h
                                        Dic['g'] = g
                                        Dic['trials'] = trials
                                        
                                        
                                        joblib.dump(Dic, filename= save_file+filenm)
                                        break;
                                else:
                                    break;
                    else:
                        break;
        else:
            break
                
    
    
    

        
        
        
        
        
=======
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:01:04 2020

@author: Dimi
"""


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

#from score_function_multid_seperate import score_function_multid_seperate
import scipy as sc
#import pandas as pd
from scipy.integrate import odeint
from odeintw import odeintw
import time
#save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/'


dim =2
thresholds = {1:10**(-2), 2:5*10**(-3), 3:10**(-3),4:5*10**(-4),5:10**(-4) }
Nstart = {1:100,2:500,3:5000,4:10000,5:50000}
run_id = 4


i_start = 0
i_end = 20
trials = 20#20
h = 0.001
t_start = 0
T = 3.1
timegrid = np.arange(0,T,h)
g = 1
M = 100

x0 = np.ones(dim)*0.5
C_0 = np.zeros((dim,dim))
np.fill_diagonal(C_0,0.25**2)


####### f # multidim OU
def f(x,t):        
    ret = np.ones((dim,dim))*0.5
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
    A = np.ones((dim,dim))*0.5
    np.fill_diagonal(A, -4)    
    return A@C + C@A.T + 1*np.eye(dim,dim)

thr = thresholds[run_id]
Ns_start = Nstart[run_id]
Ns = np.arange(Ns_start,20000,10)

means = np.zeros((dim,len(timegrid)))
covs = np.zeros((dim,dim,len(timegrid)))
kll = np.zeros((len(timegrid)))
#Wasdis = np.zeros((len(timegrid),trials,Ns.size))
N_inf = 2000
### simulate analytical moments
#integrate
m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_0,timegrid)
#F_inf = np.zeros((dim,N_inf, timegrid.size))
#for ti,tt in enumerate(timegrid):
#    
#    F_inf[:,:,ti] = np.random.multivariate_normal(m_t[ti], C_t[ti], size=N_inf).T
flags1 = np.zeros(20)
flags2 = np.zeros(20)
flags3 = np.zeros(20)
N_star = np.zeros(20)

#N2 = dict()    
    
Ns = np.arange(Ns_start,5500000,1000)    
    
for i in range(i_start,i_end):
    print('Entering trial %d: '%i)
    
    for ss,N in enumerate(Ns):
        if flags1[i] ==0:
    
            print('Sample size 1st loop: %d'%N)
            startime = time.time()
            #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
            np.random.seed(i)
            ##simulated trajectory
            initial = np.random.multivariate_normal(x0, C_0, N)    
            D = np.zeros((dim,N,2))
            for ti,t in enumerate(timegrid):
                if ti==0: 
                    
                    D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                else:
                    D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N))
                    D[:,:,0] = D[:,:,1]   
                    ### calculate KL for every timepoint
                means[:,ti] = np.mean(D[:,:,0],axis=1)
                #covs[:,:,ti] = np.cov(D[:,:,0])
                kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
    #            ##wasserstein distance
    #            xs = F_inf[:,:,ti] 
    #            xt = D[:,:,ti]  #index i denotes repetition
    #            # loss matrix
    #            Mdist = ot.dist(xs.T, xt.T)
    #            Mdist /= Mdist.max()
    #            a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
    #            Wasdis[ti,i,ss] = ot.emd2(a,b,Mdist)
                
                #print('KL: %.7f and Wasserstein %.7f' %(kll[ti,i,ss],Wasdis[ti,i,ss]))
            if np.std(kll)<thr:
                flags1[i] = 1
                N_star[i] = N
                
                Ns2 = np.arange(N-1000,N+100,100)   
                
                for ss,N2 in enumerate(Ns2):
                    if flags2[i] ==0:
                
                        print('Sample size 2nd loop: %d'%N2)
                        startime = time.time()
                        #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
                        np.random.seed(i)
                        ##simulated trajectory
                        initial = np.random.multivariate_normal(x0, C_0, N2)    
                        D = np.zeros((dim,N2,2))
                        for ti,t in enumerate(timegrid):
                            if ti==0: 
                                
                                D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                            else:
                                D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N2))
                                D[:,:,0] = D[:,:,1]    
                            ### calculate KL for every timepoint
                            means[:,ti] = np.mean(D[:,:,0],axis=1)
                            #covs[:,:,ti] = np.cov(D[:,:,0])
                            kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
                        if np.std(kll)<thr:
                            flags2[i] = 1
                            N_star[i] = N2
                            Ns3 = np.arange(N2-100,N2+10,10)
                            
                            for ss,N3 in enumerate(Ns3):
                                if flags3[i] ==0:
                            
                                    print('Sample size: %d'%N3)
                                    startime = time.time()
                                    #print('Dimension Few: %d, particles: %d, seed:%d'%(dim,N,i))
                                    np.random.seed(i)
                                    ##simulated trajectory
                                    initial = np.random.multivariate_normal(x0, C_0, N3)    
                                    D = np.zeros((dim,N3,2))
                                    for ti,t in enumerate(timegrid):
                                        if ti==0: 
                                            
                                            D[:,:,0] = initial.T#np.random.normal(loc=x0, scale=0.25,size=N)
                                        else:
                                            D[:,:,1] = D[:,:,0] + h* f(D[:,:,0],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,N3))
                                            D[:,:,0] = D[:,:,1]    
                                        ### calculate KL for every timepoint
                                        means[:,ti] = np.mean(D[:,:,0],axis=1)
                                        #covs[:,:,ti] = np.cov(D[:,:,0])
                                        kll[ti] = np.linalg.norm(means[:,ti]-m_t[ti])
                                    if np.std(kll)<thr:
                                        flags3[i] = 1
                                        N_star[i] = N3
                            
                                        filenm = 'Comp_complexity_particle_numberStoch_MEAN_FLUCTUATIONS_OU%d_trial_%d'%(run_id, i)

                                        Dic = dict()
                                        Dic['Ns'] = Ns
                                        Dic['dim'] = dim
                                        Dic['M'] = M
                                        
                                        Dic['N_star'] = N_star
                                        #Dic['kll'] = kll            
                                        Dic['timegrid'] = timegrid
                                        Dic['T'] = T
                                        Dic['h'] = h
                                        Dic['g'] = g
                                        Dic['trials'] = trials
                                        
                                        
                                        joblib.dump(Dic, filename= save_file+filenm)
                                        break;
                                else:
                                    break;
                    else:
                        break;
        else:
            break
                
    
    
    

        
        
        
        
        
>>>>>>> 24b3f5817b4dc75710dd31fe9512d82b4e847d38
            