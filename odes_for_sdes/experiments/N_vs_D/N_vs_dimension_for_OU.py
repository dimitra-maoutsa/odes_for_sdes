#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:48:14 2020

@author: dimitra
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
#import ot
#from score_function_multid_seperate import score_function_multid_seperate
import scipy as sc
import pandas as pd
from scipy.integrate import odeint
from odeintw import odeintw

save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'

#%%
def W_2(m1,m2,S1, S2):
    """
    Calculates Wasserstein-2 distance between two gaussiandistributions with mean m1, m2 
    and covariance matrices S1 and S2
    
    """
    sqrtS1= np.linalg.cholesky(S1+0.0001*np.eye(S1.shape[0]))
    #print(sqrtS1 @ S2 @ sqrtS1)
    sqrtall= np.linalg.cholesky(sqrtS1 @ S2 @ sqrtS1 +0.0001*np.eye(S1.shape[0]) )
    
    W2 = np.linalg.norm(m1-m2)**2 + np.trace(S1) + np.trace(S2) - 2* np.trace(  sqrtall )
    
    return np.sqrt(W2)


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
    


m1 = np.array([1,0])
#m2 = np.array([1,0])
S1 = np.array([[4,0],[0,5]])
S2 = np.array([[4,1],[0,5]])

trials = 100
wd = np.zeros(trials)
kll = np.zeros(trials)
kl2 = np.zeros(trials)
klboth = np.zeros(trials)

for i in range(trials):
    np.random.seed(i)
    sampl1 = np.random.multivariate_normal(m1, S1, 2000)    
    sampl2 = np.random.multivariate_normal(m1, S1, 2000) 
    
    emp_m1 = np.mean(sampl1,axis=0) 
    emp_m2 = np.mean(sampl2,axis=0)
    
    emp_cov1 = np.cov(sampl1.T) 
    emp_cov2 = np.cov(sampl2.T) 
    
    
    #wd[i] = W_2(emp_m1, emp_m2, emp_cov1 , emp_cov2)
    
    kll[i] = KL(emp_m1, emp_m2, emp_cov1 , emp_cov2)
    
    kl2[i] =  KL(emp_m2, emp_m1, emp_cov2 , emp_cov1)
    klboth[i] = 0.5*(kll[i]+kl2[i])

metr = np.concatenate((wd,kll,kl2,klboth), axis=0)
typ = np.repeat(['w2','k1','k2','kb'] ,100)   
list_of_tuples = list(zip(metr, typ))     
df = pd.DataFrame(list_of_tuples, columns = ['x', 'metric']) 
plt.figure(),
sns.violinplot(x=df["metric"], y=df["x"], inner="quartile")




#%%

trials = 200
dims = [2,3,4,5,6,7]
Ns = np.arange(200,12000,200)
kll = np.zeros((len(dims),trials,Ns.size))
threshold = 0.005
be_th = np.zeros((len(dims),trials))
be_th.fill(np.nan)
for di,dim in enumerate(dims):
    print(dim)
    x0 = np.ones(dim)*0.5
    C_0 = np.zeros((dim,dim))
    np.fill_diagonal(C_0,0.25**2)
    
    m1 = np.array(x0)
    S1 = np.array(C_0)    
    flag = np.zeros(trials)
    
    for ss,N in enumerate(Ns):
        print(N)
        for i in range(trials):
            np.random.seed(i)
            sampl1 = np.random.multivariate_normal(m1, S1, N)    
            #sampl2 = np.random.multivariate_normal(m1, S1, N) 
            
            emp_m1 = np.mean(sampl1,axis=0) 
            #emp_m2 = np.mean(sampl2,axis=0)
            
            emp_cov1 = np.cov(sampl1.T) 
            #emp_cov2 = np.cov(sampl2.T) 
            
            
            #wd[i] = W_2(emp_m1, emp_m2, emp_cov1 , emp_cov2)
            
            kll[di,i,ss] = KL(m1, emp_m1, S1 , emp_cov1)
        
            if flag[i]==0 and kll[di,i,ss]<threshold:
                
                flag[i] = 1
                be_th[di,i] = N
    print(np.sum(flag))
    print('-------')
    
kl1 = np.mean(kll,axis=1)   
klstd = np.std(kll,axis=1)

plt.figure(),
ax = plt.gca()
for di in range(len(dims)):
    color=next(ax._get_lines.prop_cycler)['color']
    plt.plot(Ns,kl1[di],lw=2,label='D=%d'%dims[di],color=color)
    plt.plot(Ns, kl1[di]+klstd[di], '--', color=color,alpha=0.8,lw=2) 
    plt.plot(Ns, kl1[di]-klstd[di], '-.',color=color,alpha=0.8,lw=2) #0.005
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('KL')
plt.legend()

bemean = np.mean(be_th,axis=1)
bestd = np.std(be_th,axis=1)
plt.figure(),
plt.plot(dims, bemean)
plt.plot(dims, bemean + bestd, 'k--')
plt.plot(dims, bemean - bestd, 'k--')
plt.xlabel('D')
plt.ylabel(r'$N^{0.005}$')


typ = np.repeat(dims ,trials)   
list_of_tuples = list(zip(be_th.reshape(-1,), typ))     
df = pd.DataFrame(list_of_tuples, columns = [r'$N^{0.005}$', 'D']) 
plt.figure(),
sns.violinplot(x=df["D"], y=df[r'$N^{0.005}$'], inner="quartile")

#siz = np.repeat(Ns ,100)   
#list_of_tuples = list(zip(metr, siz))     
#df = pd.DataFrame(list_of_tuples, columns = ['x', 'metric']) 
Dictio = dict()
Dictio['kl1']=kl1
Dictio['be_th'] = be_th
Dictio['threshold'] = threshold
Dictio['dims'] = dims
Dictio['Ns'] = Ns
joblib.dump(Dictio,filename = 'D:\DataAssimilation\My_papers\Otto\KL_between_true_and_sample_for_simple_gaussians_vs_D_benchmark')

#%%
dim = 2
M = 100
seeds = np.arange(10, 30,1)
#dims = [1,2,3,4,5,6]
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
g = 1
x0 = 0.5

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

#### analytic moments
    
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
#integrate
m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_0,timegrid)





##################

N_inf = 1000*dim

AF = joblib.load(save_file+'OU_%d_D_samples_from_analytic_trajectories_N_%d'%(dim,N_inf))

N_was = np.zeros(20)

for si,seed in enumerate(seeds):
    flag = 1
    np.random.seed(seed)
    N = 200#int(0.8*N_inf)
    
    while flag:
        print('Generating deterministic trajectory....')
        #create deterministic samples 
        D = np.zeros((dim,N,timegrid.size))
        for ti,t in enumerate(timegrid):
            if ti==0: 
                for di in range(dim): 
                    D[di,:,0] = np.random.normal(loc=x0, scale=0.25,size=N)
            else:
                D[:,:,ti] = D[:,:,ti-1] + h* f_seperate(D[:,:,ti-1],t)
                
        print('Calculating Wasserstein distance...')        
        KLdis = np.zeros(timegrid.size)
        for ti in range(timegrid.size):
            
            m1 = m_t[ti]
            m2 = np.mean(D[:,:,ti],axis=1)
            
            S1 = C_t[ti]
            S2 = np.cov(D[:,:,ti])
            
#            Dict = {'D':D, 'KLdis':KLdis}
#            
#            joblib.dump(Wasdis,filename=save_file+'OU_%d_D_Deterministic_Trajectory_and_Wasserstein_between_analytic%d_vs_deterministic_N%d_M_%d_seed_%d'%(dim,N_inf,N,M,seed))
        
        print('Current Waserstein: %.4f' %(np.mean(Wasdis)/dim))
        
        if (np.mean(Wasdis)/dim) < 0.05:
            
            N_was[si] = N
            flag =0
            
        elif (np.mean(Wasdis)/dim) > 0.05:
            
            N += 100
            print('Unaccepted for %d dim - running for:%d - seed: %d'%(dim,N,si))
            
            
            
            
            
            
            
            
            
"""
xs = AF[:,:,ti] 
xt = D[:,:,ti]  
# loss matrix
Mdist = ot.dist(xs.T, xt.T)
Mdist /= Mdist.max()
a, b = np.ones((N_inf,)) / (N_inf), np.ones((N,)) / N  # uniform distribution on samples
Wasdis[ti] = ot.emd2(a,b,Mdist)


"""