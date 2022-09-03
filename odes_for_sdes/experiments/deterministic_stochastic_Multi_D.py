#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:54:14 2020

@author: dimitra
"""


import numpy as np
from matplotlib import pyplot as plt
import copy
from score_fucntion2_multid import score_function_multid
from score_function_multid_seperate import score_function_multid_seperate
#import sdeint
import seaborn as sns
#from score_function_lite import dens_est_lite
from grad_log_p import score_functionD
from plot_2D_distrib import multivariateGrid
import pandas as pd
from copy import deepcopy
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis  
from stochastic_models import climate_4d
import joblib
save_folder = 'Otto_results/nonlinear'
folder_name = 'Lorenz63/'
PLOT_MARGINALS = False
col_gr = '#3D4A49'
col_ro = '#c010d0'

h = 0.001 #sim_prec
t_start = 0.
t_end = 1
T =5#.5 #t_end
#initial condition
dim = 2
#x0 = np.array([0,1,0])
#x0 = np.array([0,0,0,0])
g = 10#np.sqrt(0.0025)#np.array([0.5,2,0.5,1])
#potential
a_1 = 1 #center/position of boundary
b_1 = 1 # distance between minima
a_2 = 1 #center/position of boundary
b_2 = 1 # distance between minima
Vmax = 4
#V = lambda x,y: (Vmax/b_1**4)*((x-a_1)**2 - b_1**2)**2 + (Vmax/b_2**4)*((y-a_2)**2 - b_2**2)**2
#drift function (-grad V)
#f = lambda x,t: -np.array([4*Vmax*(x[0]-a_1)*((x[0]-a_1)**2-b_1**2)/b_1**4,4*Vmax*(x[1]-a_2)*((x[1]-a_2)**2-b_2**2)/b_2**4]).reshape(x.shape)

#f = lambda x,t: np.array([ - 1*x[0]+1*np.sin(x[1]),-x[1]]) # 

#f = lambda x,t: np.array([ -4*x[0]*x[0]*x[0]+4*x[0]-x[1],-4*x[1]*x[1]*x[1]+4*x[1]-2*x[0]]) #double well 


a = 0.1
b = 0.075
c = 0.1
I0 = 0.3
f= lambda x,t: np.array([ x[0]*(a-x[0])*(x[0]-1)-x[1]+I0,-c*x[1]+b*x[0]])  # FHN neuron

#f = lambda x,t: np.array([-x[0]**1+0.5*x[1] , -x[1]**1+0.3*x[0]])

#f = lambda x,t: np.array([x[0]*(2-x[1]), x[1]*(x[0]-1)]) #lotka voltera like
ki = 1
#f = lambda x,t: np.array([-ki*x[1]+(1-x[0]**2-x[1]**2)*x[0], ki*x[0]+(1-x[0]**2-x[1]**2)*x[1]]) # harmonic osci - non conservative

#f = lambda x,t:  np.array([-2*x[0]+0.5*x[2] , -2*x[1]+0.5*x[0],-2*x[2]+0.5*x[1]])
rho = 28
sigm = 10
beta = 8/3.
#f = lambda x,t:  np.array([sigm* (x[1]-x[0]) , x[0]*(rho-x[2])-x[1],x[0]*x[1] - (beta)* x[2]]) # Lorentz 63 rh0=28 :chaotic, rho=0.5 : fixed point

x0 = [-0,-0,5]
#x0 = [1,0.5]
#epsil = 1
#f = lambda x,t: climate_4d(x,t,epsil)                     #climate 4d model                     


Ns = 4000 # number of stochastic particles
N = 4000  #number of particles
Ninf = 10000
#diffission coefficient/fucntion

gii = lambda x,t:np.array([1*g,0.01* g])#np.multiply(np.ones(dim),g)
xs = np.zeros((dim,Ninf))
for ii in range(dim):
    xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=Ninf)
#xs=np.array([(),(np.random.normal(loc=x02, scale=0.25,size=N)),(np.random.normal(loc=x03, scale=0.25,size=N))])
print(xs.shape)
opt_l = []
opt_s = []

def f_eff(x,t):#plain GP prior
    lnthsc = 40#np.max(x) - np.min(x)
    gpsi= score_function_multid(x.T,'None',g**2,C=0.01,which=1,l=lnthsc)
    gpsi2 = np.array([gpsi[:N],gpsi[N:]]).reshape(x.shape)
#    print(gpsi.shape)
#    print(x.shape)
    #print(f(x,t).shape)
    return (f(x,t)-0.5*g**2 * gpsi2)



# def f_eff_lite(x,t):#plain GP prior
#     lnthsc = np.max(x) - np.min(x)
# #    print(x.shape)
#     #gpsi= score_function_multid(x.T,'None',g**2,C=1/N,which=1,l=lnthsc)
#     gln3,sigm,l = dens_est_lite(x.T, sigma =lnthsc, lmbda = 1/N, num_evals = 5, num_in_evals=100, opt=True)
#     print('-----')
#     print(sigm)
#     print(l)
#     print('------')
#     opt_l.append(l)
#     opt_s.append(sigm)
#     gpsi2 = np.zeros((2,N))    
#     for i in range(N):
#         gpsi2[:,i] = gln3(np.array([x[:,i]]).reshape(2,))
#     #print(f(x,t).shape)
#     return (f(x,t)-0.5*g**2 * gpsi2)


def f_seperate(x,t):#plain GP prior
    N_sparse = 200
    dimi, N = x.shape
    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
    
    Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )
    
    gpsi = np.zeros((dimi, N))
    lnthsc = 2*np.std(x,axis=1)
    #print(lnthsc)
    for ii in range(dimi):
        print('here')
        gpsi[ii,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=ii+1)[0] #C=0.1 !!!!!
    
    ##lnthsc = [2*np.std(x[0,:]),2*np.std(x[1,:]),2*np.std(x[2,:])]#
    #gi_term = np.tile(gii(x,t)**2 ,(N,1)).T
    #print(np.multiply( gi_term, gpsi))
    
    #lnthsc = 2*np.std(x[0,:])
    ##gpsix= score_function_multid_seperate(x.T,Zxx.T,'None',g**2,C=0.001,which=1,l=lnthsc)[0] #C=0.1 !!!!!
    
    #lnthsc = 2*np.std(x[1,:])
    ##gpsiy= score_function_multid_seperate(x.T,Zxx.T,'None',g**2,C=0.001,which=1,l=lnthsc,which_dim=2)[0]
    #lnthsc = 2*np.std(x[2,:])
    ##gpsiz= score_function_multid_seperate(x.T,Zxx.T,'None',g**2,C=0.001,which=1,l=lnthsc,which_dim=3)[0]
    ##N = gpsix.size
    ##gpsi = np.array([gpsix,gpsiy,gpsiz])
    
    #gi_term = np.tile(gii(x,t)**2 ,(N,1)).T
    
    
    
    
    return (f(x,t)-0.5* g**2* gpsi)







  

timegrid = np.arange(0,T,h)
#Z = np.zeros((2,N,timegrid.size))
F = np.zeros((dim,Ns,timegrid.size))
G = np.zeros((dim,N,timegrid.size))
#M = np.zeros((dim,Ninf,timegrid.size))
#w = np.zeros((N,timegrid.size))
#v = np.zeros((N,N,timegrid.size))




#Z = sdeint.itoint(f_eff, gi, xs, timegrid).T



for ti,t in enumerate(timegrid):
    
    print(ti)
    if ti==0: 
        #Z[:,:,ti] = deepcopy(xs) 
        G[:,:,ti] = deepcopy(xs[:,:N]) 
        F[:,:,ti] = deepcopy(xs[:,:Ns])
        #M[:,:,ti] = deepcopy(xs) 
    else:
        #feff_t = f_eff_lite(Z[:,:,ti-1],ti)
        G[:,:,ti] = G[:,:,ti-1] + h* f_seperate(G[:,:,ti-1],t)
        #M[:,:,ti] = M[:,:,ti-1] + (h)* f_eff(M[:,:,ti-1],t)
        
            #Z[:,i,ti] = Z[:,i,ti-1] + h* feff_t[:,i]
            
            #F[:,i,ti] = F[:,i,ti-1] + h* f(F[:,i,ti-1],ti) + (g)*np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=dim) ]).reshape(dim,)
        F[:,:,ti] = F[:,:,ti-1] + h* f(F[:,:,ti-1],ti) + g*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,Ns))
#        for i in range(Ninf):
#            #Z[:,i,ti] = Z[:,i,ti-1] + h* feff_t[:,i]
#            
#            #F[:,i,ti] = F[:,i,ti-1] + h* f(F[:,i,ti-1],ti) + (g)*np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=dim) ]).reshape(dim,)
#            M[:,i,ti] = M[:,i,ti-1] + h* f(M[:,i,ti-1],ti) + np.multiply(gii(0,0),np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=dim) ]).reshape(dim,))


#Dict = {'G':G, 'h':h, 'timegrid': timegrid, 'T': T, 'N':N, 'g':g, 'x0':x0,'F':F}
#folder_name = 'HarmOsc1/'
#joblib.dump(Dict,folder_name+ 'HarmonicOscill_FP_deterministic_samples_dt_%.4f_N_%d_Ns_%d_noise_%.3f'%(h,N,Ns,g) + '.gz', compress='gzip')  
folder_name = 'FHN/'
#joblib.dump(Dict,folder_name+ 'FHN100_FP_deterministic_samples_dt_%.4f_N_%d_Ns_%d_noise_%.3f'%(h,N,Ns,g) + '.gz', compress='gzip')  


Dict = {'G':G,'F':F, 'h':h, 'timegrid': timegrid, 'T': T, 'N':N, 'g':g, 'x0':x0,'M':200, 'Mtype':'in'}
#joblib.dump(Dict,filename='Lorenz_%d_deter_and_stoch_M_200_in'%N)


#%%
M = joblib.load('Lorenz_FP_stochastic_samples_rho_28_dt_0.0010_N_150000.gz')
M= M['F']
N=4000
Di = joblib.load('Lorenz_%d_deter_and_stoch_M_200_in'%N)
timegrid = Di['timegrid'] 
F = Di['F']
G = Di['G']
#%%
col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #

col_grn = '#208f33'
col_gr2 = '#272e2d'

plt.rc('axes', linewidth=1.5)
plt.rc('axes',edgecolor='#0a0a0a')
#plt.rcParams['text.usetex'] = True

font = {'family' : 'sans-serif',
    'weight' : 'semibold',
    'size'   : 24,
    'style'  : 'normal'}

fig_width_pt = 1014#546.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'font.size': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
           'xtick.top': False,
           'ytick.right': False, 'figure.figsize':fig_size }
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=2
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})
#%%
from plot_statistics import plot_statistics


plot_statistics(timegrid,[M[:,:,:1500],F,G],labelss=['x','y','z'],labelkey = [r'S$_{\infty}$','S','D'], colors = [col_gr2,col_ye,col_grn ])

#%%     
import SeabornFig2Grid as sfg
import matplotlib.gridspec as gridspec

iiii = 0
figl = []

if False:
    for ti in range(0,timegrid.size,500): 
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1) 
        #sns.distplot(G[0,:,ti],100, norm_hist =True,label='D2', ax=ax1)
        sns.distplot(F[0,:,ti],100, norm_hist =True,label='S', ax=ax1)
        ax2 = fig.add_subplot(2,1,2) 
        #sns.distplot(G[1,:,ti],100, norm_hist =True,label='D2', ax=ax2)
        sns.distplot(F[1,:,ti],100, norm_hist =True,label='S', ax=ax2)
        plt.title(t)
        #ax1.hist(Z[0,:,ti],'b',alpha=0.5,label='D',histtype='step',density=True)
        #ax1.hist(F[0,:,ti],'r',alpha=0.5,label='S',histtype='step',density=True)
else:    
#    for ti in range(0,timegrid.size,100):    
#        dfG = pd.DataFrame(G[:,:,ti].T, columns=['x','y'])
#        dfF = pd.DataFrame(F[:,:,ti].T, columns=['x','y'])
#        #dfM = pd.DataFrame(M[:,:,ti].T, columns=['x','y'])
#        dfG['kind'] = 'deter_s'
#        dfF['kind'] = 'stoch'
#        #dfM['kind'] = 'deter'
#        df=pd.concat([dfG,dfF])
#        
#        fig0 = multivariateGrid('x', 'y', 'kind', df=df,legend= ti==0)
#        figl.append(fig0)
#        iiii += 1
##        plt.savefig(save_folder+'rotatenew2dOttomine_ti_%d_lite_N_%d_h_%.5f.png'%(ti,N,h))
#        plt.close()
#    fig = plt.figure(figsize=(16,10))
#    gs = gridspec.GridSpec(4, 3)
#    for iii in range(iiii):
#        sfg.SeabornFig2Grid(figl[iii], fig, gs[iii])
#        #plt.close(figl[iii])
#        gs.tight_layout(fig)
#    plt.savefig('2d_Otto_hist_seperate_grads_N_%d.png'%N)
#    plt.close()
    
#    rG = np.zeros(( N,len(timegrid)))
#    rF = np.zeros((N, len(timegrid)))
#    
#    for ti in range(len(timegrid)):
#        for ii in range(N):
#            rG[ii, ti] = np.sqrt(G[0,ii,ti]**2 + G[1,ii,ti]**2)
#            rF[ii, ti] = np.sqrt(F[0,ii,ti]**2 + F[1,ii,ti]**2)
#    
#    
#    plt.figure(),
#    plt.subplot(2,2,1),
#    plt.plot(timegrid, np.mean(rG,axis=0),label='D')
#    plt.plot(timegrid, np.mean(rF,axis=0),label='S')
#    plt.legend()
#    plt.subplot(2,2,2),
#    plt.plot(timegrid, np.std(rG,axis=0),label='D')
#    plt.plot(timegrid, np.std(rF,axis=0),label='S')
#    plt.subplot(2,2,3),
#    plt.plot(timegrid, skew(rG,axis=0),label='D')
#    plt.plot(timegrid, skew(rF,axis=0),label='S')
#    plt.subplot(2,2,4),
#    plt.plot(timegrid, kurtosis(rG,axis=0),label='D')
#    plt.plot(timegrid, kurtosis(rF,axis=0),label='S')
#    plt.savefig('Both_2d_Otto_r_statistics_N_%d'%N)
#    plt.close()
    storing_in = 'sequ/'
    
    
        
#    plt.figure(figsize = (10,8))
#    plt.scatter(Zx[0], Zx[1] ,c=fv(Zx[0],Zx[1]), cmap = 'hot', s=1)
#    plt.colorbar().set_label('Analytic solution', fontsize=14)
#    plt.xlabel('x', fontsize=14)
#    plt.ylabel('y', fontsize=14)
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    ax.scatter3D(F[0,1,:], F[1,1,:], F[2,1,:], c=F[2,1,:], cmap='Greens');
    
    
#    ii = 1
#    plt.figure(figsize=(16,10))
#    #for ti in range(0,timegrid.size,100):     
#    for ti in range(0,12): 
#        plt.subplot(4,3,ii)
#        plt.hist(G[0,:,ti],100,label='D',density=True)
#        plt.hist(F[0,:,ti],100,alpha=0.5,label='S',density=True)
#        plt.ylabel('x')        
#        ii += 1
#    plt.legend()
#    plt.title('tenfirst')
#    plt.savefig('Both_2d_Otto_marginal_hist_seperate_grads_N_%d_x_tenfirst.png'%N)
#    plt.close()
#    for di in range(dim):
#    
#        ii = 1
#        plt.figure(figsize=(16,10))
#        for ti in range(0,timegrid.size,150):     
#        #for ti in range(0,12,1):  
#            print(ti)
#        #for ti in range(0,10): 
#            plt.subplot(4,3,ii)
#            plt.hist(G[di,:,ti],300,label='D',density=True, histtype='step', linewidth=2)
#            plt.hist(F[di,:,ti],300,label='S',density=True, histtype='step', linewidth=2)
#            plt.ylabel('x_%d'%di)        
#            ii += 1
#        plt.legend()
#        
#        plt.savefig(folder_name+'Both_%d_d_Otto_marginal_hist_seperate_grads_N_%d_x_%d_noise_'%(dim,N,di)+'_'.join( list(map(str),[g]) ) + '.png')
#        plt.close()
    labelss = ['x','y','z'] 
    
#    for di in range(dim):
#    
#        ii = 1
#        plt.figure(figsize=(16,10))
#        for ti in range(0,timegrid.size,150):     
#        #for ti in range(0,12,1):  
#            print(ti)
#        #for ti in range(0,10): 
#            plt.subplot(4,3,ii)
#            sns.distplot(F[di,:,ti],100,hist=True, kde=True, label='S',color=col_gr)
#            sns.distplot(G[di,:,ti],100,hist=True, kde=True, label='D',color=col_ro)
#            plt.ylabel('%s'%labelss[di])
#            #plt.ylabel('x_%d'%di)        
#            ii += 1
#        plt.legend()
#        
#        plt.savefig(folder_name+'Both_%d_d_Otto_marginal_hist_seperate_grads_N_%d_x_%d_noise_'%(dim,N,di)+'_%.3f'%g + '.png')
#        plt.close()
#    
    
#    Dict = joblib.load(folder_name+'Lorenz_FP_stochastic_samples_rho_28_dt_0.0010_N_150000.gz')
#    F = Dict['F']  
    col_gr = '#3D4A49'
    col_ro = '#208f33'#'#c010d0'
    col_ye =  '#847a09'    
      
    plt.figure(figsize=(16,10))
    for di in range(dim):
        plt.subplot(dim,4,di*4+1)
        plt.plot(timegrid, np.mean(M[di,:,:],axis=0),lw=2.5,label=r'S$^{\infty}$',color=col_gr)
        plt.plot(timegrid, np.mean(F[di,:,:],axis=0),lw=2.5,label='S',color=col_ye)
        plt.plot(timegrid, np.mean(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)       
        
        #plt.ylabel(u'mean_x_%d'%di)    
        plt.ylabel(r'$\langle %s \rangle $'%labelss[di], fontsize=14)          
        if di==0:
            plt.legend()
            plt.gca().set_title('mean', fontsize=14)
        if di==dim-1:
            plt.xlabel('time', fontsize=14)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
            
      
        plt.subplot(dim,4,di*4+2)
        plt.plot(timegrid, np.std(M[di,:,:],axis=0),lw=2.5,label=r'S$^{\infty}$',color=col_gr)
        plt.plot(timegrid, np.std(F[di,:,:],axis=0),lw=2.5,label='S',color=col_ye)
        plt.plot(timegrid, np.std(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        plt.ylabel(r'$ \sigma_{%s}$'%(labelss[di]), fontsize=14)          
        if di==0:            
            plt.gca().set_title('std', fontsize=14)
        if di==dim-1:
            plt.xlabel('time', fontsize=14)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
      
        plt.subplot(dim,4,di*4+3)
        plt.plot(timegrid, skew(M[di,:,:],axis=0),lw=2.5,label=r'S$^{\infty}$',color=col_gr)
        plt.plot(timegrid, skew(F[di,:,:],axis=0),lw=2.5,label='S',color=col_ye)
        plt.plot(timegrid, skew(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        plt.ylabel(r'$s_{%s} $'%(labelss[di]), fontsize=14)          
        if di==0:            
            plt.gca().set_title('skewness', fontsize=14)
        if di==dim-1:
            plt.xlabel('time', fontsize=14)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
      
        plt.subplot(dim,4,di*4+4)
        plt.plot(timegrid, kurtosis(M[di,:,:],axis=0),lw=2.5,label=r'S$^{\infty}$',color=col_gr)
        plt.plot(timegrid, kurtosis(F[di,:,:],axis=0),lw=2.5,label='S',color=col_ye)
        plt.plot(timegrid, kurtosis(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        plt.ylabel(r'$k_{%s}  $'%(labelss[di]), fontsize=14)          
        if di==0:            
            plt.gca().set_title('kurtosis', fontsize=14)
        if di==dim-1:
            plt.xlabel('time', fontsize=14)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)   
    
    plt.subplots_adjust( wspace=0.3, hspace=0.2)
    plt.savefig(folder_name+'Both_%d_d_Otto_statistics_seperate_grads_N_%d_Ns_%d_noise_%.3f'%(dim,N,Ns,g)+'_' + '.png')
    plt.close()
    
    
    #colum = [ 'x_%d'%iii for iii in range(dim)   ]
#    colum = ['x','y']
#    colum = ['V',u'$\omega$']
#    ax_lims = np.zeros((dim,2))
#    for di in range(dim):
#        
#        ax_lims[di,0] = min(np.min(G[di]), np.min(F[di]))
#        ax_lims[di,1] = max(np.max(G[di]), np.max(F[di]))
#        padi = (ax_lims[di,1] -ax_lims[di,0])/10
##        ax_lims[di,0] -= padi
##        ax_lims[di,1] += padi
#        
#        
#    for ti in [1000]:#range(0,timegrid.size-1,50):
##    for ti in range(0,50,1):
#    
#        #xss = np.zeros(-2,2,100)
#        
#        dfM = pd.DataFrame(M[:,:,ti].T, columns=colum)
#        dfG = pd.DataFrame(G[:,:,ti].T, columns=colum)
#        #dfF = pd.DataFrame(F[:,:,ti].T, columns=colum)
#        #dfM = pd.DataFrame(M[:,:,ti].T, columns=['x','y'])
#        dfG['kind'] = 'deter'
#        #dfF['kind'] = 'deter'
#        dfM['kind'] = 'stoch'
#        #dfM['kind'] = 'deter'
#        df=pd.concat([dfF,dfM])
#        
#        
#        fig0 = multivariateGrid(colum[0], colum[1], 'kind', df=df,k_is_color=True,ax_lims=ax_lims)
#        #dim=1
#        plt.savefig(folder_name +'Both_%d_d_Otto_hist_seperate_grads_N_%d_Ns_%d_JOINT_t_%05d_noise_'%(dim,N,Ns,ti)+'_noise_g%.3f'%g + '.png')
#        plt.close()
#        
###        fig0 = multivariateGrid(colum[0], colum[2], 'kind', df=df,k_is_color=True)
##        dim=3
##        plt.savefig(folder_name +'Both_%d_d_Otto_hist_seperate_grads_N_%d_JOINT_t_%05d_noise_'%(dim,N,ti)+'_noise_g%.2f'%g + '.png')
##        plt.close()
##        
##        fig0 = multivariateGrid(colum[1], colum[2], 'kind', df=df,k_is_color=True) 
##        dim=2
##        plt.savefig(folder_name +'Both_%d_d_Otto_hist_seperate_grads_N_%d_JOINT_t_%05d_noise_'%(dim,N,ti)+'_noise_g%.2f'%g + '.png')
##        plt.close()
#        
#        
    