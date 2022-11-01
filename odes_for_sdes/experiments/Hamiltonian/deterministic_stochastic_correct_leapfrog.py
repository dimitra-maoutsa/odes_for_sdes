# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:51:05 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from score_function_multid_seperate import score_function_multid_seperate
from copy import deepcopy
from plot_2D_distrib import multivariateGrid
import pandas as pd
from plot_statistics import plot_statistics

#%%
#h = 0.00005 #sim_prec
h = 0.0005
t_start = 0.

T =15#0#.5 
dim = 2
#initial condition
x0 = np.array([-0.2,-0.3])
#x0 = np.array([0,0,0,0])
sigm = 1


N= 1000
gii = lambda x:np.tile( np.array([ 0,sigm]) ,(N,1)).T# multiply 0 with x in order to get the dimensionality of x 
xs = np.zeros((dim,N))
for ii in range(dim):
    xs[ii] = np.random.normal(loc=x0[ii], scale=0.05,size=N)
#mi = -0.25 # 0 # 0.25
#f =lambda x: np.array([ mi-x[1]**2 ,x[0]   ]   ) #x_0 = x , x_1 = v
#omega = 5
#f =lambda x,y: np.array([-0.5*omega**2*x[0]*h+x[1] ,0.5*(-0.5*omega**2*x[0]-0.5*omega**2*y[0])]   ) #x_0 = x_t , x_1 = v_t ; y_0=x_{t+1}, y_1=v_{t+1} #harmonic
#f = lambda x: -x
gamma = 1

f = lambda x: -4*x # Langevin linear

V = lambda x: 2*x**2


#f = lambda x: -4*x**3+4*x # Langevin DW
#
#V = lambda x: x**2*(x**2-2)

def f_seperate(x,Zxx):#plain GP prior
    #N_sparse = 50
    dimi, N = x.shape
    
    bnds = np.zeros((dimi,2))
    for ii in range(dimi):
        bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
    
    #Zxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(N_sparse)) for bnd in bnds ] )
    
    gpsi = np.zeros((1, N))
    lnthsc = 2*np.std(x,axis=1)
    
            
    gpsi[0,:]= score_function_multid_seperate(x.T,Zxx.T,'None',0,C=0.001,which=1,l=lnthsc,which_dim=2)[0] #C=0.1 !!!!!   
    
    
    return (-0.5* sigm**2*gpsi )  


timegrid = np.arange(0,T,h)

F = np.zeros((dim,N,timegrid.size))
G = np.zeros((dim,N,timegrid.size))
gradlog = np.zeros((N,timegrid.size))
gamma = 1
bnds = np.zeros((2,2))
Zxx = np.array([ np.random.uniform(low=-1,high=1,size=(50)) for bnd in bnds ] )

for ti,t in enumerate(timegrid):
    
    #print(ti)
    if ti==0 or ti==1: 
        
        G[:,:,ti] = deepcopy(xs[:,:N]) 
        F[:,:,ti] = deepcopy(xs[:,:N])
        
        
    else:
        if ti%2==0: # enter only on even timesteps
            #########################################################################################################
            ###deterministic
            gradlog[:,ti-2] = f_seperate( G[:,:,ti-2],Zxx) 
            # assume that h = dt/2
            ### set v for step n+1/2
            G[1,:,ti-1] = G[1,:,ti-2] + h* ( -gamma* G[1,:,ti-2]+ f(G[0,:,ti-2]) ) -h*( (sigm**2)/2 * gradlog[:,ti-2] ) #first v step
            ### set x for step n+1
            G[0,:,ti] = G[0,:,ti-2] + 2*h* G[1,:,ti-1] 
            ### set x step n+1/2 = x@step n+1
            ####set the x half step equal with the whole step so that it would be easier to pass the matrix to f_seperate
            #G[0,:,ti-1] = G[0,:,ti-2] + h* G[1,:,ti-1] 
            G[0,:,ti-1] = G[0,:,ti]+ h* G[1,:,ti-1]
            ### set v for step n+1
            gradlog[:,ti-1] = f_seperate( G[:,:,ti-1],Zxx) 
            G[1,:,ti] = G[1,:,ti-1] + h* ( -gamma* G[1,:,ti-1]+ f(G[0,:,ti-1]))  -h*( (sigm**2)/2 * gradlog[:,ti-1] ) #here give again the ti-2 array to compute log density
            
            ############################################################################################################
            #stochastic
            
            F[1,:,ti-1] = np.exp(-gamma*h)* F[1,:,ti-2] + ((1-np.exp(-gamma*h))/gamma)*f(F[0,:,ti-2]) + sigm*np.sqrt((1-np.exp(-gamma*2*h))/(2*gamma))*np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(N)) ])
            #F[1,:,ti-1] =  F[1,:,ti-2] + h*(f(F[0,:,ti-2]) -gamma* F[1,:,ti-2])
        
            F[0,:,ti] = F[0,:,ti-2] + F[1,:,ti-1] *2*h 
            F[0,:,ti-1] = F[0,:,ti]+ F[1,:,ti-1] *h # here also copy the x whole step to the laf step to avoid the zeros when taking averages
            #F[1,:,ti] =  F[1,:,ti-1] + h*( f(F[0,:,ti-1]) -gamma* F[1,:,ti-1])
            F[1,:,ti] =  np.exp(-gamma*h)* F[1,:,ti-1] + ((1-np.exp(-gamma*h))/gamma)*f(F[0,:,ti]) + sigm*np.sqrt((1-np.exp(-gamma*2*h))/(2*gamma))*np.array([np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(N)) ])
        
        
Dict = dict()
Dict['F'] = F
Dict['G'] = G
Dict['gradlog'] = gradlog
Dict['Nspar'] = 50
Dict['timegrid'] = timegrid
Dict['T'] = T
Dict['sigm'] = sigm
Dict['f'] = '-x'
Dict['h'] = h
Dict['x0'] = x0
Dict['N'] = N
Dict['gamma'] = gamma
#joblib.dump(Dict,filename='Data_for_hamiltonian_N_%d_M_50'%N)

#%%
col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #
col_gr2 = '#272e2d'
col_grn = '#208f33'

plot_statistics(timegrid[::2],[F[0,:,::2], G[0,:,::2]],labelss=['x','p','z'],labelkey = ['S','D'],colors = [col_ye,col_grn ])
plot_statistics(timegrid[::2],[F[1,:,::2], G[1,:,::2]],labelss=['x','p','z'],labelkey = ['S','D'],colors = [col_ye,col_grn ])
#%%

#%%
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1]])
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=5)

col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #
col_gr2 = '#272e2d'
col_grn = '#208f33'

plt.rc('axes', linewidth=1.5)
plt.rc('axes',edgecolor='#0a0a0a')
#plt.rcParams['text.usetex'] = True

font = {'family' : 'sans-serif',
    'weight' : 'semibold',
    'size'   : 24,
    'style'  : 'normal'}


params = {'backend': 'ps',
          'axes.labelsize': 26,
          'font.size': 20,
          'legend.fontsize': 10,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
           'xtick.top': True,
           'ytick.right': True
          }
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False

#%%
#N=1000
#Dict = joblib.load('Data_for_hamiltonian_N_%d_M_50'%N)
#F =   Dict['F']
#G = Dict['G']
##gradlog =Dict['gradlog'] 
#
#timegrid = Dict['timegrid'] 
#%%
differences = np.zeros((int(timegrid.size/2),N))
for ti,tt in enumerate( timegrid[::2]):
    #differences[ti,:] = np.abs(G[1,:,ti]- gpsis[ti])
    if ti< timegrid.size-2:
        differences[ti,:] = np.linalg.norm((G[1,:,ti]+ G[1,:,ti+1])/2- (gradlog[:,ti]- gradlog[:,ti+2])/2)
    
differences2 = np.zeros((timegrid.size))
for ti,tt in enumerate( timegrid[:-1]):
    #differences2[ti] = np.abs(np.mean(G[1,:,ti])- np.mean(gpsis[ti]))
    if ti< timegrid.size-2:
        differences2[ti] = np.linalg.norm(np.mean((G[1,:,ti]+ G[1,:,ti+1])/2)- np.mean((gradlog[:,ti]- gradlog[:,ti+2])/2))
    
#%%    
plt.figure(figsize=(5,5)),
plt.plot(timegrid[::2], differences[:,0], lw=5, alpha = 0.7)
plt.xlim([10,None])
plt.ylim([-0.2,0.2])
plt.figure(),
plt.plot(timegrid, differences2)

#%%

cpmap_grn = sns.cubehelix_palette(3, start=2, rot=0, dark=0.2, light=.65, reverse=False,as_cmap=True)
cpmap_yel = sns.cubehelix_palette(3, start=1.5, rot=0, dark=0.2, light=.75, reverse=False,as_cmap=True)
## plot individual particle energies
E = np.zeros((N,timegrid.size-1))
for i in range(N):
    #E[i] = 0.5*F[0,i,:-1]**2 + 0.5*F[1,i,:-1]**2
    for ti in range(timegrid.size-1):
        if ti%2==0:
            E[i,ti] += 0.5*( ((G[1,i,ti] + G[1,i,ti+1])/2.0)**2) + V(G[0,i,ti])


Es = np.zeros((N,timegrid.size-1))
for i in range(N):
    #E[i] = 0.5*F[0,i,:-1]**2 + 0.5*F[1,i,:-1]**2
    for ti in range(timegrid.size-1):
        if ti%2==0:
            Es[i,ti] += 0.5*( ((F[1,i,ti] + F[1,i,ti+1])/2.0)**2) + V(F[0,i,ti])



            
sns.set_palette( sns.cubehelix_palette(3, start=2, rot=0, dark=0.2, light=.65, reverse=False,as_cmap=False))   
plt.figure(figsize=(5,5)), plt.plot(timegrid[::2]/2,E[:5,::2].T,lw=5,alpha=0.9)
plt.xlabel('time',fontsize=22)
plt.ylabel(r'$E^{(i)}_t$',fontsize=22)
plt.savefig('Hamiltonian_Particle_Energy_D'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Hamiltonian_Particle_Energy_D'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  


#fluctuations per particle in the steady state phase
det_fl = np.std(E[:,::2],axis=1)
sto_fl = np.std(Es[:,::2],axis=1)

colum = ['fl']
pd_list =[]
pd_list.append(pd.DataFrame(det_fl, columns=colum))
pd_list[0]['kind'] = 'D'
pd_list.append(pd.DataFrame(sto_fl, columns=colum))
pd_list[1]['kind'] = 'S'

fluctuations =pd.concat(pd_list)  
    

b=sns.boxplot(x='kind', y='fl', data=fluctuations,  palette=[cpmap_grn(100),cpmap_yel(100)],linewidth=4)

plt.ylabel(r'$ \langle (E^{(i)}_t - \langle E^{(i)}_t \rangle_t )^2 \rangle_t  $',fontsize=22)
plt.xlabel('')
b.set_xticklabels(['D','S'], size = 22)
plt.savefig('Hamiltonian_Box_Fluctuations'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Hamiltonian_Box_Fluctuations'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  





sns.set_palette( sns.cubehelix_palette(3, start=1.5, rot=0, dark=0.2, light=.75, reverse=False,as_cmap=False))#yellow            
plt.figure(figsize=(5,5)), plt.plot(timegrid[::2]/2,Es[:3,::2].T,lw=3,alpha=0.9)
plt.ylim([None,0.0006])
plt.xlabel('time',fontsize=22)
plt.ylabel(r'$E^{(i)}_t$',fontsize=22)
plt.savefig('Hamiltonian_Particle_Energy_S'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Hamiltonian_Particle_Energy_S'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
####
#plot total energy

plt.figure(figsize=(5,5)),
plt.plot(timegrid[::2]/2,np.mean(Es[:,::2],axis=0),lw=5,color = cpmap_yel(180),label='S')
plt.plot(timegrid[::2]/2,np.mean(E[:,::2],axis=0),lw=5, color=cpmap_grn(180),label='D')
plt.ylim([-0.2,5])
plt.xlim([1,None])
plt.xlabel('time',fontsize=22)
plt.ylabel(r'Total Energy',fontsize=22)
plt.legend(title=None, loc="best", ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =1,edgecolor ='#0a0a0a')

plt.savefig('Hamiltonian_Total_Energy'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Hamiltonian_Total_Energy'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  


plt.figure(), plt.plot(np.mean(G[1,:,::2]**2,axis=0)), plt.title('Average kinetic energy')
plt.plot(np.mean(F[1,:,::2]**2,axis=0))
plt.plot([0,8000], [sigm**2/(2*gamma),sigm**2/(2*gamma)],'k--')

###
#xss = np.linspace(-2,2, 100)

#f = lambda x: (x**2-1)**2  + 0.5* (1+ 10*np.exp((-200*(x-1)**2)))**2*(0.13)**2

#plt.plot(xss, f(xss))
    
