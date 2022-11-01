# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:35:38 2020

@author: Dimi
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
#import ot
#%%

Di = joblib.load('Data_for_1d_DW_for_plot_statistics')
M =Di['M'] 
F=Di['F'] 
Z=Di['Z'] 
timegrid=Di['timegrid'] 
num_ind = Di['num_ind']
    
#%%
import figure_properties
col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #
col_gr2 = '#272e2d'
col_grn = '#208f33'
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'font.size': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
           'xtick.top': False,
           'ytick.right': False
          }
plt.rcParams.update(params)
#plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})
col_gr = '#3D4A49'
colorsgr = colsg['hex']
col_ro = colorsgr[2]
col_grn = colorsgr[2]
col_ye = colsp['hex'][2]
col_gr = '#3D4A49'
#%%
from plot_statistics import plot_statistics
Ninf = 26000
N=1000
plot_statistics(timegrid, [Z,F,M],colors = [col_gr2,col_ye,col_grn ],labelkey = [r'S$_{\infty}$','S','D'])
plt.savefig('Double_well_1D_dist_S_vs_S_ti_statistics_gg_new2_%dNinf_N%d_M%d.pdf'%(Ninf,N,num_ind),  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Double_well_1D_dist_S_vs_S_ti_statistics_gg_new2_%dNinf_N%d_M%d.png'%(Ninf,N,num_ind),  bbox_inches = 'tight', pad_inches = 0.1)    



#%%
plt.rc('axes', linewidth=2.8)
plt.rc('axes',edgecolor='#464646')
#sns.set(style="white",rc={'xtick.labelsize': 12, 'ytick.labelsize': 12})
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = False



from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
times = [5,231,1244]
for ti in times:
    f, ax = plt.subplots(1,1,figsize=(5,5))
    sns.despine(bottom=True, left=True)
    sns.distplot(Z[:,ti],60,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=ax,kde_kws={"lw":"3","bw":'silverman'}, hist_kws=dict(edgecolor=None))
    sns.distplot(M[:,ti],60,hist=True, kde=True, label='D',color=col_ro,ax=ax,kde_kws={"lw":"3","bw":'silverman'}, hist_kws=dict(edgecolor=None))
    #sns.distplot(F1[:,ti],50,hist=True, kde=True, label='S',color='b')
    #plt.ylabel('%s'%labelss[di])
    plt.xlabel(r'$x$')     
    plt.ylabel(r'$\hat{P}(x)$')  
    if ti<200:
        plt.ylim([0,5.98])
    else:
        plt.ylim([0,0.98])
    plt.locator_params(axis='y', nbins=3)
    if ti<100:
        plt.legend(frameon=False,fontsize=20,loc=2) 
    plt.subplots_adjust(left=0.17,right=0.95,top=0.980)
    
    
#    if ti>101:
#           
#        axins = zoomed_inset_axes(ax, 2, loc=1)
#        sns.distplot(Z[:,ti],100,hist=True, kde=True, label=r'S$^\infty$',color=col_gr,ax=axins,kde_kws={"lw":"3"})
#        sns.distplot(M[:,ti],100,hist=True, kde=True, label='D',color=col_ro,ax=axins,kde_kws={"lw":"3"})
#        if ti== 3200:
#            x1, x2, y1, y2 = 1.1,1.3, 0.5, 1.5 # specify the limits # specify the limits
#        elif ti==101:
#            x1, x2, y1, y2 = 1.2,1.4, 0.3, 0.8
#        elif ti== 4400:
#            x1, x2, y1, y2 = 1.3,1.6, 0, 0.5
#        axins.set_xlim(x1, x2) # apply the x-limits
#        axins.set_ylim(y1, y2) # apply the y-limits
#        plt.yticks(visible=False)
#        plt.xticks(visible=False)
#        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2")
#    
#    axins2 = zoomed_inset_axes(ax, 2, loc=2)
#    sns.distplot(Z[:,ti],100,hist=True, kde=True, label=r'S$^\infty$',color=col_gr,ax=axins2,kde_kws={"lw":"3"})
#    sns.distplot(M[:,ti],100,hist=True, kde=True, label='D',color=col_ro,ax=axins2,kde_kws={"lw":"3"})
#    if ti== 3200:
#        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5 # specify the limits # specify the limits
#    elif ti==101:
#        x1, x2, y1, y2 = 0.45,0.7, 0., 0.5
#    elif ti==4400:
#        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5
#    axins2.set_xlim(x1, x2) # apply the x-limits
#    axins2.set_ylim(y1, y2) # apply the y-limits
#    plt.yticks(visible=False)
#    plt.xticks(visible=False)
#    mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.2")
    
    plt.savefig('DW_1D_dist_D_vs_S_ti%d_gg.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0.05)     
    plt.savefig('DW_1D_dist_D_vs_S_ti%d_gg.png'%ti,  bbox_inches = 'tight', pad_inches = 0.05)  
    
    
for ti in times:
    f, ax = plt.subplots(1,1,figsize=(5,5))
    sns.despine(bottom=True, left=True)
    sns.distplot(Z[:,ti],60,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=ax,kde_kws={"lw":"3","bw":'silverman'})
    #sns.distplot(Fd1[:,ti],50,hist=True, kde=True, label='D',color=col_ro)
    sns.distplot(F[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=ax,kde_kws={"lw":"3","bw":'silverman'})
    #plt.ylabel('%s'%labelss[di])
    plt.xlabel(r'$x$')    
    #plt.ylim([0,2.9])
    if ti<200:
        plt.ylim([0,5.98])
    else:
        plt.ylim([0,0.98])
    plt.ylabel(r'$\hat{P}(x)$')      
    plt.locator_params(axis='y', nbins=3)
    if ti<100:
        plt.legend(frameon=False,fontsize=20,loc=1)
    plt.subplots_adjust(left=0.17,right=0.95,top=0.980) 

#    if ti>101:
#        
#       
#        axins = zoomed_inset_axes(ax, 2, loc=1)
#        sns.distplot(Z[:,ti],60,hist=True, kde=True, label=r'S$^\infty$',color=col_gr,ax=axins,kde_kws={"lw":"3"})    
#        sns.distplot(F[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=axins,kde_kws={"lw":"3"})
#        if ti== 3200:
#            x1, x2, y1, y2 = 1.1,1.3, 0.5, 1.5 # specify the limits # specify the limits
#        elif ti==101:
#            x1, x2, y1, y2 = 1.2,1.4, 0.3, 0.8
#        elif ti== 4400:
#            x1, x2, y1, y2 = 1.3,1.6, 0, 0.5
#            
#        axins.set_xlim(x1, x2) # apply the x-limits
#        axins.set_ylim(y1, y2) # apply the y-limits
#        plt.yticks(visible=False)
#        plt.xticks(visible=False)
#        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2")
#        
#    axins2 = zoomed_inset_axes(ax, 2, loc=2)
#    sns.distplot(Z[:,ti],60,hist=True, kde=True, label=r'S$^\infty$',color=col_gr,ax=axins2,kde_kws={"lw":"3"})    
#    sns.distplot(F[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=axins2,kde_kws={"lw":"3"})
#    if ti== 3200:
#        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5 # specify the limits # specify the limits
#    elif ti==101:
#        x1, x2, y1, y2 = 0.45,0.7, 0., 0.5
#    elif ti==4400:
#        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5
#    axins2.set_xlim(x1, x2) # apply the x-limits
#    axins2.set_ylim(y1, y2) # apply the y-limits
#    plt.yticks(visible=False)
#    plt.xticks(visible=False)
#    mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.2")
      
    plt.savefig('DW_1D_dist_S_vs_S_ti%d.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0.05)     
    plt.savefig('DW_1D_dist_S_vs_S_ti%d.png'%ti,  bbox_inches = 'tight', pad_inches = 0.05)  
    



#%%    

plt.rcParams["axes.edgecolor"] = "1.0"
plt.rcParams["axes.linewidth"]  = 2  

#Ks = np.zeros(timegrid[::10].size)
#Ksp = np.zeros(timegrid.size)
#for i,ti in enumerate(timegrid[:-10]):
#    Ks[i], Ksp[i] = ks(F[:,i],F[:,i+10])
#plt.figure(),
#plt.plot(timegrid[::10],Ks,'o')
#plt.xlabel('timestep')
#plt.ylabel('KL')
#plt.title('N=%d, h=%.3f,T=%d'%(N,h,T))



#for i,ti in enumerate(timegrid[::100]):
#    #plt.figure(),
#    #plt.hist(Y[ni,:],100,histtype='step')
#    #plt.hist(Z[:,i],100,histtype='step',density=True,label='D')
#    #plt.hist(F[:,i],100,histtype='step',density=True,label = 'S')
#    #plt.xlim(-2,2)
#    #plt.legend()
#    #plt.show()
#    fl = plt.figure(figsize=(12,8)),
#    #sns.distplot(Z[:,i],100,norm_hist =True,label='D')
#    sns.distplot(M[:,i],100,norm_hist =True,label='D')
#    sns.distplot(F[:,i],100,norm_hist =True,label='S')
#    plt.xlim(-2, 2)
#    #plt.legend()
#    #plt.plot(tspan_fineplot,np.mean(BP,axis=0),'r',lw=3,label='$\mu^{samples}$') 
#    #plt.plot(tspan_fineplot,OUms,'r--',lw=3,label='$\mu^{theory}$') 
#    #plt.plot(tspan_fineplot,(np.mean(BP,axis=0)+np.std(BP,axis=0)),color='#282828',lw=3,label='$\sigma_P^{samples}$')   
#    #plt.plot(tspan_fineplot,(np.mean(BP,axis=0)-np.std(BP,axis=0)),color='#282828',lw=3)  
#    #plt.plot(tspan_fineplot,OUms-OUCs,'#282828',linestyle='--',lw=3,label='$\sigma_P^{theory}$')   
#    #plt.plot(tspan_fineplot,OUms+OUCs,'#282828',linestyle='--',lw=3)  
#    #plt.plot([t1,t2],[y1,y2],'go',markersize=8)
#    plt.tick_params(axis='both', which='major', labelsize=15)
#    legend = plt.legend(frameon = 1,prop={'size': 15})
#    frame = legend.get_frame()
#    frame.set_facecolor('white')
#    frame.set_edgecolor('white')
#    plt.title(u'$t=%.2f$' %(ti),fontsize=15)
#    plt.xlabel('x',fontsize=20)
#    plt.ylabel('density',fontsize=20)
    
#    plt.show()
#    fl[0].savefig(plotsave+"SPARSEDeterministic_stochastic_seaborn_grad_prior_N=%d_h=%.3f_T=%d_i=%04d_seed_%02d.png"%(N,h,T,i,se), bbox_inches='tight')
#    plt.close()
#




"""
Der_P = np.zeros((N,4))
P = np.zeros((N,4))
for i,ii in enumerate([0,1,2,3]):
    plt.figure()
    plt.hist(Z[:,ii],100,histtype='step',density=True)
    P[:,i] = f_eff(Z[:,ii],0)
    Der_P[:,i] = f_effD(Z[:,ii],0)  
    plt.figure()
    plt.plot(Z[:,ii],P[:,i],'o')
    plt.plot(Z[:,ii],Der_P[:,i],'o')
    plt.show()
    
    
Der_P = np.zeros((N,4))
P = np.zeros((N,4))
inputx = np.sort(np.random.normal(loc=0, scale=0.01,size=N))
for i,ii in enumerate([0]):
    plt.figure()
    plt.hist(inputx,100,histtype='step',density=True)
    P[:,i] = f_eff(inputx,0)
    Der_P[:,i] = f_effD(inputx,0)  
    plt.figure()
    plt.plot(inputx,2*P[:,i],'o')
    plt.plot(inputx,Der_P[:,i],'o')
    plt.show()
    
"""
"""    
plt.figure()
plt.hist(xs,100)
#plt.xlim(xmin=-2.5, xmax = 2.5)
plt.show()


for ni in range(0,210,50):#Z.shape[1]):
    plt.figure(),
    #plt.hist(Y[ni,:],100,histtype='step')
    plt.hist(Z[:,ni],100,histtype='step',density=True,label='D')
    plt.hist(F[:,ni],100,histtype='step',density=True,label = 'S')
    #plt.xlim(-2,2)
    plt.title(ni)
    plt.legend()
    plt.show()
"""

col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'

from scipy.stats import skew, kurtosis  
dim=1   
#timegrid = np.arange(0,10,0.001)
sns.set(style="white")
plt.figure(figsize=(4,20))
plt.rc('axes', linewidth=2.8)
plt.rc('axes',edgecolor='#464646')
for di in range(dim):
    plt.subplot(4,1,1)
    plt.plot(timegrid, np.mean(Z[:,:],axis=0),lw=3.5,label=r'S$^\infty$',color=col_gr)
    plt.plot(timegrid, np.mean(F[:,:],axis=0),lw=3,label='S',color=col_ye)
    plt.plot(timegrid, np.mean(M[:,:],axis=0),lw=3,label='D',color=col_ro,alpha=0.8)    
    #plt.plot(timegrid, np.mean(G[:,:],axis=0),lw=3,label='KDE',color='m',alpha=0.8)    
    plt.locator_params(axis='x', nbins=4) 
    #plt.ylabel(u'mean_x_%d'%di)    
    plt.locator_params(axis='y', nbins=4)
    plt.ylabel(r'$\langle x \rangle $')          
    
        #plt.gca().set_title('mean', fontsize=14)
#    if di==dim-1:
#        plt.xlabel('time', fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
        
  
    plt.subplot(4,1,2)
    plt.plot(timegrid, np.std(Z[:,:],axis=0),lw=3.5,label=r'S$^\infty$',color=col_gr)
    plt.plot(timegrid, np.std(F[:,:],axis=0),lw=3,label='S',color=col_ye) 
    plt.plot(timegrid, np.std(M[:,:],axis=0),lw=3,label='D',color=col_ro,alpha=0.8)      
    #plt.plot(timegrid, np.std(G[:,:],axis=0),lw=3,label='KDE',color='m',alpha=0.8)      
    plt.ylabel(r'$ \sigma_{x}$')  
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4) 
    if di==0:
        plt.legend(fontsize='medium')        
#    if di==0:            
#        plt.gca().set_title('std', fontsize=14)
#    if di==dim-1:
#        plt.xlabel('time', fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
  
    plt.subplot(4,1,3)
    plt.plot(timegrid, skew(Z[:,:],axis=0),lw=3.5,label=r'S$^\infty$',color=col_gr)
    plt.plot(timegrid, skew(F[:,:],axis=0),lw=3,label='S',color=col_ye)  
    plt.plot(timegrid, skew(M[:,:],axis=0),lw=3,label='D',color=col_ro,alpha=0.8)  
    #plt.plot(timegrid, skew(G[:,:],axis=0),lw=3,label='KDE',color='m',alpha=0.8)      
    plt.ylabel(r'$s_{x} $')     
    plt.locator_params(axis='y', nbins=4) 
    plt.locator_params(axis='x', nbins=4)     
#    if di==0:            
#        plt.gca().set_title('skewness', fontsize=14)
#    if di==dim-1:
#        plt.xlabel('time', fontsize=14)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
  
    plt.subplot(4,1,4)
    plt.plot(timegrid, kurtosis(Z[:,:],axis=0),lw=3.5,label=r'S$^\infty$',color=col_gr)
    plt.plot(timegrid, kurtosis(F[:,:],axis=0),lw=3,label='S',color=col_ye) 
    plt.plot(timegrid, kurtosis(M[:,:],axis=0),lw=3,label='D',color=col_ro,alpha=0.8)      
    #plt.plot(timegrid, kurtosis(G[:,:],axis=0),lw=3,label='KDE',color='m',alpha=0.8) 
    plt.ylabel(r'$k_{x}  $')  
    plt.locator_params(axis='y', nbins=4)   
    plt.locator_params(axis='x', nbins=4)      
#    if di==0:            
#        plt.gca().set_title('kurtosis', fontsize=14)
    if di==dim-1:
        plt.xlabel('time')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)   
plt.subplots_adjust( bottom=0.1,top=0.98,right=0.98,left=0.2,wspace=0.3, hspace=0.3)
plt.savefig('Double_well_1D_dist_S_vs_S_ti_statistics_gg_new2_%dNinf.pdf'%Ninf,  bbox_inches = 'tight', pad_inches = 0)     
plt.savefig('Double_well_1D_dist_S_vs_S_ti_statistics_gg_new2_%dNinf.png'%Ninf,  bbox_inches = 'tight', pad_inches = 0)  
    
    
    