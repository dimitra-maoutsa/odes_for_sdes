# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 06:56:08 2020

@author: Dimi
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis 

col_gr = '#3D4A49'
col_ro = '#c010d0' 

plt.rc('axes', linewidth=2)
plt.rc('axes',edgecolor='#0a0a0a')
def plot_statistics(timegrid,F,labelss=['x','y','z'],labelkey = ['Sinf','S','D'], colors = [col_gr,col_ro,'gray' ],additional = None):
    ### TO DO: for 1d systems where 1st dimension is sample size check and convert arrays to 3d from 2d   
    ### F has to be a list with samples for the different systms and labelskey are the labels that will appear
    ### on the plot key that will designate for whcih system the samples come from
    num_sys = len(F)
    alphas = [0.7,0.85,0.98]
    zorder = [2,1,3]
    if len(F[0].shape)==3:
        dim = F[0].shape[0]
    else:
        dim=1
    fig_width_pt = 546.0  #1014# Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    fig1=plt.figure(figsize=fig_size)
    for di in range(dim):
        plt.subplot(dim,4,di*4+1)
        if not (additional is None):
            
            plt.plot(timegrid, additional[0][di],lw=5.5,label='A',color=col_gr, zorder=2)
        for si in range(num_sys):
            if dim>1:
                plt.plot(timegrid, np.mean(F[si][di,:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=(si+1)**2,alpha=alphas[si])
            else:
                plt.plot(timegrid, np.mean(F[si][:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=zorder[si],alpha=alphas[si])
            
                
        
        #plt.ylabel(u'mean_x_%d'%di) 
        if di==2:
            plt.ylabel(r'$\langle %s \rangle $'%labelss[di], labelpad=5)
        else:            
            plt.ylabel(r'$\langle %s \rangle $'%labelss[di])          
        if di==0:
            ### this line for 2D OU
            plt.legend(loc=9, bbox_to_anchor=(0.75,1.5),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')
            ##plt.legend(loc=9, bbox_to_anchor=(1.0,1.35),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')#state dependent
            #plt.legend(loc=9, bbox_to_anchor=(1.0,1.4),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')#state dependent
            #plt.legend(loc=9, bbox_to_anchor=(0.8,1.65),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')
            #plt.gca().set_title('mean', fontsize=14)
        if di==dim-1:
            plt.xlabel('time')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
            
      
        plt.subplot(dim,4,di*4+2)
        if not (additional is None):
            plt.plot(timegrid, np.sqrt(additional[1][di,di]),lw=5.5,label='A',color=col_gr, zorder=2)
        for si in range(num_sys):
            if dim>1:
                plt.plot(timegrid, np.std(F[si][di,:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=(si+1)**2,alpha=alphas[si])
            else:
                plt.plot(timegrid, np.std(F[si][:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=zorder[si],alpha=alphas[si])
            
                
        #plt.plot(timegrid, np.std(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        plt.ylabel(r'$ \sigma_{%s}$'%(labelss[di]))          
#        if di==0:            
#            plt.gca().set_title('std', fontsize=14)
        if di==dim-1:
            plt.xlabel('time')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
      
        plt.subplot(dim,4,di*4+3)
        if not (additional is None):
            plt.plot(timegrid, additional[2][di],lw=5.5,label='A',color=col_gr)
        for si in range(num_sys):
            if dim> 1:
                plt.plot(timegrid, skew(F[si][di,:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=(si+1)**2,alpha=alphas[si])
            else:
                plt.plot(timegrid, skew(F[si],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=zorder[si],alpha=alphas[si])
            
                
        #plt.plot(timegrid, skew(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        #plt.ylabel(r'$s_{%s} $'%(labelss[di]), labelpad=-14)   #for 2D OU ###################
        plt.ylabel(r'$s_{%s} $'%(labelss[di]), labelpad=-6)
#        if di==0:            
#            plt.gca().set_title('skewness', fontsize=14)
        if di==dim-1:
            plt.xlabel('time')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
      
        plt.subplot(dim,4,di*4+4)
        if not (additional is None):
            plt.plot(timegrid, additional[3][di],lw=5.5,label='A',color=col_gr)
        for si in range(num_sys):
            if dim> 1:
                plt.plot(timegrid, kurtosis(F[si][di,:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=(si+1)**2,alpha=alphas[si])
            else:
                plt.plot(timegrid, kurtosis(F[si][:,:],axis=0),lw=4.5,label=labelkey[si],color=colors[si],zorder=zorder[si],alpha=alphas[si])
                           
        
        #plt.plot(timegrid, kurtosis(G[di,:,:],axis=0),lw=2.5,label='D',color=col_ro)        
        if di==0:
            plt.ylabel(r'$k_{%s}  $'%(labelss[di]), labelpad=-22) 
        elif di==2:
            plt.ylabel(r'$k_{%s}  $'%(labelss[di]), labelpad=-5) 
        else:
            plt.ylabel(r'$k_{%s}  $'%(labelss[di]), labelpad=-8) 
#        if di==0:            
#            plt.gca().set_title('kurtosis', fontsize=14)
        if di==dim-1:
            plt.xlabel('time')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)   
    
    plt.subplots_adjust( wspace=0.98, hspace=0.2,bottom=0.15)
    
    #return fig1