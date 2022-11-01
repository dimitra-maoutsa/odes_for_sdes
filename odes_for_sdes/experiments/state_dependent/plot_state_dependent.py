# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:42:53 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import joblib
from scipy.stats import skew, kurtosis  
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.ticker as mplt
from scipy.stats import wasserstein_distance as wd

#%%
foldername = 'fewer_n_vs_d/state_dependent/'#'Otto_dynamics_paper_data/state_dependent/'#
#%%
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
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
           'xtick.top': True,
           'ytick.right': True
          }
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})
#%%

Ns = [500,1000,1500,2000,2500]
Ms = [50,100,150,200]
Ninf = 35000
sim_prec = 0.001
t_start = 0.
T=4.5
h = sim_prec #step
timegrid = np.arange(0,T,h)
#%%


S_Wass = np.zeros((len(Ns),timegrid.size,20))
for ni,N in enumerate(Ns):
    
    S_Wass[ni,:,:] = joblib.load(foldername+'Wasserstein_stochastic_N_%d'%N)


D_Wass = np.zeros((len(Ns),timegrid.size,20,len(Ms)))

for ni,N in enumerate(Ns):
    for mi,M in enumerate(Ms):
        
        D_Wass[ni,:,:,mi] = joblib.load(foldername+'Wasserstein_deterministic_N_%d_M_%d'%(N,M))
        
        
#%% plot overtime vs N
#mean_N = np.mean(np.mean(S_Wass,axis=2),axis=1)
#std_N = np.std(np.std(S_Wass,axis=2),axis=1) first acros trials then overtime
alpas = [0.45,0.45,0.45,0.25]  


colorsgr = colsg['hex']#sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.95)

mean_NS = np.mean(S_Wass[:,:],axis=(1,2))
std_NS = np.std(S_Wass[:,:],axis=(1,2))

mean_ND = np.mean(D_Wass[:,:],axis=(1,2))
std_ND = np.std(D_Wass[:,:],axis=(1,2))
from matplotlib.ticker import ScalarFormatter
plt.figure(figsize=(5,5)),

plt.plot(Ns,mean_NS,'--',label='S',lw=4,color=colsp['hex'][1])#col_ye)
plt.fill_between(Ns,mean_NS+std_NS,mean_NS-std_NS,alpha=0.45,color=colsp['hex'][2], lw=0,joinstyle='round',capstyle= 'round')
for mi,M in enumerate(Ms):
    if mi==0 or mi==3:
        if mi== 3:
            mmi = 0
        elif mi==0:
            mmi=2
        plt.plot(Ns,mean_ND[:,mi],label='D - M:%d'%Ms[mi],lw=4, color=(colorsgr)[mmi]) 
    
        plt.fill_between(Ns,mean_ND[:,mi]+std_ND[:,mi],mean_ND[:,mi]-std_ND[:,mi],alpha=alpas[mi],capstyle= 'round',joinstyle='round', color=(colorsgr)[mmi], lw=0)
#plt.plot(Ns,np.mean(np.mean(D_Wass[:,:,:,-1],axis=2),axis=1),'.',label='N=%d'%N)         
plt.yscale('log') 
#plt.ylim([0.001,None])
plt.xlabel('N')    
plt.ylabel( r'$\langle \mathcal{W}_1(P_t^{A},P_t^N) \rangle_t$')   
plt.legend()

ax = plt.gca()
#ax.tick_params(direction="in",length=6)
#font1 = {'family' : 'sans-serif',
#    'weight' : 'semibold',
#    'style'  : 'normal'}
#ax.set_xticklabels(ax.get_xticks(), font1)
#ax.set_yticklabels(ax.get_yticks(), font1)
plt.locator_params(axis='x', nbins=6)

ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=True, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))
#ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))

legend = ax.legend()        
legend.get_frame().set_linewidth(1.8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title=None,
          handletextpad=0.5, columnspacing=0,
          loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.3)
        


plt.savefig('State_1D_vs_N_green'+'W1_overtime'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('State_1D_vs_N_green'+'W1_overtime'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
    


#ax.yaxis.major.formatter._useMathText = True
#from matplotlib.ticker import ScalarFormatter
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#%% plot overtime vs M
#mean_N = np.mean(np.mean(S_Wass,axis=2),axis=1)
#std_N = np.std(np.std(S_Wass,axis=2),axis=1) first acros trials then overtime

mean_NS = np.mean(S_Wass,axis=(1,2))
std_NS = np.std(S_Wass,axis=(1,2))

mean_ND = np.mean(D_Wass,axis=(1,2))
std_ND = np.std(D_Wass,axis=(1,2))

plt.figure(),

#plt.plot(Ns,mean_NS,'--',label='S',lw=3)
#plt.fill_between(Ns,mean_NS+std_NS,mean_NS-std_NS,alpha=0.25)
for ni,N in enumerate(Ns):
    
        plt.plot(Ms,mean_ND[ni,:],label='N=%d'%Ns[ni],lw=3) 
    
        #plt.fill_between(Ms,mean_ND[ni,:]+std_ND[ni,:],mean_ND[ni,:]-std_ND[ni,:],alpha=0.45)
#plt.plot(Ns,np.mean(np.mean(D_Wass[:,:,:,-1],axis=2),axis=1),'.',label='N=%d'%N)         
#plt.yscale('log')        
plt.legend()

#%%
det_W1_list = []
max_indx = 0
colum = ['w1']

for ni,N in enumerate(Ns):
    for mi,M in enumerate(Ms):       
        
        #print(max_indx)
        det_W1_list.append(pd.DataFrame(np.mean(D_Wass[ni,:,:,mi],axis=0), columns=colum))
        det_W1_list[max_indx]['N'] = N
        det_W1_list[max_indx]['M'] = M
        max_indx += 1   
det_W1=pd.concat(det_W1_list)




#%%

f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))
#sns.despine(bottom=True, left=True)
#sns.despine(top=True, right=True)

ii=0
# Show each observation with a scatterplot
#sns.stripplot(x='N', y=u'$KL^{T}$', hue='D', color=col_gr,#palette=sns.color_palette(colors1),
#              data=df_klend, dodge=True, alpha=.1, zorder=1,ax=ax[ii])

#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])

# Show the conditional means

g1=sns.pointplot(x='M', y='w1', hue='N',
              data=det_W1, dodge=.732,   palette='mycolorteal',#color=col_grn,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,linestyles='-',edgecolor='gray',errwidth=3, join=True,alpha=0.6,zorder=0)
plt.setp(g1.lines, alpha=.98,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
#g2=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
#              data=df_klmiddle, dodge=.932,  color=col_grn,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
#              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True)#, capsize=0.2)
#
#plt.setp(g2.lines, alpha=.8,linewidth=4) 
#plt.setp(g2.collections,edgecolor=col_gr,linewidth=1.5)

#ax.set_ylabel(u'$\langle $KL$(P_{\infty}^{A},P_{\infty}^N) \rangle_t $',fontsize=22)
ax.set_ylabel(r'$\langle \mathcal{W}_1(P_t^{A},P_t^N) \rangle_t$')
ax.set_xlabel('M')    
ax.set_yscale('log')
plt.ylim([0.0021,0.008])
#ax.tick_params(direction="in",length=6)
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')

#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center',rotation=90)

#ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -2))
#import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=10**(-3), useMathText=True)
#ax.ticklabel_format()
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax.get_yaxis().get_offset_text().offset_text_position = "top"
## Get the offset value
#offset = ax.yaxis.get_offset_text()
#colors = ['black', 'red', 'green']
#lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
#labels = ['black data', 'red data', 'green data']
#plt.legend(lines, labels)
#
#l1 = plt.legend([p1], ['D', 'S'], handletextpad=0, columnspacing=1, loc="best", ncol=1, frameon=False)   
# Improve the legend 
handles, labels = ax.get_legend_handles_labels()

labels2 = list(map(lambda x: 'N:%s'%x,labels ))

#l1 = ax.legend([handles[-2],handles[2]], ['D','S'], title=None,
#          handletextpad=0, columnspacing=-0.10,
#          loc="best", ncol=2, fontsize=16,bbox_to_anchor=(0.35, 0.11),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.1)

ax.legend(handles, labels2, title=None,
          handletextpad=0, columnspacing=-0.1,
          ncol=3, fontsize=16,bbox_to_anchor=(0.070, 0.79),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.2)         
     

#ax.add_artist(l1) 

plt.savefig('State_1D_vs_M_green'+'W1_overtime'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('State_1D_vs_M_green'+'W1_overtime'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 




#%%
#mean_N = np.mean(np.mean(S_Wass,axis=2),axis=1)
#std_N = np.std(np.std(S_Wass,axis=2),axis=1) first acros trials then overtime

mean_NS_stat = np.mean(S_Wass[:,-1],axis=(1))
std_NS_stat = np.std(S_Wass[:,-1],axis=(1))

mean_ND_stat = np.mean(D_Wass[:,-1],axis=(1))
std_ND_stat = np.std(D_Wass[:,-1],axis=(1))

#plt.figure(),
#
#plt.plot(Ns,mean_NS_stat,'--',label='S',lw=3)
#plt.fill_between(Ns,mean_NS_stat+std_NS_stat,mean_NS_stat-std_NS_stat,alpha=0.25)
#for mi,M in enumerate(Ms):
#    if mi==0 or mi==3:
#        plt.plot(Ns,mean_ND_stat[:,mi],label='M=%d'%Ms[mi],lw=3) 
#    
#        plt.fill_between(Ns,mean_ND_stat[:,mi]+std_ND_stat[:,mi],mean_ND_stat[:,mi]-std_ND_stat[:,mi],alpha=0.45)
##plt.plot(Ns,np.mean(np.mean(D_Wass[:,:,:,-1],axis=2),axis=1),'.',label='N=%d'%N)         
#plt.yscale('log')        
#plt.legend()

plt.figure(figsize=(5,5)),

plt.plot(Ns,mean_NS,'--',label='S',lw=4,color=colsp['hex'][1])
plt.fill_between(Ns,mean_NS+std_NS,mean_NS-std_NS,alpha=0.45,color=colsp['hex'][2], lw=0,joinstyle='round',capstyle= 'round')
for mi,M in enumerate(Ms):
    if mi==0 or mi==3:
        if mi== 3:
            mmi = 0
        elif mi==0:
            mmi=2
        plt.plot(Ns,mean_ND_stat[:,mi],label='D - M:%d'%Ms[mi],lw=4, color=colorsgr[mmi]) 
    
        plt.fill_between(Ns,mean_ND_stat[:,mi]+std_ND_stat[:,mi],mean_ND_stat[:,mi]-std_ND_stat[:,mi],alpha=alpas[mi],capstyle= 'round',joinstyle='round', color=colorsgr[mmi], lw=0)
#plt.plot(Ns,np.mean(np.mean(D_Wass[:,:,:,-1],axis=2),axis=1),'.',label='N=%d'%N)         
plt.yscale('log') 
#plt.ylim([0.001,None])
plt.xlabel('N')    
plt.ylabel( r'$ \mathcal{W}_1(P_{\infty}^{A},P_{\infty}^N) $')   
plt.locator_params(axis='x', nbins=6)


ax = plt.gca()
#font1 = {'family' : 'sans-serif',
#    'weight' : 'semibold',
#    'style'  : 'normal'}
#ax.set_xticklabels(ax.get_xticks(), font1)
#ax.set_yticklabels(ax.get_yticks(), font1)


ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))

#legend = ax.legend()        
#legend.get_frame().set_linewidth(1.8)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, title=None,
#          handletextpad=0.5, columnspacing=0,
#          loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.3)
#        


plt.savefig('State_1D_vs_N_green'+'W1_stat'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('State_1D_vs_N_green'+'W1_stat'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    

#%%
det_W1_liststat = []
max_indx = 0
colum = ['w1']

for ni,N in enumerate(Ns):
    for mi,M in enumerate(Ms):       
        
        #print(max_indx)
        det_W1_liststat.append(pd.DataFrame(D_Wass[ni,-1,:,mi], columns=colum))
        det_W1_liststat[max_indx]['N'] = N
        det_W1_liststat[max_indx]['M'] = M
        max_indx += 1   
det_W1_stat=pd.concat(det_W1_liststat)




#%%

f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))
#sns.despine(bottom=True, left=True)
#sns.despine(top=True, right=True)

ii=0
# Show each observation with a scatterplot
#sns.stripplot(x='N', y=u'$KL^{T}$', hue='D', color=col_gr,#palette=sns.color_palette(colors1),
#              data=df_klend, dodge=True, alpha=.1, zorder=1,ax=ax[ii])

#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])

# Show the conditional means

g1=sns.pointplot(x='M', y='w1', hue='N',
              data=det_W1_stat, dodge=.732,  palette='mycolorteal',# color=col_grn,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,linestyles='-',edgecolor='gray',errwidth=3, join=True,alpha=0.96,zorder=0)
plt.setp(g1.lines, alpha=.8,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=1.5,zorder=100)
#g2=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
#              data=df_klmiddle, dodge=.932,  color=col_grn,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
#              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True)#, capsize=0.2)
#
#plt.setp(g2.lines, alpha=.8,linewidth=4) 
#plt.setp(g2.collections,edgecolor=col_gr,linewidth=1.5)

#ax.set_ylabel(u'$\langle $KL$(P_{\infty}^{A},P_{\infty}^N) \rangle_t $',fontsize=22)
ax.set_ylabel(r'$ \mathcal{W}_1(P_{\infty}^{A},P_{\infty}^N)$')
ax.set_xlabel('M')    
ax.set_yscale('log')
#plt.ylim([0.0021,0.008])
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')

#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center',rotation=90)

#ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -2))
#import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=10**(-3), useMathText=True)
#ax.ticklabel_format()
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax.get_yaxis().get_offset_text().offset_text_position = "top"
## Get the offset value
#offset = ax.yaxis.get_offset_text()
#colors = ['black', 'red', 'green']
#lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
#labels = ['black data', 'red data', 'green data']
#plt.legend(lines, labels)
#
#l1 = plt.legend([p1], ['D', 'S'], handletextpad=0, columnspacing=1, loc="best", ncol=1, frameon=False)   
# Improve the legend 
#handles, labels = ax.get_legend_handles_labels()
#
#labels2 = list(map(lambda x: 'N : %s'%x,labels ))
#
##l1 = ax.legend([handles[-2],handles[2]], ['D','S'], title=None,
##          handletextpad=0, columnspacing=-0.10,
##          loc="best", ncol=2, fontsize=16,bbox_to_anchor=(0.35, 0.11),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.1)
#
#ax.legend(handles, labels2, title=None,
#          handletextpad=0, columnspacing=-0.1,
#          ncol=3, fontsize=16,bbox_to_anchor=(0.070, 0.79),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.2)         
     
ax.legend().remove()
#ax.add_artist(l1) 

plt.savefig('State_1D_vs_M_green'+'W1_stat'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('State_1D_vs_M_green'+'W1_stat'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 

#%%
#mean_N = np.mean(np.mean(S_Wass,axis=2),axis=1)
#std_N = np.std(np.std(S_Wass,axis=2),axis=1) first acros trials then overtime



plt.figure(),

#plt.plot(Ns,mean_NS,'--',label='S',lw=3)
#plt.fill_between(Ns,mean_NS+std_NS,mean_NS-std_NS,alpha=0.25)
for ni,N in enumerate(Ns):
    
        plt.plot(Ms,mean_ND_stat[ni,:],label='N=%d'%Ns[ni],lw=3) 
    
        #plt.fill_between(Ms,mean_ND[ni,:]+std_ND[ni,:],mean_ND[ni,:]-std_ND[ni,:],alpha=0.45)
#plt.plot(Ns,np.mean(np.mean(D_Wass[:,:,:,-1],axis=2),axis=1),'.',label='N=%d'%N)         
plt.yscale('log')        
plt.legend()

#%%

dat = joblib.load(foldername+'data_for_statedependent_sin_linearf_for_plot_statistics') #for OU F

Sinf = dat['Z']
S = dat['F']
D = dat['M']
timegrid = dat['timegrid']

#%%  plot distributions

N=1000
M=100
D1=joblib.load(foldername+'State_dependent_DW_sin_deterministic_trajectories_N_%d_M_%d'%(N,M))
D=D1[:,:,1]
D1 = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectories_N_%d'%(N) )
S = D1[:,:,1]
Sinf = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectory_Ninf_%d'%(Ninf) )
#%%
del dat

#%%

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
#%%
from plot_statistics import plot_statistics
col_gr = '#3D4A49'
colorsgr = colsg['hex']
col_grn = colorsgr[2]
col_ye = colsp['hex'][2]
plot_statistics(timegrid,[Sinf,S,D],labelss=['x','y','z'],labelkey = [r'S$_{\infty}$','S','D'], colors = [col_gr2,col_ye,col_grn ])
plt.savefig('State_1D_vs_M_green'+'statistics'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('State_1D_vs_M_green'+'statistics'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 


#%%
col_gr = '#3D4A49'
colorsgr = colsg['hex']
col_ro = colorsgr[2]
col_ye = colsp['hex'][2]
plt.rc('axes', linewidth=2.8)
plt.rc('axes',edgecolor='#464646')
#sns.set(style="white",rc={'xtick.labelsize': 12, 'ytick.labelsize': 12})
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = False

params = {'backend': 'ps',
          'axes.labelsize': 38,
          'font.size': 20,
          'legend.fontsize': 10,
          'xtick.labelsize': 34,
          'ytick.labelsize': 34,
          'text.usetex': True,
           'xtick.top': False,
           'ytick.right': False
          }
plt.rcParams.update(params)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

for ti in [101,4400,3200]:
    f, ax = plt.subplots(1,1,figsize=(5,5))
    sns.despine(bottom=True, left=True)
    sns.distplot(Sinf[:,ti],100,hist=True, kde=True, label=r'S$^\infty$',color=col_gr,ax=ax,kde_kws={"lw":"3"})
    sns.distplot(D[:,ti],100,hist=True, kde=True, label='D',color=col_ro,ax=ax,kde_kws={"lw":"3"})
    #sns.distplot(F1[:,ti],50,hist=True, kde=True, label='S',color='b')
    #plt.ylabel('%s'%labelss[di])
    plt.xlabel(r'$x$')     
    plt.ylabel(r'${P}(x)$')     
    plt.ylim([0,2.9])
    plt.locator_params(axis='y', nbins=3)
    if ti<1000:
        plt.legend(frameon=False,fontsize=30,loc=1) 
    plt.subplots_adjust(left=0.17,right=0.95,top=0.980)
    
    
    if ti>101:
           
        axins = zoomed_inset_axes(ax, 2, loc=1)
        sns.distplot(Sinf[:,ti],100,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=axins,kde_kws={"lw":"3"})
        sns.distplot(D[:,ti],100,hist=True, kde=True, label='D',color=col_ro,ax=axins,kde_kws={"lw":"3"})
        if ti== 3200:
            x1, x2, y1, y2 = 1.1,1.3, 0.5, 1.5 # specify the limits # specify the limits
        elif ti==101:
            x1, x2, y1, y2 = 1.2,1.4, 0.3, 0.8
        elif ti== 4400:
            x1, x2, y1, y2 = 1.3,1.6, 0, 0.5
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2")
    
    axins2 = zoomed_inset_axes(ax, 2, loc=2)
    sns.distplot(Sinf[:,ti],100,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=axins2,kde_kws={"lw":"3"})
    sns.distplot(D[:,ti],100,hist=True, kde=True, label='D',color=col_ro,ax=axins2,kde_kws={"lw":"3"})
    if ti== 3200:
        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5 # specify the limits # specify the limits
    elif ti==101:
        x1, x2, y1, y2 = 0.45,0.7, 0., 0.5
    elif ti==4400:
        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5
    axins2.set_xlim(x1, x2) # apply the x-limits
    axins2.set_ylim(y1, y2) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.2")
    
    plt.savefig('State_dependent_1D_dist_D_vs_S_ti%d_gg.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0)     
    plt.savefig('State_dependent_1D_dist_D_vs_S_ti%d_gg.png'%ti,  bbox_inches = 'tight', pad_inches = 0)  
    
    
for ti in [101,4400,3200]:
    f, ax = plt.subplots(1,1,figsize=(5,5))
    sns.despine(bottom=True, left=True)
    sns.distplot(Sinf[:,ti],60,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=ax,kde_kws={"lw":"3"})
    #sns.distplot(Fd1[:,ti],50,hist=True, kde=True, label='D',color=col_ro)
    sns.distplot(S[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=ax,kde_kws={"lw":"3"})
    #plt.ylabel('%s'%labelss[di])
    plt.xlabel(r'$x$')    
    plt.ylim([0,2.9])
    plt.ylabel(r'${P}(x)$')      
    plt.locator_params(axis='y', nbins=3)
    if ti<1000:
        plt.legend(frameon=False,fontsize=30,loc=1)
    plt.subplots_adjust(left=0.17,right=0.95,top=0.980) 

    if ti>101:
        
       
        axins = zoomed_inset_axes(ax, 2, loc=1)
        sns.distplot(Sinf[:,ti],60,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=axins,kde_kws={"lw":"3"})    
        sns.distplot(S[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=axins,kde_kws={"lw":"3"})
        if ti== 3200:
            x1, x2, y1, y2 = 1.1,1.3, 0.5, 1.5 # specify the limits # specify the limits
        elif ti==101:
            x1, x2, y1, y2 = 1.2,1.4, 0.3, 0.8
        elif ti== 4400:
            x1, x2, y1, y2 = 1.3,1.6, 0, 0.5
            
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.2")
        
    axins2 = zoomed_inset_axes(ax, 2, loc=2)
    sns.distplot(Sinf[:,ti],60,hist=True, kde=True, label=r'S$_\infty$',color=col_gr,ax=axins2,kde_kws={"lw":"3"})    
    sns.distplot(S[:,ti],60,hist=True, kde=True, label='S',color=col_ye,ax=axins2,kde_kws={"lw":"3"})
    if ti== 3200:
        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5 # specify the limits # specify the limits
    elif ti==101:
        x1, x2, y1, y2 = 0.45,0.7, 0., 0.5
    elif ti==4400:
        x1, x2, y1, y2 = 0.35,0.6, 0., 0.5
    axins2.set_xlim(x1, x2) # apply the x-limits
    axins2.set_ylim(y1, y2) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins2, loc1=4, loc2=3, fc="none", ec="0.2")
      
    plt.savefig('State_dependent_1D_dist_S_vs_S_ti%d.pdf'%ti,  bbox_inches = 'tight', pad_inches = 0)     
    plt.savefig('State_dependent_1D_dist_S_vs_S_ti%d.png'%ti,  bbox_inches = 'tight', pad_inches = 0)  
    
    
    

#%%

#S_Wass = np.zeros((len(Ns),timegrid.size,20))
#Dinf = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectory_Ninf_%d'%(Ninf) )
#for ni,N in enumerate(Ns):
#    D = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectories_N_%d'%(N) ) 
#    for tr in range(20):
#        print('N: %d, trial: %d' %(N,tr))
#        for ti in range(timegrid.size):
#            
#            S_Wass[ni,ti,tr] =  wd(Dinf[:,ti],D[:,ti,tr])
#        


#%%
            
#S_Wass = np.zeros((timegrid.size,20))
#Dinf = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectory_Ninf_%d'%(Ninf) )
#for ni,N in enumerate(Ns):
#    D = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectories_N_%d'%(N) ) 
#    for tr in range(20):
#        print('N: %d, trial: %d' %(N,tr))
#        for ti in range(timegrid.size):
#            
#            S_Wass[ti,tr] =  wd(Dinf[:,ti],D[:,ti,tr])
#        
#
#    joblib.dump(S_Wass, filename=foldername+'Wasserstein_stochastic_N_%d'%N)        

#%%
    
#D_Wass = np.zeros((timegrid.size,20))
#Dinf = joblib.load(foldername+'State_dependent_DW_sin_stochastic_trajectory_Ninf_%d'%(Ninf) )
#for ni,N in enumerate(Ns):
#    for mi,M in enumerate(Ms):
#        
#        D= joblib.load(foldername+'State_dependent_DW_sin_deterministic_trajectories_N_%d_M_%d'%(N,M)) 
#        
#        for tr in range(20):
#            print('N: %d, trial: %d' %(N,tr))
#            for ti in range(timegrid.size):
#                
#                D_Wass[ti,tr] =  wd(Dinf[:,ti],D[:,ti,tr])
#            
#
#        joblib.dump(D_Wass, filename=foldername+'Wasserstein_deterministic_N_%d_M_%d'%(N,M))
