# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:35:04 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import joblib
from scipy.stats import skew, kurtosis  
import pandas as pd
#from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mplt
from try_analytic_ou_dims import analytic_OU
foldername = 'fewer_n_vs_d/ou2d/' #'Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
#save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
#%%


#N_inf = 2000
#filename=foldername+'OU_2D_Wasserstein_between_stochastic%d_vs_stochastic%d_all_to_all'%(N_inf,N)
##Wasdis[i*20+j,ti]        
# 50,100,150,200]
#'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M)
##F[:,:,ti,repi]
#filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_deterministic_N%d_M_%d_all_to_all'%(N_inf,N,M))
#
#
#filename=save_file+'OU_2D_KDE_deterministic_trajectories%d'%(N)      
##S[:,:,ti,repi] 
#(Wasdis,filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_stochastic_N%d_all_to_all'%(N_inf,N))
#      (KLdis,filename=foldername+'OU_2D_KL_between_analytic%d_vs_stochastic_N%d_all_to_all'%(N_inf,N))
#         
#
#Wasdis[i,ti]
#KLdis[i,ti] 
#'OU_2D_Wasserstein_between_analytic%d_vs_KDE_N%d_all_to_all'%(N_inf,N))
#        
#%%

col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #

col_grn = '#208f33'


col_gr = '#3D4A49'
#colorsgr = colsg['hex']
#col_grn = colorsgr[2]
#col_ro = col_grn
#col_ye = colsp['hex'][2]
#%%

#load wasserstein
dfmax_list = []
max_indx = 0
dfmean_list = []
mean_indx = 0
colum = [u'$W_1$']
dfstat_list = []
stat_indx = 0
dfstd_list = []
std_indx = 0

sdfmax_list = []
smax_indx = 0
sdfmean_list = []
smean_indx = 0

sdfstat_list = []
sstat_indx = 0
sdfstd_list = []
sstd_indx = 0

N_inf = 2000
Ns = [ 1000, 1500, 2000, 2500]
Ms =  [50,100,150,200]
Was_D = np.zeros((20,3000,len(Ns),len(Ms)))
for ni,N in enumerate(Ns):
    for mi,M in enumerate(Ms):
        Was_D[:,:,ni,mi] = joblib.load(filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_deterministic_N%d_M_%d_all_to_all'%(N_inf,N,M))

        dfmax_list.append(pd.DataFrame(np.max(Was_D[:,:,ni,mi],axis=1), columns=colum))#max over time
        dfmax_list[max_indx]['N'] = N    
        dfmax_list[max_indx]['kind'] = 'D - M:%d'%M
        dfmax_list[max_indx]['M'] = M
        max_indx += 1
        
        
        dfstd_list.append(pd.DataFrame(np.std(Was_D[:,:,ni,mi],axis=1), columns=colum))#max over time
        dfstd_list[std_indx]['N'] = N    
        dfstd_list[std_indx]['kind'] = 'D - M:%d'%M
        dfstd_list[std_indx]['M'] = M
        std_indx += 1
        
        dfmean_list.append(pd.DataFrame(np.mean(Was_D[:,:,ni,mi],axis=1), columns=colum))
        dfmean_list[mean_indx]['N'] = N
        dfmean_list[mean_indx]['kind'] = 'D - M:%d'%M
        dfmean_list[mean_indx]['M'] = M
        mean_indx += 1
        
        dfstat_list.append(pd.DataFrame(Was_D[:,-2,ni,mi], columns=colum))#max over time
        dfstat_list[stat_indx]['N'] = N    
        dfstat_list[stat_indx]['kind'] = 'D - M:%d'%M
        dfstat_list[stat_indx]['M'] = M
        stat_indx += 1
#    if N==1000:
#        Was_KDE = joblib.load(filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_KDE_N%d_all_to_all_UPDATED'%(N_inf,N) )
#     
#        dfmax_list.append(pd.DataFrame(np.max(Was_KDE,axis=1), columns=colum))#max over time
#        dfmax_list[max_indx]['N'] = N
#    
#        dfmax_list[max_indx]['kind'] = 'KDE'#'D - M:%d'%M
#        dfmax_list[max_indx]['M'] = 0
#        max_indx += 1  
#        
#        dfstd_list.append(pd.DataFrame(np.std(Was_KDE,axis=1), columns=colum))#max over time
#        dfstd_list[std_indx]['N'] = N    
#        dfstd_list[std_indx]['kind'] = 'KDE'
#        dfstd_list[std_indx]['M'] = 0
#        std_indx += 1
#    
#        dfmean_list.append(pd.DataFrame(np.mean(Was_KDE,axis=1), columns=colum))
#        dfmean_list[mean_indx]['N'] = N
#        dfmean_list[mean_indx]['kind'] = 'KDE'#'D - M:%d'%M
#        dfmean_list[mean_indx]['M'] = 0
#        mean_indx += 1    
#        
#        dfstat_list.append(pd.DataFrame(Was_KDE[:,-2], columns=colum))#max over time
#        dfstat_list[stat_indx]['N'] = N    
#        dfstat_list[stat_indx]['kind'] = 'KDE'
#        dfstat_list[stat_indx]['M'] = 0
#        stat_indx += 1
    
    Was_S = joblib.load(filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_stochastic_N%d_all_to_all'%(N_inf,N)  )
    
    if N==2000:
        Was_S2 = joblib.load(filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_stochastic_N%d_all_to_allmissing'%(N_inf,N)  )
        Was_S[18:,:] = Was_S2[18:,:]
        
    if N==2500:
        Was_S2 = joblib.load(filename=foldername+'OU_2D_Wasserstein_between_analytic%d_vs_stochastic_N%d_all_to_allmissing'%(N_inf,N)  )
        Was_S[13:,:] = Was_S2[13:,:]
    
        
    
    
    sdfmax_list.append(pd.DataFrame(np.max(Was_S,axis=1), columns=colum))#max over time
    sdfmax_list[smax_indx]['N'] = N

    sdfmax_list[smax_indx]['kind'] = 'S'#'D - M:%d'%M
    sdfmax_list[smax_indx]['M'] = 0
    smax_indx += 1  
    
    sdfstd_list.append(pd.DataFrame(np.std(Was_S,axis=1), columns=colum))#max over time
    sdfstd_list[sstd_indx]['N'] = N   
    sdfstd_list[sstd_indx]['kind'] = 'S'
    sdfstd_list[sstd_indx]['M'] = 0
    sstd_indx += 1

    sdfmean_list.append(pd.DataFrame(np.mean(Was_S,axis=1), columns=colum))
    sdfmean_list[smean_indx]['N'] = N
    sdfmean_list[smean_indx]['kind'] = 'S'
    sdfmean_list[smean_indx]['M'] = 0
    smean_indx += 1      
    
    sdfstat_list.append(pd.DataFrame(Was_S[:,-2], columns=colum))#max over time
    sdfstat_list[sstat_indx]['N'] = N  
    sdfstat_list[sstat_indx]['kind'] = 'S'
    sdfstat_list[sstat_indx]['M'] = 0
    sstat_indx += 1
    
        
        
df_max=pd.concat(dfmax_list)
df_mean=pd.concat(dfmean_list)

df_stationary = pd.concat(dfstat_list)
df_std = pd.concat(dfstd_list)


sdf_max=pd.concat(sdfmax_list)
sdf_mean=pd.concat(sdfmean_list)

sdf_stationary = pd.concat(sdfstat_list)
sdf_std = pd.concat(sdfstd_list)
#df_max.to_pickle("./OU_2Dmaxdf.pkl") 
#df_mean.to_pickle("./OU_2Dmeandf.pkl") 



    


#%%

#ylims = [ 0.2, 0.2,0.2,0.2] 
#ylabs = [r'$\left< \mathcal{W}_1(P_t^{N^{\infty}},P_t^N) \right>_t$', r'max$_t \,\mathcal{W}_1 (P_t^{N^{\infty}},P_t^N)$'] 
ylabs = [r'$\left< \mathcal{W}_1(P_t^{A},P_t^N) \right>_t$', r'max$_t \,\mathcal{W}_1 (P_t^{A},P_t^N)$',\
         r'$ \mathcal{W}_1 (P_{\infty}^{A},P_{\infty}^N)$', r'$\sigma_t \left(\mathcal{W}_1(P_t^{A},P_t^N) \right)$'] 

plt.rc('axes', linewidth=1.5)
plt.rc('axes',edgecolor='#0a0a0a')
#plt.rcParams['text.usetex'] = True

font = {'family' : 'sans-serif',
    'weight' : 'semibold',
    'size'   : 24,
    'style'  : 'normal'}
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Arial Bold'
#plt.rcParams['mathtext.it'] = 'Arial'
#plt.rc('font', **font)

#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans:bold'
#plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
fig_width_pt = 600#546.0  # Get this from LaTeX using \showthe\columnwidth
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
           'xtick.top': True,
           'ytick.right': True, 'figure.figsize':fig_size
          }
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})

left_sp = [0.17,0.22,0.17,0.13]
title_save =[ 'mean', 'max','stationary','std']
col_teal = '#237763'

#%%
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1],[132/256,122/256,9/256,1]])
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=5)
to_plot = [df_mean,df_max,df_stationary,df_std] #
sto_plot = [sdf_mean,sdf_max,sdf_stationary,sdf_std]
ylims = [ 0.0006, 0.0015,0.0006,0.00011] 
flatui = ["#cc00cc", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
#sns.palplot(sns.color_palette(flatui))
for ii in range(4):
    f, ax = plt.subplots(1,1)
    #sns.despine(bottom=True, left=True)
    #sns.despine(top=True, right=True)
#    g0=sns.stripplot(x='N', y=u'$W_1$', hue='kind',palette=sns.color_palette(flatui),#sns.cubehelix_palette(8, start=3.6, rot=0, dark=0.5, light=.95, reverse=True),#c
#                  data=sto_plot[ii],size=5, dodge=.12, alpha=0.55, zorder=1,ax=ax,edgecolor='#e534eb',jitter=0.05,marker='o')
##    
#    # Show each observation with a scatterplot
#    sns.stripplot(x='N', y=u'$W_1$', hue='kind',palette=sns.cubehelix_palette(8, start=3.3, rot=0, dark=0.6, light=.95, reverse=True),#c
#                  data=to_plot[ii],size=5, dodge=.95, alpha=.35, zorder=1,ax=ax,edgecolor='gray',marker='P')
##    
    
#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])
    
    # Show the conditional means
    g1=sns.pointplot(x='N', y=u'$W_1$', hue='kind',
                  data=to_plot[ii], dodge=.75, join=False, color=col_ro, #color =  col_teal     ,#'#464646',
                  markers=["v",'d','s','P','o','+'], scale=2.4,errwidth=3, ci='sd',ax=ax,linewidth=8,edgecolor='gray', capsize=.06,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=5) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=10)
    g2=sns.pointplot(x='N', y=u'$W_1$', hue='kind',
                  data=sto_plot[ii], dodge=.65, join=False, palette=cpal, #color =  col_teal     ,#'#464646',
                  markers=['o'], scale=2.4,errwidth=3, ci='sd',ax=ax,linewidth=8,edgecolor='gray', capsize=.06,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=5) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=10)
    ax.set_ylabel(ylabs[ii])
    ax.set_xlabel('N')   
    ax.set_ylim(-0.00001,ylims[ii])
    ax.tick_params(direction="in",length=6)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    #ax.set_yscale('log')
#    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#    ylabels = ax.get_yticklabels()
#    ax.set_yticklabels( ylabels,ha='center')
    plt.subplots_adjust(right=0.95,top=0.95,bottom=0.15,left=left_sp[ii])
    if ii>0:
        ax.get_legend().remove()

    else:
        # Improve the legend 
        legend = ax.legend()        
        legend.get_frame().set_linewidth(1.8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:], labels[:], title=None,
                  handletextpad=0, columnspacing=1,
                  loc=3, ncol=2, frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')
            

    #plt.subplots_adjust(bottom=0.15,left=left_sp[ii])#,hspace,left,right,top,wspace) 
    plt.savefig('OU_2D_vs_N_green2'+title_save[ii]+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
    plt.savefig('OU_2D_vs_N_green2'+title_save[ii]+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
    
    
#%%
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'font.size': 20,
          'legend.fontsize': 16,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
           'xtick.top': False,
           'ytick.right': False, 'figure.figsize':fig_size
          }
plt.rcParams.update(params)   
##moments
N = 1000
M = 50
F_S = joblib.load(foldername+'OU_2D_stochastic_trajectories_N%d'%(N))

F_D = joblib.load(foldername+'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))    

#F = joblib.load(foldername+'OU_2D_KDE_deterministic_trajectories%d_UPDATED_Sl_11'%(N))
#F_KDE1 = F[:,:,:,0]
#del F
#F = joblib.load(foldername+'OU_2D_KDE_deterministic_trajectories%d_UPDATED_Sl'%(1500))
#F_KDE = F[:,:,:,0]
#del F
#foldername+'OU_2D_Wasserstein_between_analytic%d_vs_KDE_N%d_all_to_all_UPDATED_11'%(N_inf,N)
dim=2
timegrid = np.arange(0,3,0.001)
x0 = np.array([0.5, 0.5])
C_0 = np.zeros((dim,dim))
np.fill_diagonal(C_0,0.25**2)
m_t,C_t = analytic_OU(x0,C_0,timegrid)


#%%
#plot_statistics(timegrid,[F_S, F_D,F_KDE1, F_KDE],labelss=['x','y','z'],labelkey = ['S','D','KDE1','KDE'], colors = [col_gr,col_ro,'gray','k' ])
col_gr = '#3D4A49'
col_ro = '#c010d0' 
col_grn = '#208f33'

colorsgr = colsg['hex']
col_grn = colorsgr[2]
col_ro = col_grn
col_ye = colsp['hex'][2]
from plot_statistics import plot_statistics  
addi = [m_t.T, C_t.T, np.zeros((2,timegrid.size)), np.zeros((2,timegrid.size))]    
plot_statistics(timegrid,[F_S[:,:,:,0], F_D[:,:,:,0]],additional=addi,labelss=['x','y','z'],labelkey = ['S','D'], colors = [col_ye,col_grn,'gray','k' ])
 
plt.savefig('OU_2D_statistics.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_2D_statistics.png',  bbox_inches = 'tight', pad_inches = 0.1)       
    
    
#%%  

df_mtstd_list = [] 
mt_indx = 0

df_Ctstd_list = [] 
Ct_indx = 0

sdf_mtstd_list = [] 
smt_indx = 0

sdf_Ctstd_list = [] 
sCt_indx = 0
colum = u'$m_t$'#r'$\sigma\left( \hat{m}_t - m_t \right)$'
colum2 = 'ct'#r'$\sigma\left( \hat{C}_t - C_t \right)$'
for ni,N in enumerate(Ns):
    print(N)
    for mi,M in enumerate(Ms):
        print(mi)
        F_D = joblib.load(foldername+'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))     
        #std_from_mean = ([np.sum(np.std(np.mean(F_D[:,:,:,rep],axis=1)-m_t.T,axis=1)) for rep in range(20)])
        std_from_mean = ([np.mean([np.linalg.norm(np.mean(F_D[:,:,ti,rep],axis=1)-m_t[ti]) for ti in range(timegrid.size)  ] ) for rep in range(20)])
        df_mtstd_list.append(pd.DataFrame(std_from_mean))
        df_mtstd_list[mt_indx]['N'] = N    
        df_mtstd_list[mt_indx]['kind'] = 'D - M:%d'%M
        df_mtstd_list[mt_indx]['M'] = M
        mt_indx += 1
        
        std_from_Ct = ([np.mean([np.linalg.norm(np.cov(F_D[:,:,ti,rep])-C_t[ti] ,'fro') for ti in range(timegrid.size)]) for rep in range(20)])
        df_Ctstd_list.append(pd.DataFrame(std_from_Ct))
        df_Ctstd_list[Ct_indx]['N'] = N    
        df_Ctstd_list[Ct_indx]['kind'] = 'D - M:%d'%M
        df_Ctstd_list[Ct_indx]['M'] = M
        Ct_indx += 1
    F_S = joblib.load(foldername+'OU_2D_stochastic_trajectories_N%d'%(N))
    sstd_from_mean = ([np.mean([np.linalg.norm(np.mean(F_S[:,:,ti,rep],axis=1)-m_t[ti]) for ti in range(timegrid.size)  ] ) for rep in range(20)])
    #np.array([np.sum(np.std(np.mean(F_S[:,:,:,rep],axis=1)-m_t.T,axis=1)) for rep in range(20)])
    
    sdf_mtstd_list.append(pd.DataFrame(sstd_from_mean))
    sdf_mtstd_list[smt_indx]['N'] = N    
    sdf_mtstd_list[smt_indx]['kind'] = 'S'
    sdf_mtstd_list[smt_indx]['M'] = 0
    smt_indx += 1
    
    
    sstd_from_Ct = ([np.mean([np.linalg.norm(np.cov(F_S[:,:,ti,rep])-C_t[ti] ,'fro') for ti in range(timegrid.size)]) for rep in range(20)])
    sdf_Ctstd_list.append(pd.DataFrame(sstd_from_Ct))#max over time
    sdf_Ctstd_list[sCt_indx]['N'] = N    
    sdf_Ctstd_list[sCt_indx]['kind'] = 'S'
    sdf_Ctstd_list[sCt_indx]['M'] = 0
    sCt_indx += 1
        

    
    
    
    
    
df_mt=pd.concat(df_mtstd_list)
df_Ct=pd.concat(df_Ctstd_list)
sdf_mt=pd.concat(sdf_mtstd_list)
sdf_Ct=pd.concat(sdf_Ctstd_list)
    
#%%

#df_mt.to_pickle("./OU_2Dmt.pkl") 
#df_Ct.to_pickle("./OU_2DCt.pkl") 
   
    
 #%%   
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
colums = [ 0,0]    
to_plot = [df_mt,df_Ct] #
sto_plot = [sdf_mt,sdf_Ct] #
ylims0 = [-0.0001,-0.0005]
ylims = [ 0.018,0.011,0.0006,0.00015] 
#ylims = [ 0.2, 0.2,0.2,0.2] 
#ylabs = [r'$\left< \mathcal{W}_1(P_t^{N^{\infty}},P_t^N) \right>_t$', r'max$_t \,\mathcal{W}_1 (P_t^{N^{\infty}},P_t^N)$'] 
ylabs = [r'$\langle \|\hat{m}_t - m_t\|_2\rangle_t$', r'$ \langle \| \hat{C}_t - C_t \|_F \rangle_t$'] 

plt.rc('axes', linewidth=1.5)
plt.rc('axes',edgecolor='#0a0a0a')
#plt.rcParams['text.usetex'] = True

font = {'family' : 'sans-serif',
    'weight' : 'semibold',
    'size'   : 24,
    'style'  : 'normal'}
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Arial Bold'
#plt.rcParams['mathtext.it'] = 'Arial'
#plt.rc('font', **font)

#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans:bold'
#plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
fig_width_pt = 630#546.0  # Get this from LaTeX using \showthe\columnwidth546
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]


plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
col_ro = '#208f33'
left_sp = [0.17,0.22,0.17,0.13]
title_save =[ 'm_t', 'C_t','skew','kurt']
col_teal = '#237763'
for ii in range(2):
    f, ax = plt.subplots(1,1,figsize=(5.5,5.5))
    #sns.despine(bottom=True, left=True)
    #sns.despine(top=True, right=True)

    # Show each observation with a scatterplot
#    sns.stripplot(x='N', y=colums[ii], hue='kind',palette=sns.cubehelix_palette(8, start=3.3, rot=0, dark=0.6, light=.95, reverse=True),#c
#                  data=to_plot[ii],size=4, dodge=.9, alpha=.25, zorder=1,ax=ax,edgecolor='gray',marker='P')
#    sns.stripplot(x='N', y=colums[ii], hue='kind',palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0.4, light=.95, reverse=True),#c
#                  data=sto_plot[ii],size=4, dodge=.12, alpha=.35, zorder=1,ax=ax,edgecolor='gray')
##     
#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])
    
    # Show the conditional means
    g1 = sns.pointplot(x='N', y=colums[ii], hue='kind',
                  data=to_plot[ii], dodge=.75, join=False, color =  col_ro     ,#'#464646',
                  markers=["v",'d','s','P','o','+'], scale=2.4,errwidth=3, ci='sd',ax=ax,linewidth=8,edgecolor='gray', capsize=.06,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=5) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=10)
    g1 = sns.pointplot(x='N', y=colums[ii], hue='kind',
                  data=sto_plot[ii], dodge=.65, join=False,  palette=cpal     ,#'#464646',
                  markers=['o'], scale=2.42,errwidth=3, ci='sd',ax=ax,linewidth=8,edgecolor='gray', capsize=.06,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=5) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=10)
    #plt.setp(g1.lines, alpha=.6) 
    
    ax.set_ylabel(ylabs[ii])
    ax.set_xlabel('N')   
    #ax.set_ylim(ylims0[ii],ylims[ii])
    ax.tick_params(direction="in",length=6)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,-1))
#    if ii ==1:
    ax.set_yscale('log')
#    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#    ylabels = ax.get_yticklabels()
#    ax.set_yticklabels( ylabels,ha='center')
    plt.subplots_adjust(right=0.95,top=0.95,bottom=0.15,left=left_sp[ii])
    if ii >= 0:
        ax.get_legend().remove()

    else:
        # Improve the legend 
        legend = ax.legend()        
        legend.get_frame().set_linewidth(1.8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[5:], labels[5:], title=None,
                  handletextpad=0, columnspacing=1,
                  loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =1,edgecolor ='#0a0a0a')
                
    
    
    plt.savefig('OU_2D_vs_N_greenlog2'+title_save[ii]+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
    plt.savefig('OU_2D_vs_N_greenlog2'+title_save[ii]+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
    
    
    
    
    
#%%
from plot_2D_distrib import multivariateGrid    
col_ye =  '#6c226c'#'#847a09'#	'#d0c010'  #'#dfd71c' #
plt.rc('axes', linewidth=1.5)
params = {'backend': 'ps',
          'axes.labelsize': 32,
          'font.size': 20,
          'legend.fontsize': 10,
          'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'text.usetex': True,
           'xtick.top': False,
           'ytick.right': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
col_grn = '#208f33'
for ti in [ -2]:
#    N = 2000
#    M = 50
#    g=1
#    F_S = joblib.load(foldername+'OU_2D_stochastic_trajectories_N%d'%(N))
#    
#    F_D = joblib.load(foldername+'OU_2D_deterministic_trajectories_SPARSE_N_%d_moving_inducing_M_%d'%(N,M))
#    #xss = np.zeros(-2,2,100)
#    V = lambda x,y: (2*x**2 -1*x*y+2*y**2)
#
#    dfG = pd.DataFrame(F_S[:,:,ti,0].T, columns=['x','y'])
#    dfF = pd.DataFrame(F_D[:,:,ti,0].T, columns=['x','y'])
#    
#    dfG['kind'] = 'S'
#    dfF['kind'] = 'D'
#    
#    df=pd.concat([dfF])
    
    
    fig0 = multivariateGrid('x', 'y', 'kind', df=dfG,legend= ti==0,  ax_lims=[[-1.,1.],[-1.,1.]],k_is_color=col_ye, scatter_pnts=False)
    
    col_line = '#e59400'#'#6d1c11'#'#48450b'#'#3c021e'#'#acb815''#9c870f'#
    
    level_sets =  np.arange(0.1,0.9,0.25)#[0.1,0.3, 0.5, 0.7, 0.9] #np.linspace(0,1,7)
    fv = lambda x,y: np.exp(-(2/g**2)* V(x,y))
    Zx1 = np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100), sparse=False, indexing='ij')
    Zx = np.array([Zx1[0].reshape(-1,), Zx1[1].reshape(-1,)])
#    
    analyt = fv(Zx[0],Zx[1]) 
    CS = plt.contour(Zx1[0], Zx1[1], analyt.reshape(100,100),level_sets,colors=col_line,linewidths=3,alpha=0.97,linestyles='dashed',zorder=0)    
    plt.clabel(CS, inline=1, fontsize=16)
    
    plt.savefig('OU_2D_distrib'+'S'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.12)     
    plt.savefig('OU_2D_distrib'+'S'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)  
    
    col_ye = colsp['hex'][2]
    fig1 = multivariateGrid('x', 'y', 'kind', df=dfF,legend= ti==0, ax_lims=[[-1.,1.],[-1.,1.]],k_is_color=col_grn, scatter_pnts=False)
    
    analyt = fv(Zx[0],Zx[1]) 
    CS = plt.contour(Zx1[0], Zx1[1], analyt.reshape(100,100),level_sets,colors=col_line,linewidths=3,alpha=0.97,linestyles='dashed',zorder=0)    
    plt.clabel(CS, inline=1, fontsize=16)
    plt.savefig('OU_2D_distrib'+'D'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.12)     
    plt.savefig('OU_2D_distrib'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)
    
    
    

