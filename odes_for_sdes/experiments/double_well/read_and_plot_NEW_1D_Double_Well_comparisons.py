# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:53:59 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cl
import seaborn as sns
import joblib
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

col_gr = '#3D4A49'
col_ro = '#c010d0'
#%%

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
fig_width_pt = 546.0  # Get this from LaTeX using \showthe\columnwidth
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
           'ytick.right': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
plt.rcParams['patch.linewidth']=1.5
plt.rcParams["legend.fancybox"] = False
plt.rc('font',**{'family':'serif'})

#%%

#sns.set(style="white",rc={'xtick.labelsize': 12, 'ytick.labelsize': 12})
#sns.set_style("ticks", {"xtick.major.size": 0, "ytick.major.size": 10})
#sns.set_style("dark")
Ns = [ 500,1000, 1500, 2000, 2500]
Ms = [50,100,150,200]
foldername = 'fewer_n_vs_d/new_DW/'# 'Otto_dynamics_paper_data/Figure_2_1D_Double_well_vs_N_and_M/'
N_inf = 26000
###load stochastic comparison
Wasdis = dict()
for i,N in enumerate(Ns):
    Wasdis[N] = joblib.load(foldername+'DOUBLE_WELL_Wasserstein_between_stochastic%d_vs_stochastic_%d'%(N,N_inf))
    

## "Melt" the dataset to "long-form" or "tidy" representation
#iris = pd.melt(iris, "species", var_name="measurement")
sdfmax_list = []
smax_indx = 0
colum = [u'$W_1$']
for li,N in enumerate(Ns):
    smax_indx += 1
    
    sdfmax_list.append(pd.DataFrame(Wasdis[N][:,-1], columns=colum)) #stationary
    sdfmax_list[li]['N'] = N

    sdfmax_list[li]['kind'] = 'S'
    sdfmax_list[li]['M'] = 0


sdfmean_list = []
smean_indx = 0
colum = [u'$W_1$']
for li,N in enumerate(Ns):
    #add stochastic entries
    smean_indx += 1
    sdfmean_list.append(pd.DataFrame(np.mean(Wasdis[N],axis=1), columns=colum))
    sdfmean_list[li]['N'] = N
    sdfmean_list[li]['kind'] = 'S'
    sdfmean_list[li]['M'] = 0
del Wasdis


#%%
dfmax_list = []
max_indx = 0
dfmean_list = []
mean_indx = 0
#add deterministic entries
timelength = 100 # to be replaced with 10000
for li,N in enumerate(Ns): 
    for M in Ms:
        try:
            WasdisDS = joblib.load(foldername+'DOUBLE_WELL_Wasserstein_between_stochastic%d_vs_deterministic%d__M_%d'%(N_inf,N,M)) 
        except IOError:
            try:
                WasdisDS = joblib.load(foldername+'DOUBLE_WELL_Wasserstein_between_stochastic%d_vs_deterministic%d_all_to_all_M_%d_timelength_%d'%(2630,N,M,timelength)) 
            except IOError:
                print('Didnt find file')
        
            
        dfmean_list.append(pd.DataFrame(np.mean(WasdisDS,axis=1), columns=colum))
        dfmean_list[mean_indx]['N'] = N
        dfmean_list[mean_indx]['kind'] = 'D - M:%d'%M
        dfmean_list[mean_indx]['M'] = M
        mean_indx += 1
        dfmax_list.append(pd.DataFrame((WasdisDS[:,-1]), columns=colum)) #stationary
        dfmax_list[max_indx]['N'] = N
        dfmax_list[max_indx]['kind'] = 'D - M:%d'%M
        dfmax_list[max_indx]['M'] = M
        max_indx += 1
        


#concatenate dataframes

df_max=pd.concat(dfmax_list)
df_mean=pd.concat(dfmean_list)
sdf_max=pd.concat(sdfmax_list)
sdf_mean=pd.concat(sdfmean_list)
        

to_plot = [df_mean,df_max] #
sto_plot = [sdf_mean,sdf_max]

#ylabs = [u'$A$', u'$B$'] 
#cmap = plt.cm.get_cmap("twilight_shifted", 17)
#colors1 = [ cmap(4),cmap(9),cmap(11),  cmap(13),cmap(15)]
#colors1 = list(map(lambda x: cl.rgb2hex(x[:3]),colors1))
#colors_dark = [   cmap(3),cmap(10),cmap(12),cmap(14),cmap(16)]
#colors_dark = list(map(lambda x: cl.rgb2hex(x[:3]),colors_dark))

#cmap = plt.cm.get_cmap(, 17)
#
#colors1 = [ cmap(4),cmap(9),cmap(11),  cmap(13),cmap(15)]
#colors1 = list(map(lambda x: cl.rgb2hex(x[:3]),colors1))
#colors_dark = [   cmap(3),cmap(10),cmap(12),cmap(14),cmap(16)]
#colors_dark = list(map(lambda x: cl.rgb2hex(x[:3]),colors_dark))

#plt.rc('axes', linewidth=3)
plt.rc('axes',edgecolor='#464646')

#my_pal = sns.cubehelix_palette(17, start=.5, rot=-.75,as_cmap=True)
#cmap = plt.cm.get_cmap('copper', 17)
#colors1 = [ cmap(4),cmap(16),cmap(14),  cmap(12),cmap(10)]
#colors1 = list(map(lambda x: cl.rgb2hex(x[:3]),colors1))
#colors_dark = [   cmap(3),cmap(15),cmap(13),cmap(11),cmap(9)]
#colors_dark = list(map(lambda x: cl.rgb2hex(x[:3]),colors_dark))
# Initialize the figure

        
def hex_to_rgb(the_color):
    r,g,b = bytes.fromhex(the_color[1:])
    return (r,g,b,1)
cols = list(map( lambda aa: hex_to_rgb(aa),[col_gr,col_ro,col_ro,col_ro,col_ro] ))            
cols = ['k','m','m','m','m']      


#%%
#Seperate plots for mean and stationary vs N - GREEN
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1],[132/256,122/256,9/256,1]])
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=5)  
col_ro = '#208f33'     
#plt.rc('axes', linewidth=2.8)
#plt.rc('axes',edgecolor='#464646')
title_save =[ 'mean', 'stationary']
ylabs = [r'$\left< \mathcal{W}_1(P_t^{N^{\infty}},P_t^N) \right>_t$', r'$\mathcal{W}_1 (P_{\infty}^{N^{\infty}},P_{\infty}^N)$'] 
col_teal = '#237763'
for ii in range(2):
    f, ax = plt.subplots(1,1)
    #sns.despine(bottom=True, left=True)
    #sns.despine(top=True, right=True)

    # Show each observation with a scatterplot
#    sns.stripplot(x='N', y=u'$W_1$', hue='kind',palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.95, reverse=True),#c
#                  data=sto_plot[ii], dodge=.632, alpha=.12, zorder=1,ax=ax,edgecolor='gray')
#    
#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])
    
    # Show the conditional means
    g1=sns.pointplot(x='N', y=u'$W_1$', hue='kind',
                  data=to_plot[ii], dodge=.65, join=False,  color=col_ro     ,#'#464646',
                  markers=["v",'d','s','P','o'], scale=2.2, ci='sd',ax=ax,linewidth=8,edgecolor='k', capsize=.1,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=4) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=9)
    g1=sns.pointplot(x='N', y=u'$W_1$', hue='kind',
                  data=sto_plot[ii], dodge=.75, join=False,  palette=cpal,#'#464646',
                  markers=['o'], scale=2.2, ci='sd',ax=ax,linewidth=8,edgecolor='k', capsize=.1,zorder=0)
    plt.setp(g1.lines, alpha=1,linewidth=4) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=8)
    
    
    
    
    
    
    ax.set_ylabel(ylabs[ii])
    ax.set_xlabel(r'N')    
    #ax.set_yscale('log')
    #ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#    ylabels = ax.get_yticklabels()
#    ax.set_yticklabels( ylabels,ha='center')
    plt.subplots_adjust(left=0.162,right=0.95,top=0.95)
    if ii == 1:
        ax.get_legend().remove()
        #ax.set_ylim([0.007, None])

    else:
        ax.set_yscale('log')
        ax.set_ylim([0.005, 0.11])
        # Improve the legend 
        legend = ax.legend()        
        legend.get_frame().set_linewidth(1.8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title=None,
                  handletextpad=0, columnspacing=1,
                  loc=1, ncol=2, frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')
        
        
    plt.savefig('Double_well_1D_vs_N_green'+title_save[ii]+'.pdf',  bbox_inches = 'tight', pad_inches = 0.12)     
    plt.savefig('Double_well_1D_vs_N_green'+title_save[ii]+'.png',  bbox_inches = 'tight', pad_inches = 0.12)  
    
    
    
#%%
    
#plt.rc('axes', linewidth=2.8)
plt.rc('axes',edgecolor='#464646')
title_save =[ 'mean', 'max']
col_teal = '#237763'
df_meanM =  df_mean[df_mean['M'] > 0]
df_maxM =  df_max[df_max['M'] > 0]
to_plotM = [df_meanM,df_maxM] 
for ii in range(2):
    f, ax = plt.subplots(1,1,figsize=(5,5))
    #sns.despine(bottom=True, left=True)
    #sns.despine(top=True, right=True)

    # Show each observation with a scatterplot
#    sns.stripplot(x='M', y=u'$W_1$', hue='N',palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.95, reverse=True),#c
#                  data=to_plotM[ii], dodge=.632, alpha=.12, zorder=1,ax=ax,edgecolor='gray')
#    
#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])
    
    # Show the conditional means
    g1=sns.pointplot(x='M', y=u'$W_1$', hue='N',
                  data=to_plotM[ii], dodge=.75, join=False,palette='mycolorteal',# color =  col_ro     ,#'#464646',
                  markers=["v",'d','s','P','o'], scale=1.4, ci='sd',ax=ax,linewidth=5,edgecolor='k', capsize=.1)
    plt.setp(g1.lines, alpha=1,linewidth=4) 
    plt.setp(g1.collections,edgecolor=col_gr,linewidth=1.5)
    ax.set_ylabel(ylabs[ii],fontsize=22)
    ax.set_xlabel(r'M',fontsize=22)    
    #ax.set_yscale('log')
#    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#    ylabels = ax.get_yticklabels()
#    ax.set_yticklabels( ylabels,ha='center')
    plt.subplots_adjust(left=0.162,right=0.95,top=0.95)
    if ii == 2:
        ax.get_legend().remove()

    else:
        # Improve the legend 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[5:], list(map(lambda a: 'N: '+a,labels[5:])), title=None,
                  handletextpad=0, columnspacing=1,
                  loc="best", ncol=1, frameon=False,fontsize = 'medium')
        
        
    plt.savefig('Double_well_1D_vs_M_green'+title_save[ii]+'.pdf',  bbox_inches = 'tight', pad_inches = 0.11)     
    plt.savefig('Double_well_1D_vs_M_green'+title_save[ii]+'.png',  bbox_inches = 'tight', pad_inches = 0.11)  
    
    
    
#%%
    
    
#plot relative gain in wasserstein   
mean_stochmean = dict()
mean_stochmax = dict()
for NN in Ns:     
    mean_stochmax[NN] = df_max[(df_max.loc[:,'kind']=='S') & (df_max.loc[:,'N']==NN)].mean()['$W_1$']
    mean_stochmean[NN] = df_mean[(df_mean.loc[:,'kind']=='S') & (df_mean.loc[:,'N']==NN)].mean()['$W_1$']


df_mean['rW'] = df_mean.apply(lambda row: (row['$W_1$']/ mean_stochmean[row.N]) , axis = 1)    
    
df_max['rW'] = df_max.apply(lambda row: (row['$W_1$']/ mean_stochmax[row.N]) , axis = 1)    
     





#ylabs2 = [r'$\left< \mathcal{W}_1(P_t^{N^{\infty}},P_t^N) \right>_t$', r'max$_t \,\mathcal{W}_1 (P_t^{N^{\infty}},P_t^N)$'] 

ylabs2 = [r'$ \mathcal{\rho}_{{mean}}(P_D^{N},P_S^N) $', r'$ \mathcal{\rho}_{{max}}(P_D^{N},P_S^N) $'] 

plt.rc('axes', linewidth=2.8)
plt.rc('axes',edgecolor='#464646')
title_save =[ 'mean', 'max']
col_teal = '#237763'
df_meanM =  df_mean[df_mean['M'] > 0]
df_maxM =  df_max[df_max['M'] > 0]
to_plotM = [df_meanM,df_maxM] 
for ii in range(2):
    f, ax = plt.subplots(1,1,figsize=(5,5))
    #sns.despine(bottom=True, left=True)
    sns.despine(top=True, right=True)

    # Show each observation with a scatterplot
#    sns.stripplot(x='N', y='rW', hue='M',palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.95, reverse=True),#c
                  #data=to_plotM[ii], dodge=.632, alpha=.12, zorder=1,ax=ax,edgecolor='gray')
    
#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])
    
    # Show the conditional means
    sns.pointplot(x='N', y='rW', hue='M',
                  data=to_plotM[ii], dodge=.75, join=False, color =  col_teal     ,#'#464646',
                  markers=["v",'d','s','P','o'], scale=1.1, ci='sd',ax=ax,linewidth=5,  capsize=.1,scatter_kws={'linewidths':3,'edgecolor':'k'})
    
    ax.set_ylabel(ylabs2[ii],fontsize=16)
    ax.set_xlabel(r'$N$',fontsize=16)    
    #ax.set_yscale('log')
#    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#    ylabels = ax.get_yticklabels()
#    ax.set_yticklabels( ylabels,ha='center')
    plt.subplots_adjust(left=0.162,right=0.95,top=0.95)
    if ii == 2:
        ax.get_legend().remove()

    else:
        # Improve the legend 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[4:], list(map(lambda a: 'M: '+a,labels[4:])), title=None,
                  handletextpad=0, columnspacing=1,
                  loc="best", ncol=1, frameon=False,fontsize = 'medium')
        
        
    plt.savefig('Double_well_1D_vs_ratio_green'+title_save[ii]+'.pdf',  bbox_inches = 'tight', pad_inches = 0)     
    plt.savefig('Double_well_1D_vs_ratio_green'+title_save[ii]+'.png',  bbox_inches = 'tight', pad_inches = 0)  
    
        
    
    