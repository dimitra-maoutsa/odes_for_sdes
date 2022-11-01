# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:58:50 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
foldername = 'Complexity/'
reps = 20
thresholds = {1:10**(-3), 2:5*10**(-4), 3:10**(-4),4:5*10**(-5),5:10**(-5) }


#### Deterministic
N_det = np.zeros((5,20))
for thr in range(1,6):
    
    if thr <=3:
        D = joblib.load(filename=foldername+'Comp_complexity_particle_number_OU%d'%thr)
        N_det[thr-1,:] = D['N_star']
    else:
        for repi in range(reps):
            D = joblib.load(filename=foldername+'Comp_complexity_particle_number_OU%d_trial_%d'%(thr,repi))

            N_det[thr-1,repi] = (D['N_star'])[repi]
            
#%%            
#### Stochastic
N_sto = np.zeros((5,20))
for thr in range(1,6):
    
    if thr <3:
        D = joblib.load(filename=foldername+'Comp_complexity_particle_number_OU_Stoch%d'%thr)
        N_sto[thr-1,:] = D['N_star']
    else:
        for repi in range(reps):
            D = joblib.load(filename=foldername+'Comp_complexity_particle_numberStoch_OU%d_trial_%d'%(thr,repi))

            N_sto[thr-1,repi] = (D['N_star'])[repi]
            
#%%44
plt.figure()
for i in range(1,6):

    plt.plot(thresholds[i], np.mean(N_det[i-1]),'ro')
    plt.plot(thresholds[i], np.mean(N_sto[i-1]),'ko')
    plt.yscale('log')
    plt.xscale('log')
#%%

df_list = []
colum = 'N'
max_indx = 0
for thr in range(1,6):
    df_list.append(pd.DataFrame(N_det[thr-1]))#max over time
    df_list[max_indx]['KL'] = thresholds[thr] 
    max_indx += 1
df_det=pd.concat(df_list)


sdf_list = []
colum = 'N'
max_indx = 0
for thr in range(1,6):
    sdf_list.append(pd.DataFrame(N_sto[thr-1]))#max over time
    sdf_list[max_indx]['KL'] = thresholds[thr] 
    max_indx += 1
df_sto=pd.concat(sdf_list)
#%%
col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #

col_grn = '#208f33'
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1],[132/256,122/256,9/256,1]])
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=6)

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

params = {'backend': 'ps',
          'axes.labelsize': 26,
          'font.size': 20,
          'legend.fontsize': 20,
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
col_ro = '#208f33'
#%%
from matplotlib.lines import Line2D
f, ax = plt.subplots(1,1)
#g0=sns.stripplot(x='KL', y=0,color='gray',#c
#                  data=df_sto,size=5, dodge=.12, alpha=0.95, zorder=1,ax=ax,edgecolor='#e534eb',jitter=0.05,marker='o')
##    
#    # Show each observation with a scatterplot
#sns.stripplot(x='KL', y=0,  color='gray',#c
#                  data=df_det,size=5, dodge=.65, alpha=.95, zorder=1,ax=ax,edgecolor='gray',marker='o')
#

g1=sns.pointplot(x='KL', y=0,  data=df_det, dodge=.75, join=False, color =  col_ro, #color =  col_teal     ,#'#464646',
                  markers=['s','s','s','s','s'], scale=2,errwidth=3, ci='sd',ax=ax,linewidth=5,edgecolor='gray', capsize=.06)
plt.setp(g1.lines, alpha=1,linewidth=5) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2)

g2=sns.pointplot(x='KL', y=0,data=df_sto, dodge=.65, join=False, color=col_ye,markers=['o','o','o','o','o'],
                 scale=2,errwidth=3, ci='sd',ax=ax,linewidth=5,edgecolor='gray', capsize=.06)
plt.setp(g1.lines, alpha=1,linewidth=5) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2)

#ax.get_legend().remove()
plt.xlabel(r'$\langle \mathrm{KL}\left( P^A_{t}, P^N_{t}\right) \rangle_t$')
plt.ylabel(r'$N_{KL}^*$')  
plt.yscale('log')
labels = [r'$10^{-5}$' , r'$5\cdot 10^{-5}$', r'$10^{-4}$', r'$5\cdot 10^{-4}$', r'$10^{-3}$'  ]
ax.set_xticklabels(labels)
legend_elements = [Line2D([0], [0], marker='s', color='w', label='D',markeredgecolor=col_gr,markeredgewidth=2,
                          markerfacecolor=col_ro, markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='S',markeredgecolor=col_gr,markeredgewidth=2,
                          markerfacecolor=col_ye, markersize=15)]

plt.legend(handles=legend_elements, loc='best',title=None,                
                   ncol=1, frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a')


plt.savefig('Complexityb.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('Complexityb.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    

#ax.ticklabel_format(axis='x', style='sci', scilimits=(-5,-1))
#plt.xscale('log')              