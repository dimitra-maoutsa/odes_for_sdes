# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:05:09 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
save_file='/home/dimitra/code/code/Oct18/Otto_dynamics_paper_data/Figure_vs_dim/'

from matplotlib.lines import Line2D
import matplotlib.cm as cm
#%%
from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
#newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1]])
newcolors = np.array([[48/256,44/256,3/256],[119/256,110/256,8/256,1],[167/256,154/256,12/256,1],[215/256,165/256,15/256,1]])
#48, 44, 3  ;119, 110, 8);167, 154, 12;215, 198, 15  ;243, 205, 88 ;215, 165, 15
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=4)

col_gr = '#3D4A49'
col_ro = '#208f33'#'#c010d0'
col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #

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
#%%
#save_file = 'fewer_n_vs_d/updated/'

dims =[2,3,4,5]
Ns = np.arange(500,7000,500)
M = 100
h = 0.001
t_start = 0
T = 3
timegrid = np.arange(0,T,h)
#%%
#klls = dict()
#means = dict()
#covs = dict()
#m_ts = dict()
#C_ts = dict()
#for di,dim in enumerate(dims):
#    for ni,N in enumerate(Ns):
#        try:
#            filenm = 'N_vs_dims_for_OU_with_KL_opt_N_for_M_%d_for_dim_%d_FEWERNs_UPDATED_05_N_%d'%(M,dim,N)
#        #filenm = 'N_vs_dims_for_OU_with_KL_opt_N_for_M_%d_for_dim_%d_FEWERNs'%(M,dim)
#    
#        except OSError as e:
#            print(filename)
#        D = joblib.load(save_file + filenm)
#        if ni==0:      
#            klls[dim] = D['kll']
#             
#            means[dim] = D['means']
#             
#            covs[dim] = D['covs']
#             
#            m_ts[dim] = D['m_t']
#            C_ts[dim] = D['C_t']
#        else:
#            klls[dim][:,:,ni] = D['kll'][:,:,ni]
#             
#            means[dim][:,:,:,ni] = D['means'][:,:,:,ni]
#             
#            covs[dim][:,:,:,ni] = D['covs'][:,:,:,ni]
#             
#            


#%%
#Didi = dict()
#Didi['klls'] = klls
#Didi['means'] = means
#Didi['covs'] = covs
#Didi['m_ts'] = m_ts
#Didi['C_ts'] = C_ts
#Didi['Ns'] = Ns
#joblib.dump(Didi,filename='Data_for_Dims_OUs')
            
#%% Load deterministic data
            
Didi = joblib.load('Data_for_Dims_OUs')    
klls = Didi['klls'] 
means = Didi['means'] 
covs = Didi['covs']
m_ts = Didi['m_ts'] 
C_ts = Didi['C_ts'] 
Ns = Didi['Ns'] 
       
#%%
means[6][:,:,17:,12] = np.nan    
means[5][:,:,19,9] = np.nan 

covs[6][:,:,:,17:,12] = np.nan    
covs[5][:,:,:,19,9] = np.nan         
            
#%% improved stoping times 
       
tim = dict()
plt.figure()        
for dim in dims:
    ri = np.where(m_ts[dim][:,0]<=m_ts[2][-1,0])[0][0]
    tim[dim] = int(ri)
    #plt.plot(dim,m_ts[dim][ri,0],'.')
    

#%% Load stochastic data 

Didi = joblib.load('Data_for_Dims_OUs_Stochastic')    
sklls = Didi['klls'] 
smeans = Didi['means'] 
scovs = Didi['covs']
sm_ts = Didi['m_ts'] 
sC_ts = Didi['C_ts'] 

       


#%% Dataframe for deterministic stationary KL
           
df_final_kl_list = []
max_indx = 0
colum = [u'$KL^{T}$']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        
        #print(max_indx)
        df_final_kl_list.append(pd.DataFrame(klls[dim][tim[dim],:,ni], columns=colum))
        df_final_kl_list[max_indx]['N'] = N
        df_final_kl_list[max_indx]['D'] = dim
        max_indx += 1
        
df_klend=pd.concat(df_final_kl_list) 
#%% Dataframe for stochastic stationary KL
  

sdf_final_kl_list = []
max_indx = 0
colum = [u'$KL^{T}$']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        #print(max_indx)
        sdf_final_kl_list.append(pd.DataFrame(sklls[dim][tim[dim],:,ni], columns=colum))
        sdf_final_kl_list[max_indx]['N'] = N
        sdf_final_kl_list[max_indx]['D'] = dim
        max_indx += 1


sdf_klend=pd.concat(sdf_final_kl_list)   
#%%
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


#%%Dataframe for MEAN over time KL for deterministic
           
df_middle_kl_list = []
max_indx = 0
colum = [u'$KL^{T}$']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        
        #print(max_indx)
        df_middle_kl_list.append(pd.DataFrame(np.mean(klls[dim][:,:,ni],axis=0), columns=colum))
        df_middle_kl_list[max_indx]['N'] = N
        df_middle_kl_list[max_indx]['D'] = dim
        max_indx += 1
        


df_klmiddle=pd.concat(df_middle_kl_list)   
#%%Dataframe for MEAN over time KL for stochastic
           
sdf_middle_kl_list = []
max_indx = 0
colum = [u'$KL^{T}$']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        
        #print(max_indx)
        sdf_middle_kl_list.append(pd.DataFrame(np.mean(sklls[dim][:,:,ni],axis=0), columns=colum))
        sdf_middle_kl_list[max_indx]['N'] = N
        sdf_middle_kl_list[max_indx]['D'] = dim
        max_indx += 1
        


sdf_klmiddle=pd.concat(sdf_middle_kl_list)  
#%% PLOT KL vs N OVERTIME for both
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

g1=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
              data=sdf_klmiddle, dodge=.932,  palette='mycoloryellow',#   palette='mycolorpurple',# palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,linestyles='--',edgecolor='gray',errwidth=3, join=True,alpha=0.6,zorder=0)
plt.setp(g1.lines, alpha=.5,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
              data=df_klmiddle, dodge=.932,    palette= sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True,zorder=0)#, capsize=0.2)

plt.setp(g2.lines, alpha=.5,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)

#ax.set_ylabel(u'$\langle $KL$(P_{\infty}^{A},P_{\infty}^N) \rangle_t $',fontsize=22)
ax.set_ylabel(r'$\langle $KL$(P_{t}^{A},P_{t}^N)\rangle_t $',fontsize=22)
ax.set_xlabel(r'N',fontsize=22)    
ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')

#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center',rotation=90)
xlabels = ax.get_xticklabels()
for li in range(len(xlabels)):
    if not(li%2==0):
        xlabels[li] = ''
ax.set_xticklabels( xlabels)


#colors = ['black', 'red', 'green']
#lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
#labels = ['black data', 'red data', 'green data']
#plt.legend(lines, labels)
#
#l1 = plt.legend([p1], ['D', 'S'], handletextpad=0, columnspacing=1, loc="best", ncol=1, frameon=False)   
# Improve the legend 
handles, labels = ax.get_legend_handles_labels()

labels2 = list(map(lambda x: 'D:%s'%x,labels ))

l1 = ax.legend([handles[-2],handles[2]], ['D','S'], title=None,
          handletextpad=-0.2, columnspacing=-0.10,
          loc="best", ncol=2, fontsize=16,bbox_to_anchor=(0.328, 0.106),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.1)

ax.legend(handles[4:8], labels2[4:8], title=None,
          handletextpad=-0.25, columnspacing=-0.05,
          loc="best", ncol=2, fontsize=16,bbox_to_anchor=(0.565, 0.800),frameon=True,shadow=None,framealpha =1,edgecolor ='#0a0a0a', borderpad=0.2)         
     

ax.add_artist(l1) 

plt.savefig('OU_ND_vs_N_green'+'KL_overtime'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_N_green'+'KL_overtime'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
    
#%% PLOT KL vs N OVERTIME for both
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

g1=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
              data=sdf_klend, dodge=.832,  palette='mycoloryellow',#   palette='mycolorpurple',# palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,linestyles='--',edgecolor='gray',errwidth=3, join=True,alpha=0.6,zorder=0)
plt.setp(g1.lines, alpha=.5,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='N', y=u'$KL^{T}$', hue='D',
              data=df_klend, dodge=.832,    palette= sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','P'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True, zorder=5) #capsize=0.2,

plt.setp(g2.lines, alpha=.5,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)

ax.set_ylabel(u'$ $KL$(P_{\infty}^{A},P_{\infty}^N) $',fontsize=22)
#ax.set_ylabel(r'$\langle $KL$(P_{t}^{A},P_{t}^N)\rangle_t $',fontsize=22)
ax.set_xlabel(r'N',fontsize=22)    
ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')

#ylabels = ax.get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center',rotation=90)
xlabels = ax.get_xticklabels()
for li in range(len(xlabels)):
    if not(li%2==0):
        xlabels[li] = ''
ax.set_xticklabels( xlabels)


#colors = ['black', 'red', 'green']
#lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
#labels = ['black data', 'red data', 'green data']
#plt.legend(lines, labels)
#
#l1 = plt.legend([p1], ['D', 'S'], handletextpad=0, columnspacing=1, loc="best", ncol=1, frameon=False)   
# Improve the legend 
#handles, labels = ax.get_legend_handles_labels()
#
#labels2 = list(map(lambda x: 'D = %s'%x,labels ))
#
#l1 = ax.legend([handles[-2],handles[3]], ['D','S'], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.34, 0.12))
#
#ax.legend(handles[4:8], labels2[4:8], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.4, 0.8))         
#     
#
#ax.add_artist(l1) 
ax.legend().remove()
plt.savefig('OU_ND_vs_N_green'+'KL_end'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_N_green'+'KL_end'+'.png',  bbox_inches = 'tight', pad_inches = 0.1)  
    
        
#%% Stationary means DETERMINISTIC
           
df_final_mean_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            for di in range(dim):
                df_final_mean_list.append(pd.DataFrame(np.sqrt(means[dim][di,-1,:,ni]**2), columns=colum))
                df_final_mean_list[max_indx]['N'] = N
                df_final_mean_list[max_indx]['D'] = dim
                max_indx += 1
        


df_meanend=pd.concat(df_final_mean_list)   

#%% Stationary means Stochastic
           
sdf_final_mean_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            for di in range(dim):
                sdf_final_mean_list.append(pd.DataFrame(np.sqrt(smeans[dim][di,-1,:,ni]**2), columns=colum))
                sdf_final_mean_list[max_indx]['N'] = N
                sdf_final_mean_list[max_indx]['D'] = dim
                max_indx += 1
        


sdf_meanend=pd.concat(sdf_final_mean_list)  

#%%
f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))

ii=0

g1=sns.pointplot(x='N', y='x', hue='D',
              data=sdf_meanend, dodge=.732,  palette='mycoloryellow',#    palette='mycolorpurple',#  palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=0.1,edgecolor='gray',errwidth=3, join=False,zorder=0)
plt.setp(g1.lines, alpha=.5,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='N', y='x', hue='D',
              data=df_meanend, dodge=.732,   palette= sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),# color=col_ro,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True,zorder=10)
plt.setp(g2.lines, alpha=.5,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)
ax.set_ylabel(r'$\|\hat{m}_{\infty} - m_{\infty} \|_2$',fontsize=22)
ax.set_xlabel(r'N',fontsize=22)    
ax.set_yscale('log')

ax.legend().remove()
plt.savefig('OU_ND_vs_N_green'+'mean_end'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_N_green'+'mean_end'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 


 #%%  MEAN Absolute Error of STATIONARY     MEAN 
mean_final_means = np.zeros((len(dims),len(Ns)))
std_final_means = np.zeros((len(dims),len(Ns)))
for ddi,dim in enumerate(dims):
    
    for ni,N in enumerate(Ns):
        
        collec = []
#        for di in range(dim): #RMSE
#            collec.extend((means[dim][di,int(tim[ddi]),:,ni]**2))
#        mean_final_means[ddi,ni] = np.sqrt(np.nanmean(collec))
#        
        for di in range(dim): #MAE
            collec.extend(np.abs(means[dim][di,int(tim[dim]),:,ni]))
        mean_final_means[ddi,ni] = np.nanmean(collec)
        
        std_final_means[ddi,ni] = np.nanstd(collec)


#%% STATIONARY mean VS D
iii = -1        
plt.figure()
for ni,N in enumerate(Ns):
    
    if ni in [0,6,12]:
        
        #plt.figure()
        #plt.plot(dims,mean_final_means[:,ni],lw=4,label='N=%d'%N)
        #plt.fill_between(dims,mean_final_means[:,ni]+std_final_means[:,ni],mean_final_means[:,ni]-std_final_means[:,ni],alpha=0.15)
        plt.errorbar(np.array(dims)+0.2*iii,mean_final_means[:,ni], yerr=std_final_means[:,ni],lw=4,marker='s', label='N=%d'%N,capsize=10,elinewidth=2.5,capthick=2.5)
        iii+=1
        plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('D',fontsize=22)
plt.ylabel(r'$ {KL} (P_{\infty}^{A},P_{\infty}^N)$',fontsize=22)
#plt.figure()
#plt.plot(dims,std_final_means[:,-2])
#
#plt.yscale('log')

    
#%%
hat = ['/', '\\', '/', '\\', '/', '*']
alphs = [0.35,0.21,0.35,0.21,0.25]
plt.figure(figsize=(7,7))
for di,dim in enumerate(dims):
    plt.plot(Ns,mean_final_means[di], lw=4,label='D=%d'%dim)
    plt.fill_between(Ns,mean_final_means[di]+std_final_means[di],mean_final_means[di]-std_final_means[di],alpha=alphs[di])#,hatch=hat[di],edgecolor='gray')

plt.yscale('log')
plt.xlabel('N',fontsize=22)
plt.ylabel(r'$ |\hat{\langle x_{\infty} \rangle_N} - \langle x_{\infty} \rangle|$',fontsize=22)
plt.legend()   
#plt.savefig('OU_ND_vs_N'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)

#%%  MEAN means over time 


mean_overtime_means = np.zeros((len(dims),len(Ns)))
std_overtime_means = np.zeros((len(dims),len(Ns)))
for ddi,dim in enumerate(dims):
    
    for ni,N in enumerate(Ns):
        
        collec = []
        for di in range(dim):
            #collec.extend(np.mean(means[dim][di,:int(tim[ddi]),:,ni],axis=0))
            collec.extend([ np.linalg.norm(means[dim][di,:int(tim[dim]),tr,ni]-m_ts[dim][:int(tim[dim]),di].T) for tr in range(20)])
        mean_overtime_means[ddi,ni] = np.nanmean(collec)
        std_overtime_means[ddi,ni] = np.nanstd(collec)
#%%
#%%overtime mean VS D
iii = -1        
plt.figure()
for ni,N in enumerate(Ns):
    
    if ni in [0,6,12]:
        
        #plt.figure()
        #plt.plot(dims,mean_final_means[:,ni],lw=4,label='N=%d'%N)
        #plt.fill_between(dims,mean_final_means[:,ni]+std_final_means[:,ni],mean_final_means[:,ni]-std_final_means[:,ni],alpha=0.15)
        plt.errorbar(np.array(dims)+0.2*iii,mean_overtime_means[:,ni], yerr=std_overtime_means[:,ni],lw=4,marker='s', label='N=%d'%N,capsize=10,elinewidth=2.5,capthick=2.5)
        iii+=1
        #plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('D',fontsize=22)
plt.ylabel(r'$ {mean} (P_{t}^{A},P_{t}^N)$',fontsize=22)
#plt.figure()
#plt.plot(dims,std_final_means[:,-2])
#
#plt.yscale('log')

    
#%% overtime Mean Absolute Erroe k std means
iii = -2 
hat = ['/', '\\', '/', '\\', '/', '*']
alphs = [0.35,0.21,0.35,0.21,0.25]
plt.figure(figsize=(7,7))
for di,dim in enumerate(dims):
    plt.plot(Ns,mean_overtime_means[di], lw=4,label='D=%d'%dim)
    #plt.fill_between(Ns,mean_overtime_means[di]+std_overtime_means[di],mean_overtime_means[di]-std_overtime_means[di],alpha=alphs[di])#,hatch=hat[di],edgecolor='gray')
    #plt.errorbar(np.array(Ns)+0.1*iii,mean_overtime_means[di], yerr=std_overtime_means[di],lw=4,marker='s', label='D=%d'%dim,capsize=10,elinewidth=2.5,capthick=2.5)
    iii+=1
#plt.yscale('log')
plt.xlabel('N',fontsize=22)
plt.ylabel(r'$ {mean} (P_{t}^{A},P_{t}^N)$',fontsize=22)
plt.legend()   
#plt.savefig('OU_ND_vs_N'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)

#%% overtime std means
iii = -2 
hat = ['/', '\\', '/', '\\', '/', '*']
alphs = [0.35,0.21,0.35,0.21,0.25]
plt.figure(figsize=(7,7))
for di,dim in enumerate(dims):
    plt.plot(Ns,std_overtime_means[di], lw=4,label='D=%d'%dim)
    #plt.fill_between(Ns,mean_overtime_means[di]+std_overtime_means[di],mean_overtime_means[di]-std_overtime_means[di],alpha=alphs[di])#,hatch=hat[di],edgecolor='gray')
    #plt.errorbar(np.array(Ns)+0.1*iii,mean_overtime_means[di], yerr=std_overtime_means[di],lw=4,marker='s', label='D=%d'%dim,capsize=10,elinewidth=2.5,capthick=2.5)
    iii+=1
plt.yscale('log')
plt.xlabel('N',fontsize=22)
plt.ylabel(r'$ {KL} (P_{t}^{A},P_{t}^N)$',fontsize=22)
plt.legend()   
#plt.savefig('OU_ND_vs_N'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)

#%%         
#klls[6][:,17:20,12] =  klls[6][:,7:10,12]    
#klls[5][:,19,9] =  klls[5][:,9,9]         
##plot KL
     
#for dim in dims:
#    plt.figure(dim)
#    for ni,N in enumerate(Ns):
#        plt.subplot(4,4,ni+1)
#        
#        plt.plot(newkl[dim][:,:,ni])
#        plt.title(N)
#    plt.tight_layout()
#
#
##plot final kl
##%%    
#plt.figure()    
#for dim in dims:
#    plt.subplot(2,3,dim-1)
#    for ni,N in enumerate(Ns):
#        plt.plot(N,np.mean(newkl[dim][-1,:,ni],axis=0),'.',label=N)
#        
#plt.legend()
#    


#%%    
#for dim in dims:
#    plt.figure(dim)
#    for ni,N in enumerate(Ns):
#        plt.subplot(4,4,ni+1)
#        plt.plot(means[dim][0,:,:,ni])
#        plt.plot(m_ts[dim][:,0],'k')
#        plt.title(N)
#    plt.tight_layout()
#
#
##%%    
#for dim in dims:
#    plt.figure(dim)
#    for ni,N in enumerate(Ns):
#        plt.subplot(4,4,ni+1)
#        for trial in range(20):
#            plt.plot(means[dim][0,:,trial,ni]-m_ts[dim][:,0])
#        
#        plt.title(N)
#    plt.tight_layout()    

     
#%%    
for dim in dims:
    plt.figure(dim)
    for ni,N in enumerate(Ns):
        plt.subplot(4,4,ni+1)
        for trial in range(20):
            plt.plot(covs[dim][0,1,:,trial,ni],label=trial)
        
        plt.title(N)
    plt.tight_layout()    
#%%
newkl = dict()


for dim in dims:
    print(dim)
    m1 = m_ts[dim] #time,dim
    S1 = C_ts[dim] #time,dim,dim
    newkl[dim] = np.zeros((m1.shape[0],20,Ns.size))#time,trial,N
    newkl[dim].fill(np.nan)
    for ni,N in enumerate(Ns):
        print(N)
        for trial in range(20): 
            if (dim==5 and trial==19 and N==5000) or (dim==6 and trial>=17 and N==6500):
                print('skip')
            else:
                m2 = means[dim][:,:,trial,ni] # dim,time
                S2 = covs[dim][:,:,:,trial,ni] # dim,dim,time,time
                for ti in range(m1.shape[0]):
                    newkl[dim][ti,trial,ni] = KL(m1[ti],m2[:,ti],S2[:,:,ti],S1[ti])
                    
                
 #%%  Frobenius norm of    CoVARIANCE OVERTIME
mean_overtime_covs = np.zeros((len(dims),len(Ns)))
std_overtime_covs = np.zeros((len(dims),len(Ns)))
for ddi,dim in enumerate(dims):
    
    for ni,N in enumerate(Ns):
        
        collec = []
#        for di in range(dim): #RMSE
#            collec.extend((means[dim][di,int(tim[ddi]),:,ni]**2))
#        mean_final_means[ddi,ni] = np.sqrt(np.nanmean(collec))
#        
        
        collec.extend([np.mean( [np.linalg.norm(covs[dim][:,:,ti,tr,ni]-C_ts[dim][ti,:,:].T,ord='fro') for ti in range(int(tim[dim]))] )for tr in range(20)])
        mean_overtime_covs[ddi,ni] = np.nanmean(collec)
        
        std_overtime_covs[ddi,ni] = np.nanstd(collec)
            
            
#%%overtime covs VS D
iii = -1        
plt.figure()
for ni,N in enumerate(Ns):
    
    if ni in [0,6,12]:
        
        #plt.figure()
        #plt.plot(dims,mean_final_means[:,ni],lw=4,label='N=%d'%N)
        #plt.fill_between(dims,mean_final_means[:,ni]+std_final_means[:,ni],mean_final_means[:,ni]-std_final_means[:,ni],alpha=0.15)
        plt.errorbar(np.array(dims)+0.2*iii,mean_overtime_covs[:,ni], yerr=std_overtime_covs[:,ni],lw=4,marker='s', label='N=%d'%N,capsize=10,elinewidth=2.5,capthick=2.5)
        iii+=1
        plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('D',fontsize=22)
plt.ylabel(r'$ {cov} (P_{t}^{A},P_{t}^N)$',fontsize=22)
#plt.figure()
#plt.plot(dims,std_final_means[:,-2])
#
#plt.yscale('log')

    
#%% overtime Mean Absolute Erroe k std covs
iii = -2 
hat = ['/', '\\', '/', '\\', '/', '*']
alphs = [0.35,0.21,0.35,0.21,0.25]
plt.figure(figsize=(7,7))
for di,dim in enumerate(dims):
    plt.plot(Ns,mean_overtime_covs[di], lw=4,label='D=%d'%dim)
    #plt.fill_between(Ns,mean_overtime_means[di]+std_overtime_means[di],mean_overtime_means[di]-std_overtime_means[di],alpha=alphs[di])#,hatch=hat[di],edgecolor='gray')
    #plt.errorbar(np.array(Ns)+0.1*iii,mean_overtime_means[di], yerr=std_overtime_means[di],lw=4,marker='s', label='D=%d'%dim,capsize=10,elinewidth=2.5,capthick=2.5)
    iii+=1
plt.yscale('log')
plt.xlabel('N',fontsize=22)
plt.ylabel(r'$ {cov} (P_{t}^{A},P_{t}^N)$',fontsize=22)
plt.legend()   
#plt.savefig('OU_ND_vs_N'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)

        
 #%%  Frobenius norm of STATIONARY     cov  deterministic
mean_final_covs = np.zeros((len(dims),len(Ns)))
std_final_covs = np.zeros((len(dims),len(Ns)))
for ddi,dim in enumerate(dims):
    
    for ni,N in enumerate(Ns):
        
        collec = []
#        for di in range(dim): #RMSE
#            collec.extend((means[dim][di,int(tim[ddi]),:,ni]**2))
#        mean_final_means[ddi,ni] = np.sqrt(np.nanmean(collec))
#        
        
        collec.extend([np.linalg.norm(covs[dim][:,:,int(tim[dim]),tr,ni]- C_ts[dim][int(tim[dim]),:,:].T,ord='fro') for tr in range(20)])
        mean_final_covs[ddi,ni] = np.nanmean(collec)
        
        std_final_covs[ddi,ni] = np.nanstd(collec)


 #%%  Frobenius norm of STATIONARY     cov  stoch
smean_final_covs = np.zeros((len(dims),len(Ns)))
sstd_final_covs = np.zeros((len(dims),len(Ns)))
for ddi,dim in enumerate(dims):
    
    for ni,N in enumerate(Ns):
        
        collec = []
#        for di in range(dim): #RMSE
#            collec.extend((means[dim][di,int(tim[ddi]),:,ni]**2))
#        mean_final_means[ddi,ni] = np.sqrt(np.nanmean(collec))
#        
        
        collec.extend([np.linalg.norm(scovs[dim][:,:,int(tim[dim]),tr,ni]- C_ts[dim][int(tim[dim]),:,:].T,ord='fro') for tr in range(20)])
        smean_final_covs[ddi,ni] = np.nanmean(collec)
        
        sstd_final_covs[ddi,ni] = np.nanstd(collec)


#%% STATIONARY cov VS D
iii = -1        
plt.figure()
for ni,N in enumerate(Ns):
    
    if ni in [0,6,12]:
        
        #plt.figure()
        #plt.plot(dims,mean_final_means[:,ni],lw=4,label='N=%d'%N)
        #plt.fill_between(dims,mean_final_means[:,ni]+std_final_means[:,ni],mean_final_means[:,ni]-std_final_means[:,ni],alpha=0.15)
        plt.errorbar(np.array(dims)+0.2*iii,mean_final_covs[:,ni], yerr=std_final_covs[:,ni],lw=4,marker='s', label='N=%d'%N,capsize=0,elinewidth=2.5,capthick=2.5)
        plt.errorbar(np.array(dims)+0.2*iii,smean_final_covs[:,ni], yerr=sstd_final_covs[:,ni],lw=4,marker='o', label='N=%d'%N,capsize=0,elinewidth=2.5,capthick=2.5)
        iii+=1
        plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('D',fontsize=22)
plt.ylabel(r'$ {cov end} (P_{\infty}^{A},P_{\infty}^N)$',fontsize=22)
#plt.figure()
#plt.plot(dims,std_final_means[:,-2])
#
#plt.yscale('log')

    
#%% stationary cov vs N
hat = ['/', '\\', '/', '\\', '/', '*']
alphs = [0.35,0.21,0.35,0.21,0.25]
plt.figure(figsize=(7,7))
for di,dim in enumerate(dims):
    plt.plot(Ns,mean_final_covs[di], lw=4,label='D=%d'%dim)
    plt.fill_between(Ns,mean_final_covs[di]+std_final_covs[di],mean_final_covs[di]-std_final_covs[di],alpha=alphs[di])#,hatch=hat[di],edgecolor='gray')

plt.yscale('log')
plt.xlabel('N',fontsize=22)
plt.ylabel(r'$ cov end$',fontsize=22)
plt.legend()   
#plt.savefig('OU_ND_vs_N'+'D'+'.png',  bbox_inches = 'tight', pad_inches = 0.12)






#%% Stationary covs DETERMINISTIC
           
df_final_covs_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            
            df_final_covs_list.append(pd.DataFrame([np.linalg.norm(covs[dim][:,:,int(tim[dim]),tr,ni]-C_ts[dim][int(tim[dim]),:,:],ord='fro') for tr in range(20)], columns=colum))
            df_final_covs_list[max_indx]['N'] = N
            df_final_covs_list[max_indx]['D'] = dim
            max_indx += 1
        


df_covsend=pd.concat(df_final_covs_list)   

#%% Stationary means Stochastic
           
sdf_final_covs_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            
            sdf_final_covs_list.append(pd.DataFrame([np.linalg.norm(scovs[dim][:,:,int(tim[dim]),tr,ni]-C_ts[dim][int(tim[dim]),:,:],ord='fro') for tr in range(20)], columns=colum))
            sdf_final_covs_list[max_indx]['N'] = N
            sdf_final_covs_list[max_indx]['D'] = dim
            max_indx += 1
        


sdf_covsend=pd.concat(sdf_final_covs_list)   


#%% PLOT STATIONARY COVS VS N
f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))
#sns.despine(bottom=True, left=True)
#sns.despine(top=True, right=True)

ii=0
# Show each observation with a scatterplot
#sns.stripplot(x='N', y='x', hue='D', color=col_gr,#palette=sns.color_palette(colors1),
#              data=df_meanend, dodge=True, alpha=.3, zorder=1,ax=ax)

#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])

# Show the conditional means
g1=sns.pointplot(x='N', y='x', hue='D',
              data=sdf_covsend, dodge=.732,   palette='mycoloryellow',#   palette='mycolorpurple',#  palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","H",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=0.1,edgecolor='gray',errwidth=3, join=True,zorder=0)
plt.setp(g2.lines, alpha=1,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='N', y='x', hue='D',
              data=df_covsend, dodge=.732,   palette=sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","H",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True,zorder=0)
plt.setp(g2.lines, alpha=1,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)
ax.set_ylabel(r'$ \|\hat{\Sigma}_{\infty} - \Sigma_{\infty} \|_F$',fontsize=22)
ax.set_xlabel(r'N',fontsize=22)    
ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax[ii].get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')




# Improve the legend 
#labels2 = list(map(lambda x: 'D = %s'%x,labels ))
#
#l1 = ax.legend([handles[-2],handles[3]], ['D','S'], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.33, 0.12))
#
#ax.legend(handles[4:8], labels2[4:8], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.4, 0.8))         
#     
#
#ax.add_artist(l1) 
ax.legend().remove()
plt.savefig('OU_ND_vs_N_green'+'covs_end'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_N_green'+'covs_end'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 



#%%
#only some Ns for stationary covs

small_df_covsend =  df_covsend.loc[df_covsend['N'].isin([500,3500,6500])]
small_sdf_covsend =  sdf_covsend.loc[sdf_covsend['N'].isin([500,3500,6500])]

#%% PLOT STATIONARY COVS VS D SMALL
f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))
#sns.despine(bottom=True, left=True)
#sns.despine(top=True, right=True)

ii=0
# Show each observation with a scatterplot
#sns.stripplot(x='N', y='x', hue='D', color=col_gr,#palette=sns.color_palette(colors1),
#              data=df_meanend, dodge=True, alpha=.3, zorder=1,ax=ax)

#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])

# Show the conditional means
g1=sns.pointplot(x='D', y='x', hue='N',
              data=small_sdf_covsend, dodge=.532,  linestyles='--',   palette='mycoloryellow',#   palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","d",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=0.1,edgecolor='gray',errwidth=3, join=True,zorder=0)
plt.setp(g1.lines, alpha=1,linewidth=4) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='D', y='x', hue='N',
              data=small_df_covsend, dodge=.432,  palette= sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","d",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True,zorder=0)
plt.setp(g2.lines, alpha=1,linewidth=4) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)
ax.set_ylabel(r'$ \|\hat{C}_{\infty} - C_{\infty} \|_F$',fontsize=22)
ax.set_xlabel(r'D',fontsize=22)    
ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax[ii].get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')



#handles, labels = ax.get_legend_handles_labels()
## Improve the legend 
#labels2 = list(map(lambda x: 'N = %s'%x,labels ))
#
#l1 = ax.legend([handles[-1],handles[2]], ['D','S'], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.55, 0.42))
#
#ax.legend(handles[3:6], labels2[3:6], title=None,
#          handletextpad=0, columnspacing=0.1,
#          loc="best", ncol=1, frameon=False,fontsize=16)         
#     
#
#ax.add_artist(l1) 
ax.legend().remove()
plt.savefig('OU_ND_vs_D_green'+'covs_end'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_D_green'+'covs_end'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 

#%%
#%% Overtime Deterministic covs
           
df_overtime_covs_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            
            df_overtime_covs_list.append(pd.DataFrame([np.mean( [np.linalg.norm(covs[dim][:,:,ti,tr,ni]-C_ts[dim][ti,:,:],ord='fro') for ti in range(int(tim[dim]))] )for tr in range(20)], columns=colum))
            df_overtime_covs_list[max_indx]['N'] = N
            df_overtime_covs_list[max_indx]['D'] = dim
            max_indx += 1
        


df_covsmid=pd.concat(df_overtime_covs_list)   


#%% Overtime Stochastic covs
           
sdf_overtime_covs_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            
            sdf_overtime_covs_list.append(pd.DataFrame([np.mean( [np.linalg.norm(scovs[dim][:,:,ti,tr,ni]-C_ts[dim][ti,:,:],ord='fro') for ti in range(int(tim[dim]))] )for tr in range(20)], columns=colum))
            sdf_overtime_covs_list[max_indx]['N'] = N
            sdf_overtime_covs_list[max_indx]['D'] = dim
            max_indx += 1
        


sdf_covsmid=pd.concat(sdf_overtime_covs_list)   

#%%
#only some Ns for stationary covs

small_df_covsmid =  df_covsmid.loc[df_covsmid['N'].isin([500,3500,6500])]
small_sdf_covsmid =  sdf_covsmid.loc[sdf_covsmid['N'].isin([500,3500,6500])]

#%% PLOT STATIONARY COVS VS D SMALL
f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))
#sns.despine(bottom=True, left=True)
#sns.despine(top=True, right=True)

ii=0
# Show each observation with a scatterplot
#sns.stripplot(x='N', y='x', hue='D', color=col_gr,#palette=sns.color_palette(colors1),
#              data=df_meanend, dodge=True, alpha=.3, zorder=1,ax=ax)

#    sns.swarmplot(x='N', y=u'$W_1$', hue='kind',
#                  data=to_plot[ii], dodge=True, alpha=.15,ax=ax[ii])

# Show the conditional means
g1=sns.pointplot(x='D', y='x', hue='N',
              data=small_sdf_covsmid, dodge=.532,  linestyles='--',   palette='mycoloryellow',#   palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","d",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=0.1,edgecolor='gray',errwidth=3, join=True,zorder=0)
#plt.setp(g1.lines, alpha=.6) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='D', y='x', hue='N',
              data=small_df_covsmid, dodge=.432,    palette= sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o","d",'s',"p",'*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=True,zorder=0)
#plt.setp(g1.lines, alpha=.6) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)
ax.set_ylabel(r'$ \|\hat{C}_{t} - C_{t} \|_F$',fontsize=22)
ax.set_xlabel(r'D',fontsize=22)    
ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#ylabels = ax[ii].get_yticklabels()
#ax.set_yticklabels( ylabels,ha='center')



handles, labels = ax.get_legend_handles_labels()
# Improve the legend 
labels2 = list(map(lambda x: 'N:%s'%x,labels ))

l1 = ax.legend([handles[-1],handles[2]], ['D','S'], title=None,
          handletextpad=0, columnspacing=0.1,
          loc="best", ncol=2, frameon=False,fontsize=16,bbox_to_anchor=(0.61, 0.41))

ax.legend(handles[3:6], labels2[3:6], title=None,
          handletextpad=0, columnspacing=0.1,
          loc="best", ncol=1, frameon=False,fontsize=16)         
     

ax.add_artist(l1) 

plt.savefig('OU_ND_vs_D_green'+'covs_overtime'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_D_green'+'covs_overtime'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 




#%% Overtime means DETERMINISTIC
           
df_overtime_mean_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            for di in range(dim):
                df_overtime_mean_list.append(pd.DataFrame(np.mean([np.sqrt(means[dim][di,ti,:,ni]**2) for ti in range(tim[dim])],axis=0), columns=colum))
                df_overtime_mean_list[max_indx]['N'] = N
                df_overtime_mean_list[max_indx]['D'] = dim
                max_indx += 1
        


df_meanmiddle=pd.concat(df_overtime_mean_list)   

#%% overtime means Stochastic
           
sdf_overtime_mean_list = []
max_indx = 0
colum = ['x']
for dim in dims:
    for ni,N in enumerate(Ns):
        
        if ni%2==0:
            #print(max_indx)
            for di in range(dim):
                sdf_overtime_mean_list.append(pd.DataFrame(np.mean([np.sqrt(smeans[dim][di,ti,:,ni]**2) for ti in range(tim[dim])],axis=0), columns=colum))
                sdf_overtime_mean_list[max_indx]['N'] = N
                sdf_overtime_mean_list[max_indx]['D'] = dim
                max_indx += 1
        


sdf_meanmiddle=pd.concat(sdf_overtime_mean_list)  

#%%
f, ax = plt.subplots(1,1, sharey=False,figsize=(5,5))

ii=0

g1=sns.pointplot(x='N', y='x', hue='D',
              data=sdf_meanmiddle, dodge=.732,   palette='mycoloryellow',#    palette=cpal,#palette=sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',linestyles='--',ax=ax,linewidth=0.1,edgecolor='gray',errwidth=3, join=False,zorder=0)
plt.setp(g1.lines, alpha=.5) 
plt.setp(g1.collections,edgecolor=col_gr,linewidth=2,zorder=50)
g2=sns.pointplot(x='N', y='x', hue='D',
              data=df_meanmiddle, dodge=.732,    palette=sns.cubehelix_palette(4, start=2, rot=0, dark=0.15, light=.85, reverse=True),#color ='#464646',
              markers=["o",'d','s','v','*',"D",'d','v',"<",">","^",'P','X'], scale=1.4, ci='sd',ax=ax,linewidth=8,edgecolor='gray',errwidth=3, join=False,zorder=0)
plt.setp(g2.lines, alpha=.5) 
plt.setp(g2.collections,edgecolor=col_gr,linewidth=2,zorder=100)
ax.set_ylabel(r'$\langle \|\hat{m}_{t} - m_{t} \|_2 \rangle_t$',fontsize=22)
ax.set_xlabel(r'N',fontsize=22)    
#ax.set_yscale('log')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-3,-3),useOffset=True)
ax.legend().remove()
plt.savefig('OU_ND_vs_N_green'+'mean_overtime'+'.pdf',  bbox_inches = 'tight', pad_inches = 0.1)     
plt.savefig('OU_ND_vs_N_green'+'mean_overtime'+'.png',  bbox_inches = 'tight', pad_inches = 0.1) 
