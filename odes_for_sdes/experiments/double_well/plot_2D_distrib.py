#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 02:58:34 2019

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

from matplotlib.lines import Line2D
import matplotlib.cm as cm
plt.rcParams["axes.labelsize"] = 11
#col_gr = '#3D4A49'
#col_ro = '#208f33'#'#c010d0'
#col_ye =  '#847a09'#	'#d0c010'  #'#dfd71c' #
col_gr2 = '#272e2d'
#col_grn = '#208f33'



def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])        
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)           



plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300

#creating cmaps for the contour plots
purples =  ['#9b59b6' ,'#ac0ebb','#990ca6','#860b91','#73097c','#600868','#4c0653']
#cmap_p = sns.color_palette(purples)
graz =  [ '#3D4A49', '#364241','#303b3a']#,'#2a3333','#242c2b','#1e2524','#181d1d']
#cmap_g = sns.color_palette(graz)
#brown =  ['#9b59b6' ,'#ac0ebb','#990ca6','#860b91','#73097c','#600868','#4c0653']

norm = matplotlib.colors.Normalize(0,1)
colorsp = [[norm(0.0), purples[0]],
          [norm(0.2), purples[1]],
          [norm( 0.4), purples[2]],
          [norm( 0.6), purples[3]],
          [norm(0.8), purples[4]],
          [norm( 1.0), purples[5]] ]

cmap_p = matplotlib.colors.LinearSegmentedColormap.from_list("", colorsp)

colorsg = [[norm(0.0), graz[0]],
          [norm(0.5), graz[1]],
          [norm( 1.0), graz[2]]]#,
#          [norm( 0.6), graz[3]],
#          [norm(0.8), graz[4]],
#          [norm( 1.0), graz[5]] ]

cmap_g = matplotlib.colors.LinearSegmentedColormap.from_list("", colorsg)

colsg = linear_gradient(start_hex="#004040", finish_hex="#99eebb", n=5)
newcmp = ListedColormap(colsg['hex'])

colsp = linear_gradient(start_hex="#3b003b", finish_hex="#ff05ff", n=5)
                       
newcmpp = ListedColormap(colsp['hex'])
cmaps = {'deter': newcmp, 'stoch':newcmpp}


def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.2,shade=True,legend=False,levels=[],ax_lims=False,gridsize=100,cmap=None,scatter_pnts=True):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['zorder'] = 0
            plt.scatter(*args, **kwargs)

        return scatter
    
    def colored_kde_plot(x,y,shade=True,levels=[], gridsize=gridsize,color=None):
        def kde_plot(*args, **kwargs):
            args = (x,y)
            #if c is not None:
                #kwargs['c'] = c
            kwargs['shade'] = shade
            kwargs['gridsize'] = gridsize
            #kwargs['color'] = '#3D4A49'
            #kwargs['colors'] = 'viridis'
            kwargs['n_levels'] =  np.arange(0.05,1,0.1)#[0.1,0.3, 0.5, 0.7, 0.9]#np.arange(0.05,1,0.1)#[0.1,0.3, 0.5, 0.7, 0.9]#7  #Lorenz : 8 , OU: 5 
            if name== 'D': 
                colsg = linear_gradient(start_hex="#004040", finish_hex="#7abe95", n=5)
                newcmpg = ListedColormap(colsg['hex'][::-1])
                kwargs['cmap'] =  newcmpg#sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.95, reverse=False,as_cmap=True)#green ##Lorenz
                #kwargs['cmap'] = sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=.65, reverse=False,as_cmap=True)#green ##2dou
            elif name=='S':
                colsp = linear_gradient(start_hex="#3b003b", finish_hex="#ff89ff", n=5)
                       
                newcmpp = ListedColormap(colsp['hex'][::-1])
                kwargs['cmap'] = newcmpp#sns.cubehelix_palette(8, start=1.5, rot=0, dark=0.2, light=.95, reverse=False,as_cmap=True)#yellow ##Lorenz
                #kwargs['cmap'] = sns.cubehelix_palette(8, start=1.5, rot=0, dark=0.2, light=.65, reverse=False,as_cmap=True) ## 2D OU
            else:
                grey_cmap = ListedColormap((sns.color_palette('Greys',8)[3:]))
                kwargs['cmap'] = grey_cmap#grey
            kwargs['zorder'] = 2
            kwargs["linewidths"] = 3# Lorenz:3 , OU:4
            sns.kdeplot(*args, **kwargs)
        return kde_plot
    col_ye =  '#847a09'  
    col_grn = '#266b5e'      
    colors = {'D': col_grn, 'S':col_ye, 'Si':col_gr2}
    g = sns.JointGrid(x=col_x,y=col_y,data=df,height=5)
    #g.fig.set_size_inches(6,6) ## only for OU --- comment out gia lorenz
    color = 'None'#colors[col_k]#[col_gr,col_ro]#
    legends=[]
    print(col_k)
    for name, df_group in df.groupby(col_k):
        print(name)
        legends.append(name)
        if k_is_color:
            color=k_is_color
            #cmap = cmaps[name]
        if scatter_pnts:
            g.plot_joint(colored_scatter(df_group[col_x],df_group[col_y],color),)
        g.plot_joint(colored_kde_plot(df_group[col_x],df_group[col_y],shade=False, levels=levels,gridsize=gridsize,color=color),) 
        
        if not (ax_lims is None):
            plt.xlim(ax_lims[0][0],ax_lims[0][1])
            plt.ylim(ax_lims[1][0],ax_lims[1][1])
        sns.distplot( df_group[col_x].values,ax=g.ax_marg_x,color=color, )
        sns.distplot(df_group[col_y].values,ax=g.ax_marg_y,color=color,vertical=True )
    # Do also global Hist:
#    sns.distplot(df[col_x].values,ax=g.ax_marg_x,color='grey' )
#    sns.distplot(df[col_y].values.ravel(),ax=g.ax_marg_y,color='grey', vertical=True )
    #g.set_axis_labels(u'$x_1$', u'$x_2$', fontsize=22)
    #g.set_axis_labels(u'x', u'V', fontsize=24)
    g.set_axis_labels(col_x, col_y)
    if legend:
        plt.legend(legends)
    return g
    
    
if __name__ == '__main__':
    
    n=1000
    m1=-3
    m2=3
    
    df1 = pd.DataFrame((np.random.randn(n)+m1).reshape(-1,2), columns=['x','y'])
    df2 = pd.DataFrame((np.random.randn(n)+m2).reshape(-1,2), columns=['x','y'])
    df3 = pd.DataFrame(df1.values+df2.values, columns=['x','y'])
    df1['kind'] = 'dist1'
    df2['kind'] = 'dist2'
    df3['kind'] = 'dist1+dist2'
    df=pd.concat([df1,df2,df3])
    multivariateGrid('x', 'y', 'kind', df=df)