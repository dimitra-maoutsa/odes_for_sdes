# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 04:09:11 2020

@author: Dimi
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

from matplotlib.lines import Line2D
import matplotlib.cm as cm

from matplotlib.colors import ListedColormap
#,[166/256,1153/256,2/256,1],
#newcolors = np.array([[99/256,92/256,7/256],[132/256,122/256,9/256,1],[194/256,179/256,14/256,1],[219/256,202/256,15/256,1],[240/256,221/256,17/256,1]])
newcolors = np.array([[48/256,44/256,3/256],[119/256,110/256,8/256,1],[167/256,154/256,12/256,1],[215/256,165/256,15/256,1]])
#48, 44, 3  ;119, 110, 8);167, 154, 12;215, 198, 15  ;243, 205, 88 ;215, 165, 15
newcmp = ListedColormap(newcolors)
cm.register_cmap("mycolormap", newcmp)
cpal = sns.color_palette("mycolormap", n_colors=4)
#PuBuGn
#RdPu
greenn = ["#006837", "#31a354", "#78c679","#c2e699", "#ffffcc"  ]
          
purble = ['#7a0177', '#c51b8a','#f768a1', '#fbb4b9','#feebe2']
          
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

##999932 = yellow
colsg = linear_gradient(start_hex="#004040", finish_hex="#99eebb", n=5)
newcmp = ListedColormap(colsg['hex'])
cm.register_cmap("mycolorteal", newcmp)
cpalg = sns.color_palette("mycolorteal", n_colors=5)   

sns.palplot(cpalg)


colsp = linear_gradient(start_hex="#51513d", finish_hex="#cccc99", n=5)
                       
newcmp = ListedColormap(colsp['hex'])
cm.register_cmap("mycoloryellow", newcmp)
cpalp = sns.color_palette("mycoloryellow", n_colors=5)   

sns.palplot(cpalp)               