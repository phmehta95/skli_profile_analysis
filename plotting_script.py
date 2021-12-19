import numpy as np
import ROOT
import ROOT.gROOT
import ROOT.TF1
import ROOT.TCanvas
import sys
import os
import numpy as np
np.__version__
from numpy.random import random
import scipy
scipy.__version__
from scipy import interpolate
import matplotlib.pyplot as plt
from pynverse import inversefunc
import argparse
import math
import ctypes
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
################################################################################################
#                               GETTING PROFILE DATA FROM TEXT FILE
#
###############################################################################################
angle = []
area = []
error = []


data = np.loadtxt('/user/pmehta/skli_profile_analysis/D5_scan_profile_data.txt')
angle = data[:,0]
area = data[:,1]
error = data[:,2]



def model_func(x, a, b, c, d, e, f, g, h, i):
#   return a*x**7 + b*x**6 +c*x**5 +d*x**4 +e*x**3 +f*x**2 + g*x + h 
#    return a*x**8 + b*x**7 +c*x**6 +d*x**5 +e*x**4 +f*x**3 + g*x**2 + h*x + i
    return a*x**8 + b*x**7 +c*x**6 +d*x**5 +e*x**4 +f*x**3 + g*x**2 + h*x + i

#popt, _ = curve_fit(model_func,x,y,sigma = y_weight, absolute_sigma = True)
popt, _ = curve_fit(model_func,angle,area)
a, b, c, d, e, f, g, h, i = popt

   #y_fit = model_func(x, fit_equation[0], fit_equation[1])
   #print (a,b)
#   x_line = np.arange(min(x), max(x), 1)
y_line = model_func(angle, a, b, c, d, e, f, g, h, i )
fit_equation = a*angle**8 + b*angle**7 +c*angle**6 +d*angle**5 +e*angle**4 +f*angle**3 + g*angle**2 + h*angle + i


pd_profile = ((fit_equation - area)/(area))
perc_profile = np.array(pd_profile)







fig, (ax1,ax2) = plt.subplots(2)
ax1.errorbar(angle, area, yerr=error, label = 'Profile data points')
   #ax3.plot(arr_angle,fit_equation, label = 'Polynomial fit')
ax1.plot(angle,y_line, color='red', label = 'Polynomial fit')
ax2.plot(angle, perc_profile, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax2.axhline(y=0.0, color='r', linestyle='-')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.show()


