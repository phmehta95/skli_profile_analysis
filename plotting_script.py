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
z = []
area = []
error = []


data = np.loadtxt('/user/pmehta/skli_profile_analysis/D5_scan_profile_data.txt')
z = data[:,0] #angle
area = data[:,1]
error = data[:,2]



def model_func(x, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, y, aa, bb, cc, dd, ee):
#   return a*x**7 + b*x**6 +c*x**5 +d*x**4 +e*x**3 +f*x**2 + g*x + h 
#    return a*x**8 + b*x**7 +c*x**6 +d*x**5 +e*x**4 +f*x**3 + g*x**2 + h*x + i
#    return a*x**9 + b*x**8 +c*x**7 +d*x**6 +e*x**5 +f*x**4 + g*x**3 + h*x**2 + i*x +j
#    return a*x**8 + b*x**7 +c*x**6 +d*x**5 +e*x**4 +f*x**3 + g*x**2 + h*x + i
#    return a*x**8 + b*x**7 +c*x**6 +d*x**5 +e*x**4 +f*x**3 + g*x**2 + h*x + i

#    return a*x**10 + b*x**9 +c*x**8 +d*x**7 +e*x**6 +f*x**5 + g*x**4 + h*x**3 + i*x**2 +j*x +k
#    return a*x**11 + b*x**10 +c*x**9 +d*x**8 +e*x**7 +f*x**6 + g*x**5 + h*x**4 + i*x**3 +j*x**2 +k*x + l
#    return a*x**12 + b*x**11 +c*x**10 +d*x**9 +e*x**8 +f*x**7 + g*x**6 + h*x**5 + i*x**4 +j*x**3 +k*x**2 + l*x + m
#    return a*x**16 + b*x**15 +c*x**14 +d*x**13 +e*x**12 +f*x**11 + g*x**10 + h*x**9 + i*x**8 +j*x**7 +k*x**6 + l*x**5 + m*x**4 + n*x**3 + o*x**2 + p*x + q
#    return a*x**18 + b*x**17 +c*x**16 +d*x**15 +e*x**14 +f*x**13 + g*x**12 + h*x**11 + i*x**10 +j*x**9 +k*x**8 + l*x**7 + m*x**6 + n*x**5 + o*x**4 + p*x**3 + q*x**2 + r*x +s
#    return a*x**19 + b*x**18 +c*x**17 +d*x**16 +e*x**15 +f*x**14 + g*x**13 + h*x**12 + i*x**11 +j*x**10 +k*x**9 + l*x**8 + m*x**7 + n*x**6 + o*x**5 + p*x**4 + q*x**3 + r*x**2 +s*x +t 

##    return a*x**20 + b*x**19 +c*x**18 +d*x**17 +e*x**16 +f*x**15 + g*x**14 + h*x**13 + i*x**12 +j*x**11 +k*x**10 + l*x**9 + m*x**8 + n*x**7 + o*x**6 + p*x**5 + q*x**4 + r*x**3 +s*x**2 + t*x + u 

##    return a*x**22 + b*x**21 +c*x**20 +d*x**19 +e*x**18 +f*x**17 + g*x**16 + h*x**15 + i*x**14 +j*x**13 +k*x**12 + l*x**11 + m*x**10 + n*x**9 + o*x**8 + p*x**7 + q*x**6 + r*x**5 +s*x**4 + t*x**3 + u*x**2 +v*x + y 

##    return a*x**26 + b*x**25 +c*x**24 +d*x**23 +e*x**22 +f*x**21 + g*x**20 + h*x**19 + i*x**18 +j*x**17 +k*x**16 + l*x**15 + m*x**14 + n*x**13 + o*x**12 + p*x**11 + q*x**10 + r*x**9 +s*x**8 + t*x**7 + u*x**6 +v*x**5 + y*x**4 + aa*x**3 + bb*x**2 + cc*x +dd 

    return c*x**25 +d*x**24 +e*x**23 +f*x**22 + g*x**21 + h*x**20 + i*x**19 +j*x**18 +k*x**17 + l*x**16 + m*x**15 + n*x**14 + o*x**13 + p*x**12 + q*x**11 + r*x**10 +s*x**9 + t*x**8 + u*x**7 +v*x**6 + y*x**5 + aa*x**4 + bb*x**3 + cc*x**2 +dd*x + ee 


#popt, _ = curve_fit(model_func,x,y,sigma = y_weight, absolute_sigma = True)
popt, _ = curve_fit(model_func,z,area)
c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v,y, aa, bb, cc, dd, ee = popt

   #y_fit = model_func(x, fit_equation[0], fit_equation[1])
   #print (a,b)
#   x_line = np.arange(min(x), max(x), 1)
y_line = model_func(z, c, d, e, f, g, h, i, j, k, l, m, n ,o, p, q, r, s, t, u, v, y, aa, bb, cc, dd, ee )
#fit_equation = a*angle**8 + b*angle**7 +c*angle**6 +d*angle**5 +e*angle**4 +f*angle**3 + g*angle**2 + h*angle + i
#fit_equation = a*z**9 + b*z**8 +c*z**7 +d*z**6 +e*z**5 +f*z**4 + g*z**3 + h*z**2 + i*z + j
#fit_equation = a*z**10 + b*z**9 +c*z**8 +d*z**7 +e*z**6 +f*z**5 + g*z**4 + h*z**3 + i*z**2 +j*z +k
#fit_equation = a*z**11 + b*z**10 +c*z**9 +d*z**8 +e*z**7 +f*z**6 + g*z**5 + h*z**4 + i*z**3 +j*z**2 +k*z + l
#fit_equation = a*z**12 + b*z**11 +c*z**10 +d*z**9 +e*z**8 +f*z**7 + g*z**6 + h*z**5 + i*z**4 +j*z**3 +k*z**2 + l*z + m 
#fit_equation = a*z**14 + b*z**13 +c*z**12 +d*z**11 +e*z**10 +f*z**9 + g*z**8 + h*z**7 + i*z**6 +j*z**5 +k*z**4 + l*z**3 + m*z**2 + n*z + o
#fit_equation =  a*z**16 + b*z**15 +c*z**14 +d*z**13 +e*z**12 +f*z**11 + g*z**10 + h*z**9 + i*z**8 +j*z**7 +k*z**6 + l*z**5 + m*z**4 + n*z**3 + o*z**2 + p*z + q

#fit_equation = a*z**18 + b*z**17 +c*z**16 +d*z**15 +e*z**14 +f*z**13 + g*z**12 + h*z**11 + i*z**10 +j*z**9 +k*z**8 + l*z**7 + m*z**6 + n*z**5 + o*z**4 + p*z**3 + q*z**2 + r*z +s 

#fit_equation = a*z**19 + b*z**18 +c*z**17 +d*z**16 +e*z**15 +f*z**14 + g*z**13 + h*z**12 + i*z**11 +j*z**10 +k*z**9 + l*z**8 + m*z**7 + n*z**6 + o*z**5 + p*z**4 + q*z**3 + r*z**2 +s*z + t 


##fit_equation = a*z**20 + b*z**19 +c*z**18 +d*z**17 +e*z**16 +f*z**15 + g*z**14 + h*z**13 + i*z**12 +j*z**11 +k*z**10 + l*z**9 + m*z**8 + n*z**7 + o*z**6 + p*z**5 + q*z**4 + r*z**3 +s*z**2 + t*z + u 

##fit_equation = a*z**22 + b*z**21 +c*z**20 +d*z**19 +e*z**18 +f*z**17 + g*z**16 + h*z**15 + i*z**14 +j*z**13 +k*z**12 + l*z**11 + m*z**10 + n*z**9 + o*z**8 + p*z**7 + q*z**6 + r*z**5 +s*z**4 + t*z**3 + u*z**2 +v*z + y 

##fit_equation =  a*z**26 + b*z**25 +c*z**24 +d*z**23 +e*z**22 +f*z**21 + g*z**20 + h*z**19 + i*z**18 +j*z**17 +k*z**16 + l*z**15 + m*z**14 + n*z**13 + o*z**12 + p*z**11 + q*z**10 + r*z**9 +s*z**8 + t*z**7 + u*z**6 +v*z**5 + y*z**4 + aa*z**3 + bb*z**2 + cc*z +dd 


fit_equation =  c*z**25 +d*z**24 +e*z**23 +f*z**22 + g*z**21 + h*z**20 + i*z**19 +j*z**18 +k*z**17 + l*z**16 + m*z**15 + n*z**14 + o*z**13 + p*z**12 + q*z**11 + r*z**10 +s*z**9 + t*z**8 + u*z**7 +v*z**6 + y*z**5 + aa*z**4 + bb*z**3 + cc*z**2 +dd*z + ee 


pd_profile = ((y_line - area)/(area))
#pd_profile = ((fit_equation - area))
perc_profile = np.array(pd_profile)

print (fit_equation)
print (area)
print (perc_profile)





fig, (ax1,ax2) = plt.subplots(2)
ax1.errorbar(z, area, yerr=error, label = 'Profile data points')
   #ax3.plot(arr_angle,fit_equation, label = 'Polynomial fit')
ax1.plot(z,y_line, color='red', label = 'Polynomial fit')
ax2.plot(z, perc_profile, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax2.axhline(y=0.0, color='r', linestyle='-')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.show()


