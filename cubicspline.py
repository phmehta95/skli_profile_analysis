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

parser = argparse.ArgumentParser(description='Specify which optic')
parser.add_argument('-1', '--B1', action='store_true')
parser.add_argument('-2', '--B2', action='store_true')
parser.add_argument('-3', '--B3', action='store_true')
parser.add_argument('-4', '--B4', action='store_true')
parser.add_argument('-5', '--B5', action='store_true')

args = parser.parse_args()

h1 = ROOT.TGraph
if args.B1:
   f1 = ROOT.TFile("/user/pmehta/diffuserProfiles/B1_diffuser_(D1)_air_graph_wPoly.root", "READ")
if args.B2:
   f1 = ROOT.TFile("/user/pmehta/diffuserProfiles/B2_diffuser_(D4)_air_graph_wPoly.root", "READ")
if args.B3:
   f1 = ROOT.TFile("/user/pmehta/diffuserProfiles/B3_diffuser_(D3)_air_graph_wPoly.root", "READ")
if args.B4:
   f1 = ROOT.TFile("/user/pmehta/diffuserProfiles/B4_diffuser_(D7)_air_graph_wPoly.root", "READ")
if args.B5:
   f1 = ROOT.TFile("/user/pmehta/diffuserProfiles/B5_diffuser_(D6)_air_graph_wPoly.root", "READ")

c1 = f1.Get("Canvas_1")
c1.ls()
#c1.Print()

if args.B1:
   h1 = c1.GetPrimitive("B1_Diffuser_(D1)_air_1d_theta_dist_graph")
if args.B2:
   h1 = c1.GetPrimitive("B2_Diffuser_(D4)_air_1d_theta_dist_graph")
if args.B3:
   h1 = c1.GetPrimitive("B3_Diffuser_(D3)_air_1d_theta_dist_graph")
if args.B4:
   h1 = c1.GetPrimitive("B4_Diffuser_(D7)_air_1d_theta_dist_graph")
if args.B5:
   h1 = c1.GetPrimitive("B5_Diffuser_(D6)_air_1d_theta_dist_graph")

c2 = ROOT.TCanvas('c2','c2',1200,900)
c2.cd()
#h1.Draw()
#c2.Update()

n = h1.GetN()
#print (n)

xarray = []
yarray = []

#xfitarray = []
yfitarray = []

for i in np.arange(0,n,1):
#   print (i)
   # h1.GetPoint(int(i), ctypes.c_double(x[i]), ctypes.c_double(y[i]))
   x =  h1.GetPointX(int(i))
   y =  h1.GetPointY(int(i))

   
   xarray.append(x)
   yarray.append(y)
   
print (xarray)
print (yarray)

fit = h1.GetFunction("PrevFitTMP")

######################################################################################################
arr = np.arange(np.amin(xarray), np.amax(xarray), 0.5)
s = interpolate.CubicSpline(xarray, yarray)
#s = interpolate.UnivariateSpline(xarray, yarray, k=3)

fig, (ax7) = plt.subplots(1)
plt.scatter(xarray, yarray , marker='x',color = 'r',alpha = 0.5, label = 'Data points')
#ax7.plot(arr, s(arr), 'r-', label='Cubic Spline')
ax7.plot(arr, s(arr), 'r-', label='Univariate Spline')
xarray = np.array(xarray)
yarray = np.array(yarray)

for i in range(xarray.shape[0] - 1):
    segment_x = np.linspace(xarray[i], xarray[i + 1], 100)
    # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
    exp_x = (segment_x - xarray[i])[None, :] ** np.arange(4)[::-1, None]
    # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
    segment_y = s.c[:, i].dot(exp_x)
    ax7.plot(segment_x, segment_y, label='Segment {}'.format(i), ls='--', lw=3)
ax7.legend(prop={'size': 6})
#####################################################################################################


###########################################################################################################################
############################Percentage difference for for CDF and polynomial fit###########################################
###########################################################################################################################
y = fit.GetParameter(0)
y1 = fit.GetParameter(1)
y2 = fit.GetParameter(2)
y3 = fit.GetParameter(3)
y4 = fit.GetParameter(4)
y5 = fit.GetParameter(5)
y6 = fit.GetParameter(6)
y7 = fit.GetParameter(7)

g = ROOT.TF1("ff", "([0]+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x))", -40,40)

cdf_vec_hold = 0
cdf_vec_array = []
cdf_vec_2 = []
cdf_vec_norm = []
angle = []
pdf_vec_array= []

for x in np.arange(-40,40,0.5):
   a = y
   b = y1
   c = y2
   d = y3
   e = y4
   f = y5
   g = y6
   h = y7

   function = (a+(b*x)+(c*x*x)+(d*x*x*x)+(e*x*x*x*x)+(f*x*x*x*x*x)+(g*x*x*x*x*x*x)+(h*x*x*x*x*x*x*x))

   angle.append(x)
   pdf_vec_array.append(function)

   cdf_vec_hold += function
   cdf_vec_2.append(cdf_vec_hold)

   
   cdf_vec_min = min(cdf_vec_2)
   cdf_vec_max = max(cdf_vec_2)
      
for x in cdf_vec_2:
   cdf_vec_value_norm  = (x - cdf_vec_min) / (cdf_vec_max - cdf_vec_min)
   cdf_vec_norm.append(cdf_vec_value_norm)

arr_angle = np.array(angle)
x_array = arr_angle

arr = np.arange(np.amin(x_array), np.amax(x_array), 0.5)
s = interpolate.CubicSpline(x_array, cdf_vec_norm)

fig, (ax8) = plt.subplots(1)
ax8.scatter(x_array, cdf_vec_norm, marker='x',color = 'r',alpha = 0.5, label = 'Data points')
ax8.plot(arr, s(arr), 'r-', label='Cubic Spline')


for i in range(xarray.shape[0] - 1):
    segment_x = np.linspace(xarray[i], xarray[i + 1], 100)
    # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
    exp_x = (segment_x - xarray[i])[None, :] ** np.arange(4)[::-1, None]
    # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
    segment_y = s.c[:, i].dot(exp_x)
    ax8.plot(segment_x, segment_y, label='Segment {}'.format(i), ls='--', lw=3)
ax8.legend()




#plt.legend()
plt.show()


