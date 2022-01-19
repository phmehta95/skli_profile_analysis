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

#print(xarray)
print(len(yarray))

fit = h1.GetFunction("PrevFitTMP")

for x in np.arange(-40,42,2):
   yfitval = fit.Eval(x)
   yfitarray.append(yfitval)
   
   #print (x)

print (len(yfitarray))


percentage_diff = []
for a in np.arange(-40,42,2):
   pd = ((yfitarray[a] - yarray[a])/(yarray[a]))
   percentage_diff.append(pd)

print (len(percentage_diff))

fig, (ax1,ax2) = plt.subplots(2)
ax1.scatter(xarray, yarray , marker='x',color = 'r',alpha = 0.5, label = 'Data points')
ax1.plot(xarray,yfitarray, label = 'Polynomial fit')
ax2.plot(xarray, percentage_diff, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax2.axhline(y=0.0, color='r', linestyle='-')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
if args.B1:
   ax1.set_title('Percentage difference between profile data points and fit for B1')
if args.B2:
   ax1.set_title('Percentage difference between profile data points and fit for B2')
if args.B3:
   ax1.set_title('Percentage difference between profile data points and fit for B3')
if args.B4:
   ax1.set_title('Percentage difference between profile data points and fit for B4')
if args.B5:
   ax1.set_title('Percentage difference between profile data points and fit for B5')

ax1.legend()
ax2.legend()
#plt.show()

###########################################################################################################################
############################Percentage difference for for CDF and polynomial fit########################################### 
###########################################################################################################################
#if args.B1:
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
    #for x in np.arange(-40,40,0.5)
for x in np.arange(-40,40,0.5):
   a = y
   b = y1
   c = y2
   d = y3
   e = y4
   f = y5
   g = y6
   h = y7
        #    x = x/1.3
        
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

##############################################################################################################
#                                     FITTING CDF DATA POINTS WITH POLYFIT
#
##############################################################################################################
   #sigma = np.ones(len(angle))
   #sigma[[0, -1]] = 0.01
   
   #print (len(cdf_vec_norm))
   #print (len(angle))

#   fit = np.polyfit(angle,cdf_vec_norm,4)
#   c1 = fit[0]
#   c2 = fit[1]
#   c3 = fit[2]
#   c4 = fit[3]
#   c5 = fit[4]

arr_angle = np.array(angle)
   #fit_equation = c1*arr_angle**4 + c2*arr_angle**3 + c3*arr_angle**2 + c4*arr_angle + c5
#   fit_equation = c1*arr_angle**4 + c2*arr_angle**3 + c3*arr_angle**2 + c4*arr_angle + c5
###############################################################################################################   
#                                     FITTING CDF DATA POINTS USING CURVE_FIT
#     
#
##############################################################################################################
  # sigma = np.ones(40)
  # sigma[[0, -1]] = 0.01
index = [0]

x = arr_angle
y = np.array(cdf_vec_norm)

#print (x)
#print (y)   
x = np.delete(x,index)
y = np.delete(y,index)
   #create the weighting array
y_weight = np.empty(len(y))
   #high pseudo-sd values, meaning less weighting in the fit
y_weight.fill(10)
   #low values for point 0 and the last points, meaning more weighting during the fit procedure 
y_weight[0:2] = y_weight[-5:-1] = 0.1
##y_weight[0:10]  = 0.1
   #print (x,y)
   #print (type(x))
   #print (type(y))

def model_func(x, a, b, c, d, e, f, g, h):    
   #return a*x**3 + b*x**2 +c*x +d   
   #return a*x**4 + b*x**3 +c*x**2 +d*x**1 + e
   #return a*x**6 + b*x**5 +c*x**4 +d*x**3 + e*x**2 + f*x + g
   return a*x**7 + b*x**6 +c*x**5 +d*x**4+ e*x**3 + f*x**2+ g*x**1 + h 
popt, _ = curve_fit(model_func,x,y,sigma = y_weight, absolute_sigma = True)
##popt, _ = curve_fit(model_func,x,y)
a, b, c, d, e, f, g, h = popt
   
   #y_fit = model_func(x, fit_equation[0], fit_equation[1])
   #print (a,b)
#   x_line = np.arange(min(x), max(x), 1)
y_line = model_func(x, a, b, c, d, e, f, g, h )
#fit_equation = a*x**3 + b*x**2 +c*x +d
#fit_equation = a*x**4 + b*x**3 +c*x**2 +d*x**1 + e
fit_equation = a*x**7 + b*x**6 +c*x**5 +d*x**4+ e*x**3 + f*x**2+ g*x**1 + h
#print (x)
#print (y)
#print (fit_equation)
#################################################################################################################
#                                    PERCENTAGE DIFFERENCE CALCULATION
#
#################################################################################################################
arr_angle.ravel()
pd_cdf = ((y - fit_equation)/(fit_equation))
perc_diff_cdf = np.array(pd_cdf)
#print (perc_diff_cdf)
   
fig, (ax3,ax4) = plt.subplots(2)
ax3.scatter(arr_angle, cdf_vec_norm, marker='x',color = 'r',alpha = 0.5, label = 'CDF Data points')
   #ax3.plot(arr_angle,fit_equation, label = 'Polynomial fit')
ax3.plot(x,y_line,'--', color='blue', label = 'Polynomial fit')
   #ax4.plot(arr_angle, perc_diff_cdf, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax4.plot(x, perc_diff_cdf, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax4.axhline(y=0.0, color='r', linestyle='-')
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
if args.B1:
   ax3.set_title('Percentage difference between profile data points and fit for B1')
if args.B2:
   ax3.set_title('Percentage difference between profile data points and fit for B2')
if args.B3:
   ax3.set_title('Percentage difference between profile data points and fit for B3')
if args.B4:
   ax3.set_title('Percentage difference between profile data points and fit for B4')
if args.B5:
   ax3.set_title('Percentage difference between profile data points and fit for B5')

ax3.legend()
ax4.legend()

#plt.show()

###########################################################################################################################
############################Percentage difference for invCDF and polynomial fit########################################### 
###########################################################################################################################

func = lambda j:a*j**7 + b*j**6 +c*j**5 +d*j**4+ e*j**3 + f*j**2+ g*j**1 + h
invfunc = inversefunc(func)
x1 = np.linspace(0,1,160)
output = (invfunc(x1))
   

########################################################
if args.B5:
   def inv_model_func(x1, i,r,k,l,m,n,o,p,q,s,t):    
      return i*x1**10 + r*x1**9 +k*x1**8 + l*x1**7 + m*x1**6 + n*x1**5 + o*x1**4 + p*x1**3 + q*x1**2 +s*x1 + t
########################################################
else:
   def inv_model_func(x1, i,r,k,l,m,n,o,p):    
      return i*x1**7 + r*x1**6 +k*x1**5 + l*x1**4 + m*x1**3 + n*x1**2 + o*x1 + p   


#############################################################################################
#        IF FIT SHOULD GO THROUGH CERTAIN POINTS, UNCOMMENT WEIGHTING CODE BELOW
#
############################################################################################
#create the weighting array
#y_weight = np.empty(len(y_arr))
#high pseudo-sd values, meaning less weighting in the fit
#y_weight.fill(10)
#low values for point 0 and the last points, meaning more weighting during the fit procedure 
#y_weight[0] = y_weight[-5:-1] = 0.1
#y_weight[0:2] = y_weight[-5:-1] = 0.1
   
#popt, _ = curve_fit(inv_model_func,x1,output,sigma = y_weight, absolute_sigma = True)
popt, _ = curve_fit(inv_model_func,x1,output)


#############################################################################################
#########################
if args.B5:
   i,r,k,l,m,n,o,p,q,s,t = popt
#########################   
else:
   i,r,k,l,m,n,o,p = popt


############################################################
if args.B5:
   y_line2 = inv_model_func(x1, i,r,k,l,m,n,o,p,q,s,t )
   inv_fit_equation = i*x1**10 + r*x1**9 +k*x1**8 + l*x1**7 + m*x1**6 + n*x1**5 + o*x1**4 + p*x1**3 + q*x1**2 +s*x1 + t
############################################################
else:
   y_line2 = inv_model_func(x1, i,r,k,l,m,n,o,p )
   inv_fit_equation = i*x1**7 + r*x1**6 +k*x1**5 + l*x1**4 + m*x1**3 + n*x1**2 + o*x1 + p
print (output)
print (inv_fit_equation)


pd_invcdf = ((output - inv_fit_equation)/(inv_fit_equation))
perc_diff_invcdf = np.array(pd_invcdf)
print (perc_diff_invcdf)


fig, (ax5,ax6) = plt.subplots(2)
ax5.scatter(x1, output, marker='x',color = 'r',alpha = 0.5, label = 'Inverse CDF Data points')
   #ax3.plot(arr_angle,fit_equation, label = 'Polynomial fit')
ax5.plot(x1,inv_fit_equation,'--', color='blue', label = 'Polynomial fit')
   #ax4.plot(arr_angle, perc_diff_cdf, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax6.plot(x1, perc_diff_invcdf, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax6.axhline(y=0.0, color='r', linestyle='-')
ax6.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
if args.B1:
   ax5.set_title('Percentage difference between profile data points and fit for B1')
if args.B2:
   ax5.set_title('Percentage difference between profile data points and fit for B2')
if args.B3:
   ax5.set_title('Percentage difference between profile data points and fit for B3')
if args.B4:
   ax5.set_title('Percentage difference between profile data points and fit for B4')
if args.B5:
   ax5.set_title('Percentage difference between profile data points and fit for B5')

ax5.legend()
ax6.legend()

#plt.show()


############################################################################################################
#
#                                    INV TRANSFORM SAMPLING SHAPE CHECK
#
############################################################################################################
icdf = ROOT.TF1("icdf","([0]*x**7 + [1]*x**6 +[2]*x**5 + [3]*x**4 + [4]*x**3 + [5]*x**2 + [6]*x + [7])",0,1)
icdf.SetParameter(0,i)
icdf.SetParameter(1,r)
icdf.SetParameter(2,k)
icdf.SetParameter(3,l)
icdf.SetParameter(4,m)
icdf.SetParameter(5,n)
icdf.SetParameter(6,o)
icdf.SetParameter(7,p)
if args.B5:
   icdf = ROOT.TF1("icdf","[0]*x**10 + [1]*x**9 +[2]*x**8 + [3]*x**7 + [4]*x**6 + [5]*x**5 + [6]*x**4 + [7]*x**3 + [8]*x**2 +[9]*x + [10]",0,1)
   icdf.SetParameter(0,i)
   icdf.SetParameter(1,r)
   icdf.SetParameter(2,k)
   icdf.SetParameter(3,l)
   icdf.SetParameter(4,m)
   icdf.SetParameter(5,n)
   icdf.SetParameter(6,o)
   icdf.SetParameter(7,p)
   icdf.SetParameter(8,q)
   icdf.SetParameter(9,s)
   icdf.SetParameter(10,t)

icdf.SetNpx(1000000)
r = ROOT.TRandom()
nbins = 100
h = ROOT.TH1D("h","",nbins,icdf.Eval(0),icdf.Eval(1))
for i in np.arange(0,10000000,1):
   h.Fill(icdf.Eval(r.Uniform()))
h.Scale(1.0/(10000000/nbins))
c3 = ROOT.TCanvas("c3","c3",1600,900)
c3.cd()
h.Draw()
c3.Update()
h.Fit(icdf,"R")

