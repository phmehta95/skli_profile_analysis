import ROOT
import os
from array import array
from ROOT import TMath
from ROOT import TH1F, TF1, TLine, TCanvas, TGraph
import math
import array as arr
import numpy as np
import matplotlib.pyplot as plt

#Read in collimator file and extract fit polynomial
f1  = ROOT.TFile("skli_warwick_opticlib_analyser_v1.0.root", "READ")
#fit1 = f1.Get('b1/collimator/b1_collimator_theta_air')
fit1 = f1.Get('b1/diffuser/b1_diffuser_theta_air')
#fit1.Print()

#Create array to store the y-values of the fit 
yval_vec = []

#Create array to store cumulative y values of the vit
cdf_vec = []

#Create variable to hold the value of cdf_vec
cdf_vec_hold = 0

#Create array for normalised cdf values
cdf_vec_norm = []

#Create array for x axis values of profiles
ang_vec=[]


if optic = collimator:

#Iterate over fit to find the y values at xvalues at points on the function and store them in vectors
    for x in np.arange(0.00, 3.00, 0.01):
    
        yval = fit1.Eval(x) #value of y at x
        yval_ang_correction = round(abs(yval * math.sin(x)),5) #*sinx for angular correction
        yval_vec.append(yval_ang_correction)#add to yval vector
        yval_vec.append(yval) 
        cdf_vec_hold += yval_ang_correction#add yval to cdf value
        cdf_vec_hold += yval
        cdf_vec.append(cdf_vec_hold)#append cdf_vec array with above value
        ang_water = round(x,2)
        ang_vec.append(ang_water)#create array for x values
    
    
    final_ang_vec = [4.00]
    final_y_vec = [0.00]

    ext_ang_vec = ang_vec + final_ang_vec
    ext_yval_vec= yval_vec + final_y_vec

    print ext_ang_vec
    print ext_yval_vec
#print ext_ang_vec
#print ext_yval_vec

#Min-max normalisation of cdf 
    cdf_vec_min = min(cdf_vec)
    cdf_vec_max = max(cdf_vec)
#print cdf_vec_min
#print cdf_vec_max

    for x in cdf_vec:
        cdf_vec_value_norm  = (x - cdf_vec_min) / (cdf_vec_max - cdf_vec_min)
        cdf_vec_norm.append(cdf_vec_value_norm)

    final_cdf_vec = [1.00]
    ext_cdf_vec= cdf_vec_norm + final_cdf_vec



#Print arrays
#print yval_vec
#print cdf_vec
#print cdf_vec_norm
#print ang_vec

#Plot yval values against distance from light source
plt.figure()
plt.plot(ext_ang_vec, ext_yval_vec)
plt.xlabel("Angle in degrees")
plt.ylabel("Intensity in arbitrary units")
plt.title("B4 collimator PDF")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
#plt.yscale('log')
#plt.show()

#Plot normalised cdf values against distance from light source
plt.figure()
plt.plot(ext_ang_vec, ext_cdf_vec)
plt.xlabel("Angle in degrees")
plt.ylabel("Cumulative intensity normalised to one")
plt.title("B1 collimator CDF")
#plt.yscale('log')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show() 

#Interpolating the points 
#z = np.polyfit(ang_vec, yval_vec, 2)
#new_yval_vec = []
#new_yval_vec = np.polyval(z, ang_vec)
#print new_yval_vec
#plt.figure()
#plt.plot(ang_vec, new_yval_vec)
#plt.yscale('log')
#plt.xlim(xmin=0)
#plt.ylim(ymin=0)
#plt.show()

#plt.close("all")
#Create columns from array
light_prof_dat = np.column_stack([ext_ang_vec,ext_yval_vec])
#print light_prof_dat


#Read values of yval and angle into txt files
np.savetxt('B5coll_prof.txt', light_prof_dat, fmt="%s")

#Read values of yval and angle into csv files
#np.savetxt('B1coll_prof.csv', (ext_ang_vec, ext_yval_vec), delimiter=',')

#matplotlib.pyplot.close("all")
