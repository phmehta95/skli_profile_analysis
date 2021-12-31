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
import pandas as pd
import itertools
import matplotlib.colors as mcolors
################################################################################################
#                               GETTING PROFILE DATA FROM TEXT FILE
#                                         STEVE D5 data
###############################################################################################
z = []#angle
area = []
error = []


data = np.loadtxt('/user/pmehta/skli_profile_analysis/D5_scan_profile_data.txt')
z = data[:,0] #angle
area = data[:,1]
error = data[:,2]

###################################################################################################
#                                USING CURVE_FIT
#
##################################################################################################

def model_func(x, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, y, aa, bb, cc, dd, ee):

    return c*x**25 +d*x**24 +e*x**23 +f*x**22 + g*x**21 + h*x**20 + i*x**19 +j*x**18 +k*x**17 + l*x**16 + m*x**15 + n*x**14 + o*x**13 + p*x**12 + q*x**11 + r*x**10 +s*x**9 + t*x**8 + u*x**7 +v*x**6 + y*x**5 + aa*x**4 + bb*x**3 + cc*x**2 +dd*x + ee 


popt, _ = curve_fit(model_func,z,area)
c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v,y, aa, bb, cc, dd, ee = popt

y_line = model_func(z, c, d, e, f, g, h, i, j, k, l, m, n ,o, p, q, r, s, t, u, v, y, aa, bb, cc, dd, ee )


fit_equation =  c*z**25 +d*z**24 +e*z**23 +f*z**22 + g*z**21 + h*z**20 + i*z**19 +j*z**18 +k*z**17 + l*z**16 + m*z**15 + n*z**14 + o*z**13 + p*z**12 + q*z**11 + r*z**10 +s*z**9 + t*z**8 + u*z**7 +v*z**6 + y*z**5 + aa*z**4 + bb*z**3 + cc*z**2 +dd*z + ee 


pd_profile = ((y_line - area)/(area))
#pd_profile = ((fit_equation - area))
perc_profile = np.array(pd_profile)


#fig, (ax1,ax2) = plt.subplots(2)
#ax1.errorbar(z, area, yerr=error, label = 'Profile data points')
#ax1.plot(z,y_line, color='red', label = 'Polynomial fit')
#ax2.plot(z, perc_profile, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
#ax2.axhline(y=0.0, color='r', linestyle='-')
#ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

#plt.show()
########################################################################################
#                                         USING POLYFIT
#
########################################################################################


fit = np.poly1d(np.polyfit(z,area,34))

#x_new = np.linspace(-70, 70, 70)

y_new = fit(z)


print (fit)

pd_profile_new = ((y_new - area)/(area))
#pd_profile = ((fit_equation - area))
perc_profile_new = np.array(pd_profile_new)

#print (perc_profile_new)


fig, (ax3,ax4) = plt.subplots(2)

ax3.scatter(z, area,  marker = 'x', label = 'Profile data points')
ax3.plot(z,y_new, color='red', label = 'Polynomial fit')
ax4.plot(z, perc_profile_new, linestyle='--', marker='x', color = 'm', label = '%diff between data points and fit')
ax4.axhline(y=0.0, color='r', linestyle='-')
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax3.legend()
ax4.legend()
#plt.show()


#########################################################################################
#                                GETTING DATA FROM TEXT FILE
#                                         MATT D5/HP1 DATA for AIR/WATER
#                                         POLAR ANGLE
########################################################################################
azi_angle = []
polar_angle = []
amplitude = []
amp_error = []

nline = 91

parser = argparse.ArgumentParser(description='Specify which optic and medium')
parser.add_argument("--list", nargs="+")
parser.add_argument('-D5', '--D5', action='store_true')
parser.add_argument('-HP1', '--HP1', action='store_true')
parser.add_argument('-w', '--water', action='store_true')
parser.add_argument('-a', '--air', action='store_true')
args = parser.parse_args()

if args.D5:
    if args.water:
        df = pd.read_csv("/user/pmehta/skli_profile_analysis/D5_water_matt.csv")
    if args.air:
        df = pd.read_csv("/user/pmehta/skli_profile_analysis/D5_air_matt.csv")
if args.HP1:
    if args.water:
        df = pd.read_csv("/user/pmehta/skli_profile_analysis/HP1_water_matt.csv")
    if args.air:
        df = pd.read_csv("/user/pmehta/skli_profile_analysis/HP1_air_matt.csv")

#print(df)

#azi_0 = df[df["Azimuthal"] == 0]

azi_0_angle = df.loc[df["Azimuthal"] == 0, "Polar"]
azi_0_amplitude = df.loc[df["Azimuthal"] == 0, "Amplitude"]
amp_0_error = df.loc[df["Azimuthal"] == 0, "Error"]

azi_30_angle = df.loc[df["Azimuthal"] == 30, "Polar"]
azi_30_amplitude = df.loc[df["Azimuthal"] == 30, "Amplitude"]
amp_30_error = df.loc[df["Azimuthal"] == 30, "Error"]

azi_60_angle = df.loc[df["Azimuthal"] == 60, "Polar"]
azi_60_amplitude = df.loc[df["Azimuthal"] == 60, "Amplitude"]
amp_60_error = df.loc[df["Azimuthal"] == 60, "Error"]

azi_90_angle = df.loc[df["Azimuthal"] == 90, "Polar"]
azi_90_amplitude = df.loc[df["Azimuthal"] == 90, "Amplitude"]
amp_90_error = df.loc[df["Azimuthal"] == 90, "Error"]

azi_120_angle = df.loc[df["Azimuthal"] == 120, "Polar"]
azi_120_amplitude = df.loc[df["Azimuthal"] == 120, "Amplitude"]
amp_120_error = df.loc[df["Azimuthal"] == 120, "Error"]

azi_150_angle = df.loc[df["Azimuthal"] == 150, "Polar"]
azi_150_amplitude = df.loc[df["Azimuthal"] == 150, "Amplitude"]
amp_150_error = df.loc[df["Azimuthal"] == 150, "Error"]

azi_180_angle = df.loc[df["Azimuthal"] == 180, "Polar"]
azi_180_amplitude = df.loc[df["Azimuthal"] == 180, "Amplitude"]
amp_180_error = df.loc[df["Azimuthal"] == 180, "Error"]

azi_210_angle = df.loc[df["Azimuthal"] == 210, "Polar"]
azi_210_amplitude = df.loc[df["Azimuthal"] == 210, "Amplitude"]
amp_210_error = df.loc[df["Azimuthal"] == 210, "Error"]

azi_240_angle = df.loc[df["Azimuthal"] == 240, "Polar"]
azi_240_amplitude = df.loc[df["Azimuthal"] == 240, "Amplitude"]
amp_240_error = df.loc[df["Azimuthal"] == 240, "Error"]

azi_270_angle = df.loc[df["Azimuthal"] == 270, "Polar"]
azi_270_amplitude = df.loc[df["Azimuthal"] == 270, "Amplitude"]
amp_270_error = df.loc[df["Azimuthal"] == 270, "Error"]

azi_300_angle = df.loc[df["Azimuthal"] == 300, "Polar"]
azi_300_amplitude = df.loc[df["Azimuthal"] == 300, "Amplitude"]
amp_300_error = df.loc[df["Azimuthal"] == 300, "Error"]

azi_330_angle = df.loc[df["Azimuthal"] == 330, "Polar"]
azi_330_amplitude = df.loc[df["Azimuthal"] == 330, "Amplitude"]
amp_330_error = df.loc[df["Azimuthal"] == 330, "Error"]

fig2, ax5 = plt.subplots()
ax5.errorbar(azi_0_angle,azi_0_amplitude,yerr = amp_0_error, marker='x',color = 'b',alpha = 0.5, label = 'Azimuthal = 0')
ax5.errorbar(azi_0_angle,azi_30_amplitude,yerr = amp_30_error,marker='x',color = 'g',alpha = 0.5, label = 'Azimuthal = 30')
ax5.errorbar(azi_0_angle,azi_60_amplitude,yerr = amp_60_error,marker='x',color = 'r',alpha = 0.5, label = 'Azimuthal = 60')
ax5.errorbar(azi_0_angle,azi_90_amplitude,yerr = amp_90_error,marker='x',color = 'r',alpha = 0.5, label = 'Azimuthal = 90')
ax5.errorbar(azi_0_angle,azi_120_amplitude,yerr = amp_120_error,marker='x',color = 'c',alpha = 0.5, label = 'Azimuthal = 120')
ax5.errorbar(azi_0_angle,azi_150_amplitude,yerr = amp_150_error,marker='x',color = 'm',alpha = 0.5, label = 'Azimuthal = 150')
ax5.errorbar(azi_0_angle,azi_180_amplitude,yerr = amp_180_error,marker='x',color = 'y',alpha = 0.5, label = 'Azimuthal = 180')
ax5.errorbar(azi_210_angle,azi_210_amplitude,yerr = amp_210_error,marker='x',color = 'k',alpha = 0.5, label = 'Azimuthal = 210')
ax5.errorbar(azi_0_angle,azi_240_amplitude,yerr = amp_240_error,marker='x',color = 'xkcd:sky blue',alpha = 0.5, label = 'Azimuthal = 240')
ax5.errorbar(azi_0_angle,azi_270_amplitude,yerr = amp_270_error,marker='x',color = 'xkcd:lime',alpha = 0.5, label = 'Azimuthal = 270')
ax5.errorbar(azi_0_angle,azi_300_amplitude,yerr = amp_300_error,marker='x',color = 'xkcd:orange',alpha = 0.5, label = 'Azimuthal = 300')
ax5.errorbar(azi_0_angle,azi_330_amplitude,yerr = amp_330_error,marker='x',color = 'xkcd:pink',alpha = 0.5, label = 'Azimuthal = 330')
ax5.legend()
ax5.set_xlabel("Polar angle (degrees)")
ax5.set_ylabel("Amplitude")
#plt.axis([30, 50, 0, 1])
if args.D5:
    if args.water:
        plt.title("D5 water data")
    if args.air:
        plt.title("D5 air data")
if args.HP1:
    if args.water:
        plt.title("HP1 water data")
    if args.air:
        plt.title("HP1 air data")
plt.show()
