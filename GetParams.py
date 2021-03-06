import numpy as np
import ROOT
import ROOT.TF1
import sys
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
import seaborn as sns
import cmd


parser = argparse.ArgumentParser(description='Specify which optic')
parser.add_argument("--list", nargs="+")
parser.add_argument('-c', '--collimator', action='store_true')
parser.add_argument('-d', '--diffuser', action='store_true')
parser.add_argument('-a', '--azimuthal', action='store_true')
parser.add_argument('-1', '--B1', action='store_true')
parser.add_argument('-2', '--B2', action='store_true')
parser.add_argument('-3', '--B3', action='store_true')
parser.add_argument('-4', '--B4', action='store_true')
parser.add_argument('-5', '--B5', action='store_true')
parser.add_argument('-zero', '--zerodeg', action='store_true')
parser.add_argument('-90', '--deg90', action='store_true')
parser.add_argument('-180', '--deg180', action='store_true')
parser.add_argument('-270', '--deg270', action='store_true')
parser.add_argument('-360', '--deg360', action='store_true')

args = parser.parse_args()

if args.collimator:
    f1 = ROOT.TFile("skli_warwick_opticlib_analyser_v1.0.root", "READ")
    if args.B1:
        f  = f1.Get('b1/collimator/b1_collimator_theta_air')
    elif args.B2:
        f  = f1.Get('b2/collimator/b2_collimator_theta_air')
    elif args.B3:
        f  = f1.Get('b3/collimator/b3_collimator_theta_air')
    elif args.B4:
        f  = f1.Get('b4/collimator/b4_collimator_theta_air')
    elif args.B5:
        f  = f1.Get('b5/collimator/b5_collimator_theta_air')
    
    y = f.GetParameter(0)
    y1 = f.GetParameter(1)
    y2 = f.GetParameter(2)
    y3 = f.GetParameter(3)
    y4 = f.GetParameter(4)
    y5 = f.GetParameter(5)
    y6 = f.GetParameter(6)
    y7 = f.GetParameter(7)
    y8 = f.GetParameter(8)
    #y9 = f.GetParameter(9)
    
    print (y1)
    #g = ROOT.TF1("ff", "([0]+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x))", -40,40)
    #g = ROOT.TF1("ff", "([0]+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x))")
    z = ROOT.TF1("ff", "([0])+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x)+([8]*x*x*x*x*x*x*x*x)", 0, 4)

    #g.SetParameters(y,y1,y2,y3,y4,y5,y6,y7)
    z.SetParameters(y,y1,y2,y3,y4,y5,y6,y7,y8)

    cdf_vec_hold = 0
    pdf_vec_array = []
    cdf_vec_array = []
    cdf_vec_norm = []
    angle = []
    #for x in np.arange(-40,40,0.5)
    for x in np.arange(0,3.5,0.01):
        a = y
        b = y1
        c = y2
        d = y3
        e = y4
        f = y5
        g = y6
        h = y7
        i = y8
        #    x = x/1.3
        function = (a+(b*x)+(c*x*x)+(d*x*x*x)+(e*x*x*x*x)+(f*x*x*x*x*x)+(g*x*x*x*x*x*x)+(h*x*x*x*x*x*x*x)+(i*x*x*x*x*x*x*x*x))
        function = function*abs(np.sin(x))
        angle.append(x)
        pdf_vec_array.append(function)
        cdf_vec_hold += function
        cdf_vec_array.append(cdf_vec_hold)

#    print function
#    print cdf_vec_array
#    plt.plot(angle,cdf_vec_array)
#    plt.show()
    
    final_ang_vec = [4.00]
    final_y_vec = [0.00]
    
    ext_ang_vec = angle + final_ang_vec
    ext_yval_vec= pdf_vec_array + final_y_vec
    
    
   # for x in cdf_vec_array:

    #    cdf_vec_hold += function
     #   cdf_vec_2.append(cdf_vec_hold)

    #print cdf_vec_2    
    cdf_vec_min = min(cdf_vec_array)
    cdf_vec_max = max(cdf_vec_array)

    for x in cdf_vec_array:
        
        cdf_vec_value_norm  = (x - cdf_vec_min) / (cdf_vec_max - cdf_vec_min)
        cdf_vec_norm.append(cdf_vec_value_norm)
        
    final_cdf_vec = [1.00]
    ext_cdf_vec= cdf_vec_norm + final_cdf_vec
    
#    print cdf_vec_norm
#    print ext_cdf_vec

        
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(1, 1, 1)
    ax1.plot(ext_ang_vec, ext_yval_vec)
    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Angle')
    ax1.set_title('PDF of intensity')
    #plot2 = plt.figure(2)

    fig4 = plt.figure()
    ax2 = fig4.add_subplot(1,1,1)
    ax2.plot(ext_ang_vec, ext_cdf_vec)
    ax2.set_ylabel('Intensity')
    ax2.set_xlabel('Angle')
    ax2.set_title('CDF of intensity')
   
    
##################################################################################################################################
#SPECIAL CASES: B2 and B5 collimator - need extra coefficients to fit inverse of cdf

    if args.B2 or args.B5:
        
 
        fit = np.polyfit(ext_ang_vec,ext_cdf_vec,5)
        c1 = fit[0]
        c2 = fit[1]
        c3 = fit[2]
        c4 = fit[3]
        c5 = fit[4]
        c6 = fit[5]
        #c7 = fit[6]
        #c8 = fit[7]
        #c9 = fit[8]
    
        #print (c1,c2,c3,c4,c5)

        arr_angle = np.array(ext_ang_vec)
    
        #fit_equation = c1*arr_angle**8 + c2*arr_angle**7 + c3*arr_angle**6 + c4*arr_angle**5 + c5*arr_angle**4 + c6*arr_angle**3 + c7*arr_angle**2 + c8*arr_angle + c9
        #fit_equation = c1*arr_angle**6 + c2*arr_angle**5 + c3*arr_angle**4 + c4*arr_angle**3 + c5*arr_angle**2 + c6*arr_angle**1 + c7
        fit_equation = c1*arr_angle**5 + c2*arr_angle**4 + c3*arr_angle**3 + c4*arr_angle**2 + c5*arr_angle**1 + c6
        #fit_equation = c1*arr_angle**4 + c2*arr_angle**3 + c3*arr_angle**2 + c4*arr_angle**1 + c5

        
        #Get inverse of equation

        #func = lambda j:c1*j**4 + c2*j**3 + c3*j**2 + c4*j**1 + c5
        func = lambda j:c1*j**5 + c2*j**4 + c3*j**3 + c4*j**2 + c5*j**1 + c6
        #func = lambda j:c1*j**6 + c2*j**5 + c3*j**4 + c4*j**3 + c5*j**2 + c6*j**1 + c7

        invfunc = inversefunc(func)
        x1 = np.linspace(0,4,160)

        output = (invfunc(x1))


        fit1 = np.polyfit(x1,output,16)
        d1 = fit1[0]
        d2 = fit1[1]
        d3 = fit1[2]
        d4 = fit1[3]
        d5 = fit1[4]
        d6 = fit1[5]
        d7 = fit1[6]
        d8 = fit1[7]
        d9 = fit1[8]
        d10 = fit1[9]
        d11 = fit1[10]
        d12 = fit1[11]
        d13 = fit1[12]
        d14 = fit1[13]
        d15 = fit1[14]
        d16 = fit1[15]
        d17 = fit1[16]
        
        #fit_equation1 = d1*x1**3 + d2*x1**2 + d3*x1**1 + d4
        #fit_equation1 = d1*x1**4 + d2*x1**3 + d3*x1**2 + d4*x1**1 + d5
        #fit_equation1 = d1*x1**5 + d2*x1**4 + d3*x1**3 + d4*x1**2 + d5*x1**1 + d6
        #fit_equation1 = d1*x1**6 + d2*x1**5 + d3*x1**4 + d4*x1**3 + d5*x1**2 + d6*x1**1 + d7
        #fit_equation1 = d1*x1**7 + d2*x1**6 + d3*x1**5 + d4*x1**4 + d5*x1**3 + d6*x1**2 + d7*x1**1 + d8 
        #fit_equation1 = d1*x1**8 + d2*x1**7 + d3*x1**6 + d4*x1**5 + d5*x1**4 + d6*x1**3 + d7*x1**2 + d8*x1**1 + d9
        #fit_equation1 = d1*x1**9 + d2*x1**8 + d3*x1**7 + d4*x1**6 + d5*x1**5 + d6*x1**4 + d7*x1**3 + d8*x1**2 + d9*x1**1 +d10
        #fit_equation1 = d1*x1**10 + d2*x1**9 + d3*x1**8 + d4*x1**7 + d5*x1**6 + d6*x1**5 + d7*x1**4 + d8*x1**3 + d9*x1**2 +d10*x1**1 + d11
        #fit_equation1 = d1*x1**11 + d2*x1**10 + d3*x1**9 + d4*x1**8 + d5*x1**7 + d6*x1**6 + d7*x1**5 + d8*x1**4 + d9*x1**3 +d10*x1**2 + d11*x1**1 + d12
        #fit_equation1 = d1*x1**15 + d2*x1**14 + d3*x1**13 + d4*x1**12 + d5*x1**11 + d6*x1**10 + d7*x1**9 + d8*x1**8 + d9*x1**7 +d10*x1**6 + d11*x1**5 + d12*x1**4 + d13*x1**3 +d14*x1**2 + d15*x1**1 + d16
        fit_equation1 = d1*x1**16 + d2*x1**15 + d3*x1**14 + d4*x1**13 + d5*x1**12 + d6*x1**11 + d7*x1**10 + d8*x1**9 + d9*x1**8 +d10*x1**7 + d11*x1**6 + d12*x1**5 + d13*x1**4 +d14*x1**3 + d15*x1**2 + d16*x1**1 + d17 
        #Plotting

        fig, (ax1,ax2) = plt.subplots(2)
        ax1.plot(ext_ang_vec, fit_equation,color = 'r',alpha = 0.5, label = 'Polynomial fit')
        ax1.scatter(angle, cdf_vec_norm, s = 5, color = 'b', label = 'Data points')
        ax2.plot(x1, invfunc(x1),color = 'g', label = 'Inverse cdf func')
        ax2.plot(x1, fit_equation1, color = 'm', label = 'Inverse cdf fit')
        ax2.set_xlim([0,1])
        ax1.set_title('Polynomial fit to cdf and inverse cdf')
        ax1.legend()
        ax2.legend()
        plt.show()



##################################################################################################################################
    fit = np.polyfit(ext_ang_vec,ext_cdf_vec,4)
    c1 = fit[0]
    c2 = fit[1]
    c3 = fit[2]
    c4 = fit[3]
    c5 = fit[4]
    #c6 = fit[5]
    #c7 = fit[6]
    #c8 = fit[7]
   # c9 = fit[8]
    
   
    
    print (c1,c2,c3,c4,c5)

    arr_angle = np.array(ext_ang_vec)
    #print arr_angle
    #print(type(arr_angle))

    #arr_angle2 = arr_angle.ravel()
    #print arr_angle2
    #print(type(arr_angle2)) 


    fit_equation = c1*arr_angle**4 + c2*arr_angle**3 + c3*arr_angle**2 + c4*arr_angle + c5
   # fit_equation = c1*arr_angle**8 + c2*arr_angle**7 + c3*arr_angle**6 + c4*arr_angle**5 + c5*arr_angle**4 + c6*arr_angle**3 + c7*arr_angle**2 + c8*arr_angle + c9
   
    

    #Get inverse of equation
    func = lambda j: c1*j**4 + c2*j**3 + c3*j**2 + c4*j + c5
    #func = lambda j:c1*j**8 + c2*j**7 + c3*j**6 + c4*j**5 + c5*j**4 + c6*j**3 + c7*j**2 + c8*j + c9
    invfunc = inversefunc(func)
    x1 = np.linspace(0,4,160)

    output = (invfunc(x1))


    fit1 = np.polyfit(x1,output,7)
    d1 = fit1[0]
    d2 = fit1[1]
    d3 = fit1[2]
    d4 = fit1[3]
    d5 = fit1[4]
    d6 = fit1[5]
    d7 = fit1[6]
    d8 = fit1[7]
    print (d1,d2,d3,d4,d5,d6,d7,d8)

#    fit_equation1 = d1*x1**6 + d2*x1**5 + d3*x1**4 + d4*x1**3 + d5*x1**2 + d6*x1 + d7
#   fit_equation1 = d1*x1**7 + d2*x1**6 + d3*x1**5 + d4*x1**4 + d5*x1**3 + d6*x1**2 + d7*x1 + d8
    fit_equation1 = d1*x1**7 + d2*x1**6 + d3*x1**5 + d4*x1**4 + d5*x1**3 + d6*x1**2 + d7*x1 + d8

    
    #Plotting

    #fig1 = plt.figure
    #ax1 = fig1.subplots()
    #ax2 = fig1.subplots()

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(ext_ang_vec, fit_equation,color = 'r',alpha = 0.5, label = 'Polynomial fit')
    ax1.scatter(angle, cdf_vec_norm, s = 5, color = 'b', label = 'Data points')
    ax2.plot(x1, invfunc(x1),color = 'g', label = 'Inverse cdf func')
    ax2.plot(x1, fit_equation1, color = 'm', label = 'Inverse cdf fit')
    ax2.set_xlim([0,1])
  #  ax2.set_ylim([0,1])
    ax1.set_title('Polynomial fit to cdf and inverse cdf')
    ax1.legend()
    ax2.legend()
    plt.show()
#########################################################################################################################################
#                                                DIFFUSER CDF AND INV CDF CALCULATION                                                   # 
#                                                                                                                                       #  
#########################################################################################################################################
if args.diffuser:
    f1 = ROOT.TFile("skli_warwick_opticlib_analyser_v1.0.root", "READ")
    if args.B1:
        f  = f1.Get('b1/diffuser/b1_diffuser_theta_air')
    elif args.B2:
        f  = f1.Get('b2/diffuser/b2_diffuser_theta_air')
    elif args.B3:
        f  = f1.Get('b3/diffuser/b3_diffuser_theta_air')
    elif args.B4:
        f  = f1.Get('b4/diffuser/b4_diffuser_theta_air')
    elif args.B5:
        f  = f1.Get('b5/diffuser/b5_diffuser_theta_air')
    
    y = f.GetParameter(0)
    y1 = f.GetParameter(1)
    y2 = f.GetParameter(2)
    y3 = f.GetParameter(3)
    y4 = f.GetParameter(4)
    y5 = f.GetParameter(5)
    y6 = f.GetParameter(6)
    y7 = f.GetParameter(7)
    #y8 = f.GetParameter(8)
    #y9 = f.GetParameter(9)
#    print y
#    print y1
#    print y2
#    print y3
#    print y4
#    print y5
#    print y6
#    print y7
    #print y8
    #print y9
    g = ROOT.TF1("ff", "([0]+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x))", -40,40)
    #g = ROOT.TF1("ff", "([0]+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x))")
#    z = ROOT.TF1("ff", "([0])+([1]*x) + ([2]*x*x) +([3]*x*x*x)+([4]*x*x*x*x)+([5]*x*x*x*x*x)+([6]*x*x*x*x*x*x)+([7]*x*x*x*x*x*x*x)+([8]*x*x*x*x*x*x*x*x)", 0, 4)

    #g.SetParameters(y,y1,y2,y3,y4,y5,y6,y7)
#    z.SetParameters(y,y1,y2,y3,y4,y5,y6,y7,y8)
    #z.Draw()

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
        #    function = function*abs(np.sin(x))
        angle.append(x)
        pdf_vec_array.append(function)
        #    print function
    #print cdf_vec_array
    #plt.plot(angle,cdf_vec_array)
    #plt.show()


        cdf_vec_hold += function
        cdf_vec_2.append(cdf_vec_hold)

    #print cdf_vec_2    
    cdf_vec_min = min(cdf_vec_2)
    cdf_vec_max = max(cdf_vec_2)

    for x in cdf_vec_2:
        cdf_vec_value_norm  = (x - cdf_vec_min) / (cdf_vec_max - cdf_vec_min)
        cdf_vec_norm.append(cdf_vec_value_norm)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(angle,pdf_vec_array)
    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Angle')
    ax1.set_title('PDF of intensity')
    #plot2 = plt.figure(2)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(angle,cdf_vec_norm)
    ax2.set_ylabel('Intensity')
    ax2.set_xlabel('Angle')
    ax2.set_title('CDF of intensity')



#    plt.plot(angle,cdf_vec_array)
#    plt.plot(angle,cdf_vec_norm)
#    plt.xlabel("Angle in degrees")
#    plt.ylabel("Intensity in arbitrary units")
#    plt.show()



    fit = np.polyfit(angle,cdf_vec_norm,4)
    c1 = fit[0]
    c2 = fit[1]
    c3 = fit[2]
    c4 = fit[3]
    c5 = fit[4]
#    print c1,c2,c3,c4,c5

    arr_angle = np.array(angle)
    #print arr_angle
    #print(type(arr_angle))

    arr_angle2 = arr_angle.ravel()
    #print arr_angle2
    #print(type(arr_angle2)) 


    fit_equation = c1*arr_angle**4 + c2*arr_angle2**3 + c3*arr_angle2**2 + c4*arr_angle2 + c5
    
       
 
    #Get inverse of equation
    func = lambda j: c1*j**4 + c2*j**3 + c3*j**2 + c4*j + c5
    invfunc = inversefunc(func)
    x1 = np.linspace(0,1,160)

    output = (invfunc(x1))


    fit1 = np.polyfit(x1,output,3)
    d1 = fit1[0]
    d2 = fit1[1]
    d3 = fit1[2]
    d4 = fit1[3]

#    print d1,d2,d3,d4

    fit_equation1 = d1*x1**3 + d2*x1**2 + d3*x1 + d4 


    #Plotting

    #fig1 = plt.figure
    #ax1 = fig1.subplots()
    #ax2 = fig1.subplots()

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(angle, fit_equation,color = 'r',alpha = 0.5, label = 'Polynomial fit')
    ax1.scatter(angle, cdf_vec_norm, s = 5, color = 'b', label = 'Data points')
    ax2.plot(x1, invfunc(x1),color = 'g', label = 'Inverse cdf func')
    ax2.plot(x1, fit_equation1, color = 'm', label = 'Inverse cdf fit')
    ax2.set_xlim([0,1])
    ax1.set_title('Polynomial fit to cdf and inverse cdf')
    ax1.legend()
    plt.show()



if args.azimuthal:
######################################################################################################################
#                                                                                                                    #
#                                       AZIMUTHAL ANGLE INV CDF CALCULATION                                          #
#                                                                                                                    #
#                                                                                                                    #
######################################################################################################################
  #  if args.zerodeg:
    #########Produce azimuthal PDF#######################
 #   z = np.linspace(0, 2*math.pi, 160)

#    if args.0deg:
#        a = 1+0.01*np.sin(z+0)
#    elif args.90deg:
#        a = 1+0.01*np.sin(z+(math.pi/2))
#    elif args.180deg:
#        a = 1+0.01*np.sin(z+math.pi)
#    elif args.270deg:
#        a = 1+0.01*np.sin(z+(3*math.pi/2))
#    elif args.360deg:
#        a = 1+0.01*np.sin(z+(2*math.pi))

    #plt.plot(z,a, 'r')
    #plt.show()
    
    ########Make CDF from azimuthal PDF################ 
    
    azimuthal_angle = []
    azimuthal_pdf = []
    azimuthal_cdf_vec_hold = 0
    azimuthal_cdf_vec = []
    azimuthal_cdf_vec_norm = []
    
    
    azi_value = input("Enter custom azimuthal angle value in radians:")
    for z in np.arange(0, 2*math.pi, 0.01):
#        if args.zerodeg:
#            a = 1+0.01*np.sin(z+0)
#        elif args.deg90:
#            a = 1+0.01*np.sin(z+(math.pi/2))
#        elif args.deg180:
#            a = 1+0.01*np.sin(z+math.pi)
#        elif args.deg270:
#            a = 1+0.01*np.sin(z+(3*math.pi/2))
#        elif args.deg360:
#            a = 1+0.01*np.sin(z+(2*math.pi))
#        elif args.custom:
#            a = 1+0.01*np.sin(z+(azi_value))

        a = 1+0.01*np.sin(z+(float(azi_value)))

        azimuthal_pdf.append(a)
        azimuthal_angle.append(z)
        azimuthal_cdf_vec_hold += a
        azimuthal_cdf_vec.append(azimuthal_cdf_vec_hold) 
        
    #print (azimuthal_cdf_vec)
    
    #plt.plot(azimuthal_angle, azimuthal_pdf)

    azimuthal_cdf_vec_min = min(azimuthal_cdf_vec)
    azimuthal_cdf_vec_max = max(azimuthal_cdf_vec)
    
    #print (azimuthal_cdf_vec_min, azimuthal_cdf_vec_max) 
    
    for z in azimuthal_cdf_vec:
        azi_cdf_vec_value_norm  = (z - azimuthal_cdf_vec_min) / (azimuthal_cdf_vec_max - azimuthal_cdf_vec_min)
        azimuthal_cdf_vec_norm.append(azi_cdf_vec_value_norm)
    
    #print(azimuthal_cdf_vec_norm)
    plt.plot(azimuthal_angle, azimuthal_cdf_vec_norm)
    #plt.show()


    fig5 = plt.figure()
    ax1 = fig5.add_subplot(1, 1, 1)
    ax1.plot(azimuthal_angle, azimuthal_pdf)
    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Angle')
    ax1.set_title('PDF of intensity')
    #plot2 = plt.figure(2)

    fig6 = plt.figure()
    ax2 = fig6.add_subplot(1,1,1)
    ax2.plot(azimuthal_angle, azimuthal_cdf_vec_norm)
    ax2.set_ylabel('Intensity')
    ax2.set_xlabel('Angle')
    ax2.set_title('CDF of intensity')


    fit = np.polyfit(azimuthal_angle,azimuthal_cdf_vec_norm,2)
    a1 = fit[0]
    a2 = fit[1]
    a3 = fit[2]
   # a4 = fit[3]
   # a5 = fit[4]
   # print (a1,a2,a3,a4)

    arr_az_angle = np.array(azimuthal_angle)
    #print arr_angle
    #print(type(arr_angle))


    #fit_equation = a1*arr_az_angle**3 + a2*arr_az_angle**2 + a3*arr_az_angle**1 + a4
    fit_equation = a1*arr_az_angle**2 + a2*arr_az_angle + a3
    


    #Get inverse of equation
    func = lambda j: a1*j**2 + a2*j + a3
    invfunc = inversefunc(func)
    x1 = np.linspace(0,2*math.pi, 629)

    output = (invfunc(x1))


    fit1 = np.polyfit(x1,output,2)
    az1 = fit1[0]
    az2 = fit1[1]
    az3 = fit1[2]
    #az4 = fit1[3]

    print (az1,az2,az3)

    fit_equation1 = az1*x1**2 + az2*x1 + az3 

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(azimuthal_angle, fit_equation,color = 'r',alpha = 0.5, label = 'Polynomial fit')
    ax1.scatter(azimuthal_angle, azimuthal_cdf_vec_norm, s = 5, color = 'b', label = 'Data points')
    ax2.plot(x1, invfunc(x1),color = 'g', label = 'Inverse cdf func')
    ax2.plot(x1, fit_equation1, color = 'm', label = 'Inverse cdf fit')
    ax2.set_xlim([0,1])
    ax1.set_title('Polynomial fit to cdf and inverse cdf')
    ax1.legend()
    ax2.legend()
    plt.show()


