import ROOT
import ROOT.TF1
import numpy as np
from numpy.random import random
from scipy import interpolate
import matplotlib.pyplot as plt


def f(x):
    f1 = ROOT.TFile("skli_warwick_opticlib_analyser_v1.0.root", "READ")
    f  = f1.Get('b1/diffuser/b1_diffuser_theta_air')
    y = f.GetParameter(0)
    y1 = f.GetParameter(1)
    y2 = f.GetParameter(2)
    y3 = f.GetParameter(3)
    y4 = f.GetParameter(4)
    y5 = f.GetParameter(5)
    y6 = f.GetParameter(6)
    y7 = f.GetParameter(7)

    a = y
    b = y1
    c = y2
    d = y3
    e = y4
    f = y5
    g = y6
    h = y7

    print a 
    print b
    print c
    print d
    print e 
    print f
    print g
    print h

    x = x/1.3
    return (a+(b*x) + (c*x*x) +(d*x*x*x)+(e*x*x*x*x)+(f*x*x*x*x*x)+(g*x*x*x*x*x*x)+(h*x*x*x*x*x*x*x))*abs(np.sin(x))
#    return a+(b*x)

def sample(g):

    x = np.linspace(-40,40,160)
    y = g(x)    
#    print y   # probability density function, pdf
    cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
#    print cdf_y
    cdf_y = (cdf_y - cdf_y.min())/(cdf_y.max() - cdf_y.min())       # takes care of normalizing cdf to 1.0
#    print cdf_y
    inverse_cdf = interpolate.interp1d(cdf_y,x)    # this is a function
#    inverse_cdf = 77.8816*x - 39.6089
#    plot.plt(inverse_cdf)
    #print inverse_cdf
    return inverse_cdf

#inverse_cdf = 77.8816*x - 39.6089


def return_samples(N=1e6):
    
    uniform_samples = random(int(N))
    required_samples = sample(f)(uniform_samples)
    #print required_samples
    return required_samples




x = np.linspace(-50,50,160)
fig,ax = plt.subplots()
ax.set_xlabel('angle')
ax.set_ylabel('probability density')

ax.hist(return_samples(1e6),bins='auto',normed=True,range=(x.min(),x.max()))

plt.show() 
