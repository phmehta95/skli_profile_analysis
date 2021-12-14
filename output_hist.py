import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#f= np.loadtxt('data11.dat', unpack='False')


#f = np.fromfile('data11.dat')
#g = np.fromfile('data11_diff.dat')
f= np.loadtxt('data11.dat', unpack='False')
g= np.loadtxt('data11_diff.dat', unpack='False')

# set bins' interval for your data
# You have following intervals: 
# 1st col is number of data elements in [0,10000);
# 2nd col is number of data elements in [10000, 20000); 
# ...
# last col is number of data elements in [100000, 200000]; 
 


fig1 = plt.figure()
plt.hist(np.cos(np.radians(f)), histtype='bar', bins = 'auto')
plt.xlim(xmin=0.9987, xmax=1)
plt.xlabel('Output cos(angle)')
plt.ylabel('No. of photons')
plt.title('Distribution of output angles')

fig2 = plt.figure()
plt.hist(np.cos(g), histtype='bar', bins = 'auto')
plt.xlim(xmin=0, xmax=1)
plt.xlabel('Output cos(angle)')
plt.ylabel('No. of photons')
plt.title('Distribution of output angles')



plt.show()

