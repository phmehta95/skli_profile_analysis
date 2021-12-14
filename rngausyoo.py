from numpy.random import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import math

Z = random(100000)

RND = 21.398*Z**3-30.794*Z**2+88.422*Z-39.833
RND = abs(RND)*np.sqrt(abs(RND))
print RND
#RND = (RND/180)*np.pi
#RND = np.arctan(RND)
#RND = (RND/np.pi)*180

RND1 = 2000*Z
RND1 = np.sqrt(RND1**2/(3610.**2))
RND1 = np.sqrt(RND1)
RND1 = np.arctan(RND1)
#RND1 = (RND1/np.pi)*180
print RND1
#RND = abs(RND)
#RND = RND *(np.arcsin(Z))

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.hist(RND, range =[-300,300])
#plot2 = plt.figure(2)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.hist(RND1, range=[-1,1])

plt.show()






