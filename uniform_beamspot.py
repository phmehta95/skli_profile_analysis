import numpy as np
import math
import matplotlib.pyplot as plt

x_array = []
inversecdf_array = []

for x in np.arange(0,1,0.001):
    x_array.append(x)
    inversecdf = 2000*x
    inversecdf = inversecdf * math.tan(x)
#    inversecdf = math.sqrt(inversecdf**2/(3610**2))
#    inversecdf = math.sqrt(inversecdf)
    
    inversecdf_array.append(inversecdf)
    

print x_array
print inversecdf_array

plt.plot(x_array,inversecdf_array)
plt.show()
