############################
# Exercise 1: Drunken man  #
############################

import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(2494)

pmov1 = [[1,0],[0,1],[-1,0],[0,-1]]
pmov2 = [[1,0],[0,1],[-1,0],[0,-2]]
pmov3 = [[1,0],[-1,0]]

ip = [0,0]

######
def path_plot (initialpos, movements, nsteps, style):
    currentpos = initialpos
    for i in range(nsteps):
        currentpos = np.sum([currentpos, random.choice(movements)], axis=0)
        plt.plot(currentpos[0], currentpos[1], style)

#plt.figure(figsize = (10, 6))
path_plot(ip, pmov1, 200, 'r+')
path_plot(ip, pmov2, 200, 'g+')
path_plot(ip, pmov3, 200, 'b+')
plt.title('Random Walk for the Three Drunk Men')



plt.show()
