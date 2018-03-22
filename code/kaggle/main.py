import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(open('winequality-red.csv', 'rt'), delimiter=",", skiprows=1)
x1 = data[:,0, None]
x2 = data[:,1, None]
x3 = data[:,2, None]
x4 = data[:,3, None]
x5 = data[:,4, None]
x6 = data[:,5, None]
x7 = data[:,6, None]
x8 = data[:,7, None]
x9 = data[:,8, None]
x10 = data[:,9, None]
x11 = data[:,10, None]
y = data[:,-1, None]
plt.scatter(x11, y)
# plt.scatter(data[:,1, None], y, color='r')
# plt.scatter(data[:,2, None], y, color='y')
plt.show()