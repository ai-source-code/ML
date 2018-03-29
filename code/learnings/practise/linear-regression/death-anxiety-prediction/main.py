import numpy as np
data = np.loadtxt(open('data.xlsx', 'rt'), delimiter=",", skiprows=1)


x = data[:,0,None]
y = data[:,1,None]

m = len(y)

print(data)