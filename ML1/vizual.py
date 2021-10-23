import numpy as np 
import matplotlib.pyplot as plt
import parser

def get_colomn(data, col):
    l = []
    for i in range(101):
        if i > 0:
            l.append(float(data[i][col]))
    return l

data = parser.csv_reader('nasa.csv')


x1 = np.arange(1, 101)
y1 = get_colomn(data, 1)

x2 = np.arange(1, 101)
y2 = get_colomn(data, 2) 

x3 = np.arange(1, 101)
y3 = get_colomn(data, 3) 

x4 = np.arange(1, 101)
y4 = get_colomn(data, 4) 

x5 = np.arange(1, 101)
y5 = get_colomn(data, 5) 

x6 = np.arange(1, 101)
y6 = get_colomn(data, 6) 

x7 = np.arange(1, 101)
y7 = get_colomn(data, 7) 

fig, ax = plt.subplots(7, 1)

ax[0].bar(x1, y1)
ax[1].bar(x2, y2)
ax[2].bar(x3, y3)
ax[3].bar(x4, y4)
ax[4].bar(x5, y5)
ax[5].bar(x6, y6)
ax[6].bar(x7, y7)


fig.set_facecolor('floralwhite')


plt.show()

