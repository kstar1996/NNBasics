import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


# from -5 to 5 with 0.1 increment, make a numpy array
x = np.arange(-5.0, 5.0, 0.1)

y = step_function(x)

plt.plot(x, y)  # make x,y array into a graph
plt.ylim(-0.1, 1.1)  # set y axis range
plt.show()
