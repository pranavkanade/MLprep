import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
plot1 = [1, 3]
plot2 = [2, 5]

euclidean_dist = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)

print(euclidean_dist)
