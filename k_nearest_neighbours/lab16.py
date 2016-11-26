import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
#
# above loop can be written as follows 

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0], new_features[1], s=100)
plt.show()

