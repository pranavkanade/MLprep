from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - (mean(xs**2))))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit_slope_and_intercept(xs, ys)

print(m, b)

#following is the line of the above data 
regression_line = [(m * x) + b for x in xs]

# predicting the y val for x val = 8
predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs, ys)

#plotting the predicted y for x val = 8
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()

# this is the basic program to calculate the slope of the line(best fit) using the mean of
# given element set.

# function best_fit_slope gives the formulae for the slope
 
# b is the y intercept of the line and the above function is modified from lab8.py to give
# the results of m and b together
