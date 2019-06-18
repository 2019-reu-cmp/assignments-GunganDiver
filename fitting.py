"""fitting.py -- fits a two overlapping Gaussian peaks
Starting Code:
Mike Moran
mmoran0032@gmail.com
2016-06-28
Benjamin Rose
benjamin.rose@me.com
2017-06-20
Chris Seymour
seymour.16@nd.edu
2019-06-18
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def gaus(x, A, mu, sigma):
    """
    Returns y-value, for each given x, making a Gaussian.
    Parameters
    ----------
    x : numpy.array
        The input values for the Gaussian function. Similar to the x values 
        used in a plotting command.
    A : float
        Maximum value of the Gaussian.
    mu : float
        Location (along x-axis) for the center of the Gaussian. Maybe outside 
        the range of `x`.
    sigma : float
        Width of the Gaussian.
    Return
    ------
    numpy.array
        A y-value (in a Gaussian shape) for each `x` given.
    """
    return A * np.exp(-(x - mu)**2/(2 * sigma**2))

def gaus2(x,A0,mu0,sigma0,A1,mu1,sigma1):
    '''
    Calculates a y value for a sum of two gaussians, with the same type of arguments as gaus(x,A,mu,sigma)
    '''
    return gaus(x,A0,mu0,s0)+gaus(x,A1,mu1,s1)

bins, counts = np.loadtxt('two_peaks.txt', unpack=True)

pars, _ = curve_fit(gaus2,bins,counts) # fitting method

plt.plot(bins, counts,label='data')
plt.plot(bins, w_gaus(bins, *pars),label='fit')
plt.legend()
plt.show()