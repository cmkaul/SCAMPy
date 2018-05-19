import numpy as np
import scipy.special as sp
from libc.math cimport exp
from scipy.stats import norm

# compute the mean of the values above a given percentile (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficient for a single updraft or nth updraft of n updrafts
cpdef double percentile_mean_norm(double percentile, Py_ssize_t nsamples):
    cdef:
        double [:] x = norm.rvs(size=nsamples)
        double xp = norm.ppf(percentile)
    return np.ma.mean(np.ma.masked_less(x,xp))

# compute the mean of the values between two percentiles (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficients for 1 to n-1 updrafts when using n updrafts
cpdef double percentile_bounds_mean_norm(double low_percentile, double high_percentile, Py_ssize_t nsamples):
    cdef:
        double [:] x = norm.rvs(size=nsamples)
        double xp_low = norm.ppf(low_percentile)
        double xp_high = norm.ppf(high_percentile)
    return np.ma.mean(np.ma.masked_greater(np.ma.masked_less(x,xp_low),xp_high))


cdef double interp2pt(double val1, double val2) nogil:
    return 0.5*(val1 + val2)

cdef double logistic(double x, double slope, double mid) nogil:
    return 1.0/(1.0 + exp( -slope * (x-mid)))

cdef double smooth_minimum(double x1, double x2,double x3, double x4,double x5,  double a) nogil:
    smin = (x1*exp(-a*x1)+x2*exp(-a*x2)+x3*exp(-a*x3)+x4*exp(-a*x4)+x5*exp(-a*x5))\
           / (exp(-a*x1)+exp(-a*x2)+exp(-a*x3)+exp(-a*x4)+exp(-a*x5))
    return smin
