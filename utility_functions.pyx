import numpy as np
import scipy.special as sp
from libc.math cimport exp, log
from scipy.stats import norm
cimport cython

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_minimum(double [:] x, double a) nogil:    
    cdef:
      unsigned int i = 0
      double num, den
    num = 0; den = 0
    while(i<len(x)):
      if (x[i]>1.0e-5):
        num += x[i]*exp(-a*(x[i]))
        den += exp(-a*(x[i]))
      i += 1
    smin = num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_minimum2(double [:] x, double l0) nogil:
    cdef:
      unsigned int i = 0, numLengths = 0
      double smin = 0.0
    while(i<len(x)):
      if (x[i]>1.0e-5):
        smin += exp(-x[i]/l0)
        numLengths += 1
      i += 1
    smin /=  float(numLengths)
    smin =- l0*log(smin)
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_minimum3(double [:] x, double a) nogil:    
    cdef:
      unsigned int i = 0
      double num, den
    num = 0; den = 0
    while(i<len(x)):
      if (x[i]>1.0e-5):
        num += x[i]*exp(-a*(x[i]))
        den += exp(-a*(x[i]))
      i += 1
    smin = num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double softmin(double [:] x, double k):
    cdef:
      unsigned int i = 1, j = 1
      double smin = 0.0, lmin, num, den, eps = 0.1, lam = 1.0
    lmin = min(x)
    num = 1.0
    den = 1.0

    while(j<len(x)):
      if (x[j]-lmin>eps*lmin):
        lam = log( ( (1.0+lmin/(k*x[j]))**(1.0/(len(x)-1.0)) - 1.0 )**(-1) )
        lam /= ((x[j]-lmin)/lmin)
        break;
      j += 1
    while(i<len(x)):
      x[i] /= lmin
      num += x[i]*exp(-lam*(x[i]-1.0))
      den += exp(-lam*(x[i]-1.0))
      i += 1
    smin = lmin*num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double softmin2(double [:] x, double k):
    cdef:
      unsigned int i = 1, j = 1
      double smin = 0.0, lmin, num, den, eps = 1.0, lam = 1.0

    lmin = min(x)
    num = 1.0
    den = 1.0

    while(j<len(x)):
      if (x[j]-lmin>eps*lmin):
        lam = log(k*x[j]/lmin)
        lam /= ((x[j]-lmin)/lmin)
        break;
      j += 1
    print(lam)
    if lam < 1.0:
      lam = 1.0
    elif lam > 5.0:
      lam = 5.0
    while(i<len(x)):
      x[i] /= lmin
      num += x[i]*exp(-lam*(x[i]-1.0))
      den += exp(-lam*(x[i]-1.0))
      i += 1
    smin = lmin*num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double softmin3(double [:] x):
    cdef:
      unsigned int i = 1, j = 1
      double smin = 0.0, lmin, num, den, eps = 1.0, lam = 1.0

    lmin = min(x)
    lam = 100.0
    num = 1.0
    den = 1.0
    lam = 2.0
    while(i<len(x)):
      x[i] /= lmin
      num += x[i]*exp(-lam*(x[i]-1.0))
      den += exp(-lam*(x[i]-1.0))
      i += 1
    smin = lmin*num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double hardmin(double [:] x):
    cdef:
      unsigned i = 0
      double lmin = 1.0e6

    while(i<len(x)):
      if (x[i]>1.0e-5 and x[i]>lmin):
        lmin = x[i]
      i += 1

    return min(x) 



