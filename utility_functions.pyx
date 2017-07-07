import numpy as np
import scipy.special as sp
from libc.math cimport exp

cdef double interp2pt(double val1, double val2) nogil:
    return 0.5*(val1 + val2)



cdef double gaussian_mean(double lower_lim, double upper_lim):
    cdef:
        double upper_x = sp.erfinv(upper_lim)
        double lower_x = sp.erfinv(lower_lim)
        double upper_int = -np.exp(-upper_x*upper_x/2.0)/np.sqrt(2.0*np.pi)
        double lower_int = -np.exp(-lower_x*lower_x/2.0)/np.sqrt(2.0*np.pi)
        double res_fac = (upper_int - lower_int)/(upper_lim - lower_lim)
    return res_fac


cdef double logistic(double x, double slope, double mid) nogil:
    return 1.0/(1.0 + exp( -slope * (x-mid)))

