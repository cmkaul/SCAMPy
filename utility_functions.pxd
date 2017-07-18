cdef double interp2pt(double val1, double val2) nogil
cdef double logistic(double x, double slope, double mid) nogil
cpdef double percentile_mean_norm(double percentile, Py_ssize_t nsamples)