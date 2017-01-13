import numpy as np


cdef double interp2pt(double val1, double val2) nogil:
    return 0.5*(val1 + val2)



