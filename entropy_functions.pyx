import numpy as np
include "parameters.pxi"

cdef inline double sd_c(double pd, double T):
    return sd_tilde + cpd*np.log(T/T_tilde) -Rd*np.log(pd/p_tilde)


cdef inline double sv_c(double pv, double T):
    return sv_tilde + cpv*np.log(T/T_tilde) - Rv * np.log(pv/p_tilde)

cdef inline double sc_c(double L, double T):
    return -L/T