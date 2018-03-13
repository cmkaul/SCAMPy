import numpy as np
cimport numpy as np
from thermodynamic_functions cimport pv_c, pd_c, sv_c, sd_c, cpm_c, exner_c

cdef double convert_forcing_entropy(double p0, double qt, double qv, double T, double qt_tendency, double T_tendency) nogil:
    cdef:
        double pv = pv_c(p0, qt, qv)
        double pd = pd_c(p0, qt, qv)
    return cpm_c(qt) * T_tendency/T + (sv_c(pv,T)-sd_c(pd,T)) * qt_tendency

cdef double convert_forcing_thetal(double p0, double qt, double qv, double T, double qt_tendency, double T_tendency) nogil:
    return T_tendency/exner_c(p0)
