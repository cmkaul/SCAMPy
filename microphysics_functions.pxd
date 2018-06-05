cdef struct mph_struct:
    double T
    double thl
    double th
    double alpha
    double thl_rain_src
    double qt
    double qv
    double ql
    double qr

cdef double r2q(double r_, double qt) nogil
cdef double q2r(double q_, double qt) nogil
cdef double rain_source_to_thetal(double p0, double T, double qt, double ql, double qi, double qr) nogil
#cdef double rain_source_to_thetal(double qr, double p0, double T) nogil
cdef double acnv_instant(double ql, double qt, double sat_treshold, double T, double p0) nogil
cdef double acnv_rate(double ql, double qt) nogil
cdef double accr_rate(double ql, double qr, double qt) nogil
cdef double evap_rate(double rho, double qv, double qr, double qt, double T, double p0) nogil
cdef double terminal_velocity(double rho, double rho0, double qr, double qt) nogil
cdef mph_struct microphysics(double T, double ql, double p0, double qt,\
                             double max_supersat, bint in_Env) nogil
