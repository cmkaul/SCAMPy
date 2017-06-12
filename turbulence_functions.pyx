import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow
include "parameters.pxi"


# Entrainment Rates

cdef entr_struct entr_detr_cloudy(double z, double z_half,  double zi) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i

    # in cloud portion from Soares 2004
    if z_half >= zi :
        _ret.entr_w = 2.0e-3
        _ret.entr_sc = 2.0e-3
        _ret.detr_w = 3.5e-3
        _ret.detr_sc= 3.5e-3
    else:
        # I think I just made this up to give a smooth blend
        _ret.entr_sc = 2.0e-3 * (1.0 - log(z_half/zi))
        _ret.entr_w = 2.0e-3 * (1.0 - log(z/zi))
        _ret.detr_w = (log(fmax(z,20.0)/(zi)) - log(20.0/(zi))) * 1e-3
        _ret.detr_sc = (log(fmax(z_half,20.0)/(zi)) - log(20.0/(zi))) * 1e-3

    return  _ret


cdef entr_struct entr_detr_dry(double z, double z_half, double zi) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret.entr_sc = 0.5*(1.0/fmax(z_half,10.0)+ 1.0/fmax(zi-z_half,10.0)) #vkb/(z + 1.0e-3)
    _ret.entr_w = 0.5*(1.0/fmax(z,10.0)+ 1.0/fmax(zi-z,10.0)) #vkb/z_half
    _ret.detr_w = 0.0
    _ret.detr_sc = 0.0

    return  _ret

cdef entr_struct entr_detr_inverse_z(double z, double z_half,  double zi) nogil:
    cdef:
        entr_struct _ret
        double er0_zmin = 10.0 # lower limit for z in computation of entrainment/detrainment rates
    _ret.entr_sc = vkb/fmax(z_half, er0_zmin)
    _ret.entr_w = vkb/fmax(z, er0_zmin)
    _ret.detr_sc= _ret.entr_sc
    _ret.detr_w = _ret.entr_w
    return _ret



# Other functions

cdef double get_wstar(double bflux, double zi ):
    return cbrt(bflux * zi)


cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit):
    cdef:
        double theta_rho_b = theta_rho[kmin]
        double h, Ri_bulk=0.0, Ri_bulk_low
        Py_ssize_t k = kmin


    # test if we need to look at the free convective limit
    if (u[kmin] * u[kmin] + v[kmin] * v[kmin]) <= 0.01:
        with nogil:
            for k in xrange(kmin,kmax):
                if theta_rho[k] > theta_rho_b:
                    break
        h = (z_half[k] - z_half[k-1])/(theta_rho[k] - theta_rho[k-1]) * (theta_rho_b - theta_rho[k-1]) + z_half[k-1]
    else:
        with nogil:
            for k in xrange(kmin,kmax):
                Ri_bulk_low = Ri_bulk
                Ri_bulk = g * (theta_rho[k] - theta_rho_b) * z_half[k]/theta_rho_b / (u[k] * u[k] + v[k] * v[k])
                if Ri_bulk > Ri_bulk_crit:
                    break
        h = (z_half[k] - z_half[k-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z_half[k-1]

    return h

# Dustbin

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag