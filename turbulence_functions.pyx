import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, fabs,atan, exp, fmax, pow
include "parameters.pxi"

cdef entr_struct entr_detr_cloudy(double z, double z_half, bint above_cloudbase, double zi) nogil:
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
        _ret.entr_sc = 2.0e-3 * (1.0 - log(z/zi))
        _ret.entr_w = 2.0e-3 * (1.0 - log(z_half/zi))
        _ret.detr_w = (log(fmax(z_half,20.0)/(zi)) - log(20.0/(zi))) * 1e-3
        _ret.detr_sc = (log(fmax(z,20.0)/(zi)) - log(20.0/(zi))) * 1e-3

    return  _ret


cdef entr_struct entr_detr_dry(double z, double z_half, bint above_cloudbase, double zi) nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret.entr_sc = 0.5*(1.0/(z+eps)+ 1.0/(fmax(1.2*zi-z,0.0)+eps)) #vkb/(z + 1.0e-3)
    _ret.entr_w = 0.5*(1.0/(z_half+eps)+ 1.0/(fmax(1.2*zi-z_half,0.0)+eps))#vkb/z_half
    _ret.detr_w = 0.0
    _ret.detr_sc = 0.0

    return  _ret

cdef entr_struct entr_detr_inverse_z(double z, double z_half, bint above_cloudbase, double zi) nogil:
    cdef:
        entr_struct _ret
        double er0_zmin = 1.0 # lower limit for z in computation of entrainment/detrainment rates
    _ret.entr_sc = vkb/fmax(z,er0_zmin)
    _ret.entr_w = vkb/fmax(z_half,er0_zmin)
    _ret.detr_sc= _ret.entr_sc
    _ret.detr_w = _ret.entr_w
    return _ret




cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag