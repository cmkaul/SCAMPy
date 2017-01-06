import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, fabs,atan, exp, fmax, pow
include "parameters.pxi"

cdef entr_struct entr_detr(double z, double z_half, bint above_cloudbase) nogil:
    cdef entr_struct _ret

    if above_cloudbase:
        _ret.entr_w = 2.0e-3
        _ret.entr_sc = 2.0e-3
        _ret.detr_w = 3.0e-3
        _ret.detr_sc= 3.0e-3
    else:
        _ret.entr_w = vkb/(z + 1.0e-3)
        _ret.entr_sc = vkb/z_half
        _ret.detr_w = 0.0
        _ret.detr_sc = 0.0

    return  _ret

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag