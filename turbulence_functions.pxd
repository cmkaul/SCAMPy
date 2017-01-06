cdef struct entr_struct:
    double entr_sc
    double entr_w
    double detr_sc
    double detr_w

cdef entr_struct entr_detr(double z, double z_half, bint above_cloudbase) nogil
cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil

