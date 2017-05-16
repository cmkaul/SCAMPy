cdef struct entr_struct:
    double entr_sc
    double entr_w
    double detr_sc
    double detr_w

cdef entr_struct entr_detr_cloudy(double z, double z_half,  double zi) nogil
cdef entr_struct entr_detr_dry(double z, double z_half,  double zi) nogil
cdef entr_struct entr_detr_inverse_z(double z, double z_half,  double zi) nogil
cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil

