cdef struct entr_struct:
    double entr_sc
    double entr_w
    double detr_sc
    double detr_w

cdef entr_struct entr_detr_cloudy(double z, double z_half,  double zi) nogil
cdef entr_struct entr_detr_dry(double z, double z_half,  double zi) nogil
cdef entr_struct entr_detr_inverse_z(double z, double z_half,  double zi) nogil

cdef double get_wstar(double bflux, double zi )
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit)
cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil

