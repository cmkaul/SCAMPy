cdef struct entr_struct:
    double entr_sc
    double detr_sc

cdef struct entr_in_struct:
    double zi
    double z
    double w
    double b
    double af
    double tke
    double ml


cdef entr_struct entr_detr_cloudy(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_dry(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_z(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil


cdef double get_wstar(double bflux, double zi )
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit)

cdef double get_surface_tke(double ustar, double wstar, double zLL, double oblength) nogil
cdef double get_surface_variance(double flux1, double flux2, double ustar, double zLL, double oblength) nogil


cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil

cdef void construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a,
                                 double *b, double *c)
cdef void construct_tridiag_diffusion_implicitMF(Py_ssize_t nzg, Py_ssize_t gw,
                                            double dzi, double dt, double *rho_ae_K_m, double *massflux,
                                            double *rho, double *alpha, double *ae, double *a, double *b,
                                            double *c)
cdef void construct_tridiag_diffusion_dirichlet(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                           double *rho_ae_K_m, double *rho, double *ae, double *a,
                                           double *b, double *c)

cdef void tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c)