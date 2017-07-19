cdef struct entr_struct:
    double entr_sc
    double entr_w
    double detr_sc
    double detr_w

cdef entr_struct entr_detr_cloudy(double z, double z_half,  double zi, double wk, double w_halfk) nogil
cdef entr_struct entr_detr_dry(double z, double z_half,  double zi, double wk, double w_halfk) nogil
cdef entr_struct entr_detr_inverse_z(double z, double z_half,  double zi, double wk, double w_halfk) nogil
cdef entr_struct entr_detr_inverse_w(double z, double z_half,  double zi, double wk, double w_halfk) nogil

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