cdef struct entr_struct:
    double entr_sc
    double detr_sc

cdef struct evap_struct:
    double T
    double ql



cdef struct entr_in_struct:
    double zi
    double wstar
    double z
    double dz
    double w
    double dw
    double b
    double dt
    double b_mean
    double b_env
    double af
    double tke
    double ml
    double T_mean
    double p0
    double alpha0
    double T_up
    double qt_up
    double ql_up
    double T_env
    double qt_env
    double ql_env
    double H_up
    double H_env
    double w_env
    double env_Hvar
    double env_QTvar
    double env_HQTcov
    double dw_env
    double L
    double tke_ed_coeff
    double Poisson_rand
    double logfn
    double zbl
    double poisson
    double n_up
    double thv_e
    double thv_u
    double dwdz
    double transport_der
    double dynamic_entr_detr
    long quadrature_order

cdef entr_struct entr_detr_dry(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_z(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil
cdef double entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_tke2(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_suselj(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_none(entr_in_struct entr_in) nogil
cdef evap_struct evap_sat_adjust(double p0, double thetal_, double qt_mix) nogil
cdef double get_wstar(double bflux, double zi )
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit)
cdef double get_mixing_tau(double zi, double wstar) nogil

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
