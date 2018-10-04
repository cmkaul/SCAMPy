cdef struct eos_struct:
    double T
    double ql

cdef double sd_c(double pd, double T)   nogil
cdef double sv_c(double pv, double T)   nogil
cdef double sc_c(double L, double T)   nogil
cdef double exner_c(double p0, double kappa=?)   nogil
cdef double theta_c(double p0, double T)   nogil
cdef double thetali_c(double p0, double T, double qt, double ql, double qi, double L)  nogil
cdef double theta_virt_c(double p0, double T, double qt, double ql, double qr)  nogil
cdef double pd_c(double p0, double qt, double qv)   nogil
cdef double pv_c(double p0, double qt, double qv)   nogil
cdef double density_temperature_c(double T, double qt, double qv)   nogil
cdef double theta_rho_c(double p0, double T, double qt, double qv)   nogil
cdef double cpm_c(double qt)   nogil
cdef double thetas_entropy_c(double s, double qt)   nogil
cdef double thetas_t_c(double p0, double T, double qt, double qv, double qc, double L)   nogil
cdef double entropy_from_thetas_c(double thetas, double qt)   nogil
cdef double buoyancy_c(double alpha0, double alpha)   nogil
cdef double qv_star_c(double p0, double qt,  double pv)   nogil
cdef double alpha_c(double p0, double T, double  qt, double qv) nogil
cdef double t_to_entropy_c(double p0, double T,  double qt, double ql, double qi)   nogil
cdef double t_to_thetali_c(double p0, double T,  double qt, double ql, double qi)   nogil
cdef double pv_star(double T)   nogil
cdef double qv_star_t(double p0, double T) nogil
cdef double latent_heat(double T) nogil
cdef double eos_first_guess_thetal(double H, double pd, double pv, double qt)    nogil
cdef double eos_first_guess_entropy(double H, double pd, double pv, double qt )   nogil
cdef eos_struct eos( double (*t_to_prog)(double, double, double, double, double) nogil,
                     double (*prog_to_t)(double, double, double, double) nogil,
                     double p0, double qt, double prog) nogil
