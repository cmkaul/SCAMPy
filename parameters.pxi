#Adapated from PyCLES: https://github.com/pressel/pycles
cdef double pi = 3.14159265359
cdef double g = 9.80665
cdef double Rd = 287.1
cdef double Rv = 461.5
cdef double eps_v = 0.62210184182   # Rd / Rv
cdef double eps_vi = 1.60745384883  # Rv / Rd
cdef double cpd = 1004.0
cdef double cpv = 1859.0
cdef double cl = 4218.0
cdef double ci = 2106.0
cdef double kappa = 0.285956175299
cdef double Tf = 273.15
cdef double Tt = 273.16
cdef double T_tilde = 298.15
cdef double p_tilde = 100000.0
cdef double pv_star_t = 611.7
cdef double sd_tilde = 6864.8
cdef double sv_tilde = 10513.6
cdef double omega = 7.29211514671e-05
cdef double ql_threshold = 1e-08
cdef double vkb = 0.4
cdef double Pr0 = 1.0
cdef double beta_m = 4.8
cdef double beta_h = 7.8
cdef double gamma_m = 15.0
cdef double gamma_h = 9.0
# constants defined in Stevens et al 2005 (that are different from scampy)
# needed for DYCOMS case setup
cdef double dycoms_cp = 1015.
cdef double dycoms_L = 2.47 * 1e6
cdef double dycoms_Rd = 287.
