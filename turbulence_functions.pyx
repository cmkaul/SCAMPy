import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin, tanh, erf
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"
from thermodynamic_functions cimport *

# Entrainment Rates
cdef entr_struct entr_detr_dry(entr_in_struct entr_in)nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret.entr_sc = 0.5*(1.0/entr_in.z + 1.0/fmax(entr_in.zi - entr_in.z, 10.0)) #vkb/(z + 1.0e-3)
    _ret.detr_sc = 0.0

    return  _ret

cdef entr_struct entr_detr_inverse_z(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret

    _ret.entr_sc = vkb/entr_in.z
    _ret.detr_sc= 0.0

    return _ret

cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret

    eps_w = 1.0/(fmax(fabs(entr_in.w),1.0)* 500)
    #eps_w = 0.12*fabs(entr_in.b) / fmax(entr_in.w * entr_in.w, 1.0)
    if entr_in.af>0.0:
        partiation_func  = entr_detr_buoyancy_sorting(entr_in)
        _ret.entr_sc = partiation_func*eps_w/2.0
        _ret.detr_sc = (1.0-partiation_func/2.0)*eps_w
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return  _ret


cdef entr_struct entr_detr_tke2(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 3.0e-3
    else:
        _ret.detr_sc = 0.0

    # _ret.entr_sc = (0.002 * sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) /
    #                 fmax(entr_in.af, 0.001) / fmax(entr_in.ml, 1.0))
    _ret.entr_sc = (0.05 * sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(entr_in.af, 0.001) / fmax(entr_in.z, 1.0))
    return  _ret

# yair - this is a new entr-detr function that takes entr as proportional to TKE/w and detr ~ b/w2
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    _ret.detr_sc = fabs(entr_in.b)/ fmax(entr_in.w * entr_in.w, 1e-3)
    _ret.entr_sc = sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(sqrt(entr_in.af), 0.001) / 50000.0
    return  _ret
#
# cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
#     cdef entr_struct _ret
#     # in cloud portion from Soares 2004
#     if entr_in.z >= entr_in.zi :
#         _ret.detr_sc= 3.0e-3 +  0.2 * fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-4)
#     else:
#         _ret.detr_sc = 0.0
#
#     _ret.entr_sc = 0.2 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-4)
#     # or add to detrainment when buoyancy is negative
#     return  _ret



cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
    #if entr_in.ql_up >= 0.0:
        _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = 0.12 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)

    return  _ret

    # w_mix = (entr_in.w+entr_in.w_env)/2
    # eps_w = 0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    #
    # if entr_in.af>0.0:
    #
    #     partiation_func  = entr_detr_buoyancy_sorting(entr_in)
    #
    #     _ret.entr_sc = partiation_func*eps_w
    #     _ret.detr_sc = (1.0-partiation_func)*eps_w
    # else:
    #     _ret.entr_sc = 0.0
    #     _ret.detr_sc = 0.0
    # return  _ret
cdef double entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            #double[:] inner
            int i_b

            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim, bmix, qv_
            double partiation_func = 0.0
            double inner_partiation_func = 0.0
            eos_struct sa
            double [:] weights
            double [:] abscissas
        with gil:
            abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)
            #print np.multiply(weights[0],1.0), np.multiply(weights[1],1.0), np.multiply(weights[2],1.0)

        if entr_in.env_QTvar != 0.0 and entr_in.env_Hvar != 0.0:
            sd_q = sqrt(entr_in.env_QTvar)
            sd_h = sqrt(entr_in.env_Hvar)
            corr = fmax(fmin(entr_in.env_HQTcov/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

            # limit sd_q to prevent negative qt_hat
            sd_q_lim = (1e-10 - entr_in.qt_env)/(sqrt2 * abscissas[0])
            sd_q = fmin(sd_q, sd_q_lim)
            qt_var = sd_q * sd_q
            sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

            for m_q in xrange(entr_in.quadrature_order):
                qt_hat    = (entr_in.qt_env + sqrt2 * sd_q * abscissas[m_q] + entr_in.qt_up)/2.0
                mu_h_star = entr_in.H_env + sqrt2 * corr * sd_h * abscissas[m_q]
                inner_partiation_func = 0.0
                for m_h in xrange(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star + entr_in.H_up)/2.0
                    # condensation
                    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    # calcualte buoyancy
                    qv_ = qt_hat - sa.ql
                    alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

                    # sum only the points with positive buoyancy to get the buoyant fraction
                    if bmix >0.0:
                        inner_partiation_func  += weights[m_h] * sqpi_inv
                partiation_func  += inner_partiation_func * weights[m_q] * sqpi_inv

        else:
            h_hat = ( entr_in.H_env + entr_in.H_up)/2.0
            qt_hat = ( entr_in.qt_env + entr_in.qt_up)/2.0

            # condensation
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
            # calcualte buoyancy
            alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qt_hat - sa.ql)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

        return partiation_func

cdef double entr_detr_buoyancy_sorting_old(entr_in_struct entr_in) nogil:
    cdef:

        double wdw_mix, wdw_env, sigma, brel_mix, brel_env, w_env, dw_env
    # take random fractions of upd and env air to mix

    w_mix = (entr_in.w+entr_in.w_env)/2
    T_mean = (entr_in.T_up+entr_in.T_env)/2
    qt_mix = (entr_in.qt_up+entr_in.qt_env)/2
    ql_mix = (entr_in.ql_up+entr_in.ql_env)/2
    dw_mix = (entr_in.dw+entr_in.dw_env)/2
    # calculate thermo. properties
    #qv_mix = qt_mix-ql_mix
    thetal_ = t_to_thetali_c(entr_in.p0, T_mean,  qt_mix, ql_mix, 0.0)
    evap = evap_sat_adjust(entr_in.p0, thetal_, qt_mix)
    qv_mix = qt_mix-evap.ql
    Tmix = evap.T
    alpha_mix = alpha_c(entr_in.p0, Tmix, qt_mix, qv_mix)
    bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean
    alpha_env = alpha_c(entr_in.p0, entr_in.T_env, entr_in.qt_env, entr_in.qt_env-entr_in.ql_env)
    alpha_up = alpha_c(entr_in.p0, entr_in.T_up, entr_in.qt_up, entr_in.qt_up-entr_in.ql_up)
    b_env = buoyancy_c(entr_in.alpha0, alpha_env)  - entr_in.b_mean
    b_up = buoyancy_c(entr_in.alpha0, alpha_up)  - entr_in.b_mean
    wdw_mix = w_mix*dw_mix
    wdw_env = entr_in.w_env*entr_in.dw_env
    wdw_up = entr_in.w*entr_in.dw
    # b_rel = buoy + momentum
    brel_mix = bmix# + wdw_mix
    brel_env = b_env# + wdw_env
    brel_up = b_up# + wdw_up

    x0 = brel_mix/fmax(fabs(brel_env),1e-6)
    sigma = entr_in.Poisson_rand*(brel_up-brel_env)/fmax(fabs(brel_env),1e-6)
    if sigma == 0.0:
        partiation_func = 0.5
    else:
        partiation_func = (1-erf((brel_env/fmax(fabs(brel_env),1e-6)-x0)/(1.4142135623*sigma)))/2

    return partiation_func

cdef evap_struct evap_sat_adjust(double p0, double thetal_, double qt_mix) nogil:
    cdef:
        evap_struct evap
        double ql_1, T_2, ql_2, f_1, f_2, qv_mix, T_1
    if qt_mix>10.0:
                with gil:
                    print qt_mix
    qv_mix = qt_mix
    ql = 0.0

    pv_1 = pv_c(p0,qt_mix,qt_mix)
    pd_1 = p0 - pv_1

    # evaporate and cool
    T_1 = eos_first_guess_thetal(thetal_, pd_1, pv_1, qt_mix)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p0,qt_mix,pv_star_1)

    if(qt_mix <= qv_star_1):
        evap.T = T_1
        evap.ql = 0.0

    else:
        ql_1 = qt_mix - qv_star_1
        prog_1 = t_to_thetali_c(p0, T_1, qt_mix, ql_1, 0.0)
        f_1 = thetal_ - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt_mix)*cpd + qv_star_1 * cpv)
        delta_T  = fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt_mix,pv_star_2)
            pv_2 = pv_c(p0, qt_mix, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt_mix - qv_star_2
            prog_2 =  t_to_thetali_c(p0,T_2,qt_mix, ql_2, 0.0)
            f_2 = thetal_ - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = fabs(T_2 - T_1)

        evap.T  = T_2
        qv = qv_star_2
        evap.ql = ql_2

    return evap

# convective velocity scale
cdef double get_wstar(double bflux, double zi ):
    return cbrt(fmax(bflux * zi, 0.0))

# BL height
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit):
    cdef:
        double theta_rho_b = theta_rho[kmin]
        double h, Ri_bulk=0.0, Ri_bulk_low
        Py_ssize_t k = kmin


    # test if we need to look at the free convective limit
    if (u[kmin] * u[kmin] + v[kmin] * v[kmin]) <= 0.01:
        with nogil:
            for k in xrange(kmin,kmax):
                if theta_rho[k] > theta_rho_b:
                    break
        h = (z[k] - z[k-1])/(theta_rho[k] - theta_rho[k-1]) * (theta_rho_b - theta_rho[k-1]) + z[k-1]
    else:
        with nogil:
            for k in xrange(kmin,kmax):
                Ri_bulk_low = Ri_bulk
                Ri_bulk = g * (theta_rho[k] - theta_rho_b) * z[k]/theta_rho_b / (u[k] * u[k] + v[k] * v[k])
                if Ri_bulk > Ri_bulk_crit:
                    break
        h = (z[k] - z[k-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k-1]

    return h


cdef double get_mixing_tau(double zi, double wstar) nogil:
    # return 0.5 * zi / wstar
    return zi / (wstar + 0.001)




# MO scaling of near surface tke and scalar variance

cdef double get_surface_tke(double ustar, double wstar, double zLL, double oblength) nogil:
    if oblength < 0.0:
        return ((3.75 + cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar + 0.2 * wstar * wstar)
    else:
        return (3.75 * ustar * ustar)

cdef double get_surface_variance(double flux1, double flux2, double ustar, double zLL, double oblength) nogil:
    cdef:
        double c_star1 = -flux1/ustar
        double c_star2 = -flux2/ustar
    if oblength < 0.0:
        return 4.0 * c_star1 * c_star2 * pow(1.0 - 8.3 * zLL/oblength, -2.0/3.0)
    else:
        return 4.0 * c_star1 * c_star2



# Math-y stuff
cdef void construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return


cdef void construct_tridiag_diffusion_implicitMF(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *massflux, double *rho, double *alpha, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X + 0.5 * massflux[k-1] * dt * dzi/rho[k]
            b[k-gw] = 1.0 + Y/X + Z/X + 0.5 * dt * dzi * (massflux[k-1]-massflux[k])/rho[k]
            c[k-gw] = -Y/X - 0.5 * dt * dzi * massflux[k]/rho[k]

    return




cdef void construct_tridiag_diffusion_dirichlet(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
                Y = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return



cdef void tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c):
    cdef:
        double * scratch = <double*> PyMem_Malloc(nz * sizeof(double))
        Py_ssize_t i
        double m

    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    with nogil:
        for i in xrange(1,nz):
            m = 1.0/(b[i] - a[i] * scratch[i-1])
            scratch[i] = c[i] * m
            x[i] = (x[i] - a[i] * x[i-1])*m


        for i in xrange(nz-2,-1,-1):
            x[i] = x[i] - scratch[i] * x[i+1]


    PyMem_Free(scratch)
    return






# Dustbin

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag


