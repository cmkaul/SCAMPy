import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, fabs,atan, exp, fmax, pow
include "parameters.pxi"

#Adapated from PyCLES: https://github.com/pressel/pycles

cdef  double sd_c(double pd, double T) nogil :
    return sd_tilde + cpd*log(T/T_tilde) -Rd*log(pd/p_tilde)


cdef  double sv_c(double pv, double T) nogil  :
    return sv_tilde + cpv*log(T/T_tilde) - Rv * log(pv/p_tilde)

cdef  double sc_c(double L, double T) nogil  :
    return -L/T

cdef double exner_c(double p0, double kappa = kappa) nogil  :
    return (p0/p_tilde)**kappa


cdef  double theta_c(double p0, double T) nogil :
    return T / exner_c(p0)


cdef  double thetali_c(double p0, double T, double qt, double ql, double qi, double L) nogil  :
    # Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    return theta_c(p0, T) * exp(-latent_heat(T)*(ql/(1.0 - qt) + qi/(1.0 -qt))/(T*cpd))

cdef  double theta_virt_c( double p0, double T, double qt, double ql, double qr) nogil :
    # Virtual potential temperature, mixing ratios are approximated by specific humidities.
    return theta_c(p0, T) * (1.0 + 0.61 * (qr) - ql);

cdef  double pd_c(double p0, double qt, double qv)  nogil :
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv)

cdef  double pv_c(double p0, double qt, double qv) nogil  :
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv)


cdef  double density_temperature_c(double T, double qt, double qv) nogil  :
    return T * (1.0 - qt + eps_vi * qv)

cdef  double theta_rho_c(double p0, double T, double qt, double qv) nogil  :
    return density_temperature_c(T,qt,qv)/exner_c(p0)


cdef  double cpm_c(double qt) nogil  :
    return (1.0-qt) * cpd + qt * cpv


cdef   double thetas_entropy_c(double s, double qt) nogil  :
    return T_tilde*exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt))


cdef  double thetas_t_c(double p0, double T, double qt, double qv, double qc, double L) nogil  :
    cdef double qd = 1.0 - qt
    cdef double pd_ = pd_c(p0,qt,qt-qc)
    cdef double pv_ = pv_c(p0,qt,qt-qc)
    cdef double cpm_ = cpm_c(qt)
    return T * pow(p_tilde/pd_,qd * Rd/cpm_)*pow(p_tilde/pv_,qt*Rv/cpm_)*exp(-L * qc/(cpm_*T))


cdef double entropy_from_thetas_c(double thetas, double qt)  nogil :
    return cpm_c(qt) * log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde


cdef  double buoyancy_c(double alpha0, double alpha)nogil  :
    return g * (alpha - alpha0)/alpha0

cdef double qv_star_c(const double p0, const double qt, const double pv) nogil  :
    return eps_v * (1.0 - qt) * pv / (p0 - pv)


cdef  double alpha_c(double p0, double T, double  qt, double qv) nogil  :
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv)


cdef   double t_to_entropy_c(double p0, double T,  double qt, double ql, double qi) nogil  :
    cdef double qv = qt - ql - qi
    cdef double pv = pv_c(p0, qt, qv)
    cdef double pd = pd_c(p0, qt, qv)
    cdef double L = latent_heat(T)
    return sd_c(pd,T) * (1.0 - qt) + sv_c(pv,T) * qt + sc_c(L,T)*(ql + qi)


cdef  double t_to_thetali_c(double p0, double T,  double qt, double ql, double qi) nogil  :
    cdef double L = latent_heat(T)
    return thetali_c(p0, T, qt, ql, qi, L)

cdef double pv_star(double T) nogil  :
    #    Magnus formula
    cdef double TC = T - 273.15
    return 6.1094*exp((17.625*TC)/float(TC+243.04))*100

cdef double qv_star_t(double p0, double T) nogil:
    cdef double pv = pv_star(T)
    return eps_v * pv / (p0 + (eps_v-1.0)*pv)

cdef  double latent_heat(double T) nogil  :
    cdef double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0



cdef  double eos_first_guess_thetal(double H, double pd, double pv, double qt)  nogil :
    cdef double p0 = pd + pv
    return H * exner_c(p0)

cdef double eos_first_guess_entropy(double H, double pd, double pv, double qt ) nogil   :
    cdef double qd = 1.0 - qt
    return (T_tilde *exp((H - qd*(sd_tilde - Rd *log(pd/p_tilde))
                              - qt * (sv_tilde - Rv * log(pv/p_tilde)))/((qd*cpd + qt * cpv))))






cdef eos_struct eos( double (*t_to_prog)(double, double,double,double, double) nogil,
                     double (*prog_to_t)(double,double, double, double) nogil,
                     double p0, double qt, double prog) nogil:
    cdef double qv = qt
    cdef double ql = 0.0

    cdef eos_struct _ret

    cdef double pv_1 = pv_c(p0,qt,qt )
    cdef double pd_1 = p0 - pv_1
    cdef double T_1 = prog_to_t(prog, pd_1, pv_1, qt)
    cdef double pv_star_1 = pv_star(T_1)
    cdef double qv_star_1 = qv_star_c(p0,qt,pv_star_1)

    cdef double ql_1, prog_1, f_1, T_2, delta_T
    cdef double qv_star_2, ql_2=0.0, pv_star_2, pv_2, pd_2, prog_2, f_2
    # If not saturated
    if(qt <= qv_star_1):
        _ret.T = T_1
        _ret.ql = 0.0

    else:
        ql_1 = qt - qv_star_1
        prog_1 = t_to_prog(p0, T_1, qt, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt)*cpd + qv_star_1 * cpv)
        delta_T  = fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt,pv_star_2)
            pv_2 = pv_c(p0, qt, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt - qv_star_2
            prog_2 =  t_to_prog(p0,T_2,qt, ql_2, 0.0   )
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = fabs(T_2 - T_1)

        _ret.T  = T_2
        qv = qv_star_2
        _ret.ql = ql_2

    return _ret


