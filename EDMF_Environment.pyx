#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
include "parameters.pxi"
import cython
from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from Variables cimport VariableDiagnostic, GridMeanVariables
from libc.math cimport fmax, fmin, sqrt, exp
from thermodynamic_functions cimport  *

cdef class EnvironmentVariable:
    def __init__(self, nz, loc, kind, name, units):
        self.values = np.zeros((nz,),dtype=np.double, order='c')
        self.flux = np.zeros((nz,),dtype=np.double, order='c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units





cdef class EnvironmentVariables:
    def __init__(self,  namelist, Grid Gr  ):
        cdef Py_ssize_t nz = Gr.nzg
        self.Gr = Gr

        self.W = EnvironmentVariable(nz, 'full', 'velocity', 'w','m/s' )

        self.QT = EnvironmentVariable( nz, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = EnvironmentVariable( nz, 'half', 'scalar', 'w','kg/kg' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 'thetal','K' )
        self.THL = EnvironmentVariable(nz, 'half', 'scalar', 'thetal', 'K')
        self.T = EnvironmentVariable( nz, 'half', 'scalar', 'temperature','K' )
        self.B = EnvironmentVariable( nz, 'half', 'scalar', 'buoyancy','m^2/s^3' )
        self.CF = EnvironmentVariable(nz, 'half', 'scalar','cloud_fraction', '-')

        # Determine whether we need 2nd moment variables
        if  namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            self.use_tke = True
            self.use_scalar_var = True
        else:
            self.use_tke = False
            self.use_scalar_var = True
        #Now add the 2nd moment variables
        if self.use_tke:
            self.TKE = EnvironmentVariable( nz, 'half', 'scalar', 'tke','m^2/s^2' )
        if self.use_scalar_var:
            self.QTvar = EnvironmentVariable( nz, 'half', 'scalar', 'qt_var','kg^2/kg^2' )
            if namelist['thermodynamics']['thermal_variable'] == 'entropy':
                self.Hvar = EnvironmentVariable(nz, 'half', 'scalar', 's_var', '(J/kg/K)^2')
                self.HQTcov = EnvironmentVariable(nz, 'half', 'scalar', 's_qt_covar', '(J/kg/K)(kg/kg)' )
            elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
                self.Hvar = EnvironmentVariable(nz, 'half', 'scalar', 'thetal_var', 'K^2')
                self.HQTcov = EnvironmentVariable(nz, 'half', 'scalar', 'thetal_qt_covar', 'K(kg/kg)' )
        #
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('env_w')
        Stats.add_profile('env_qt')
        Stats.add_profile('env_ql')
        if self.H.name == 's':
            Stats.add_profile('env_s')
        else:
            Stats.add_profile('env_thetal')
        Stats.add_profile('env_temperature')


        return
    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('env_w', self.W.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_qt', self.QT.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_ql', self.QL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 's':
            Stats.write_profile('env_s', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('env_thetal', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('env_temperature', self.T.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return

cdef class EnvironmentThermodynamics:
    def __init__(self, namelist, Grid Gr, ReferenceState Ref, EnvironmentVariables EnvVar):
        self.Gr = Gr
        self.Ref = Ref
        try:
            self.quadrature_order = namelist['condensation']['quadrature_order']
        except:
            self.quadrature_order = 5
        if EnvVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif EnvVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        self.thl_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.qt_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.t_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order ='c')

        return



    cdef void eos_update_SA_sgs(self, EnvironmentVariables EnvVar, VariableDiagnostic GMV_B):

        a, w = np.polynomial.hermite.hermgauss(self.quadrature_order)
        cdef:
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t k, m_q, m_h
            double [:] abscissas = a
            double [:] weights = w
            double  outer_int_ql, outer_int_T, outer_int_alpha, outer_int_cf
            double  inner_int_ql, inner_int_T, inner_int_alpha, inner_int_cf
            double inner_int_qt_cloudy, inner_int_T_cloudy, inner_int_thl_cloudy
            double outer_int_qt_cloudy, outer_int_T_cloudy, outer_int_thl_cloudy
            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double sqpi_inv = 1.0/sqrt(pi)
            double temp_m, alpha_m, qv_m, ql_m, qi_m, thetal_m
            double sqrt2 = sqrt(2.0)
            double sd_q_lim
            eos_struct sa

        if EnvVar.H.name == 'thetal':
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    sd_q = sqrt(EnvVar.QTvar.values[k])
                    sd_h = sqrt(EnvVar.Hvar.values[k])
                    corr = fmax(fmin(EnvVar.HQTcov.values[k]/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)
                    # limit sd_q to prevent negative qt_hat
                    sd_q_lim = (1e-10 - EnvVar.QT.values[k])/(sqrt2 * abscissas[0])
                    sd_q = fmin(sd_q, sd_q_lim)
                    qt_var = sd_q * sd_q
                    sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h
                    outer_int_alpha = 0.0
                    outer_int_T = 0.0
                    outer_int_ql = 0.0
                    outer_int_cf = 0.0
                    outer_int_qt_cloudy = 0.0
                    outer_int_thl_cloudy = 0.0
                    outer_int_T_cloudy = 0.0
                    for m_q in xrange(self.quadrature_order):
                        qt_hat    = EnvVar.QT.values[k] + sqrt2 * sd_q * abscissas[m_q]
                        mu_h_star = EnvVar.H.values[k]  + sqrt2 * corr * sd_h * abscissas[m_q]
                        inner_int_T     = 0.0
                        inner_int_ql    = 0.0
                        inner_int_alpha = 0.0
                        inner_int_cf    = 0.0
                        inner_int_qt_cloudy = 0.0
                        inner_int_T_cloudy = 0.0
                        inner_int_thl_cloudy = 0.0
                        for m_h in xrange(self.quadrature_order):
                            h_hat = sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star
                            sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k], qt_hat, h_hat)
                            temp_m = sa.T
                            ql_m = sa.ql
                            qv_m = EnvVar.QT.values[k] - ql_m
                            alpha_m = alpha_c(self.Ref.p0_half[k], temp_m, qt_hat, qv_m)
                            inner_int_ql    += ql_m    * weights[m_h] * sqpi_inv
                            inner_int_T     += temp_m  * weights[m_h] * sqpi_inv
                            inner_int_alpha += alpha_m * weights[m_h] * sqpi_inv
                            if ql_m  > ql_threshold:
                                inner_int_cf += weights[m_h] * sqpi_inv
                                inner_int_qt_cloudy += qt_hat * weights[m_h] * sqpi_inv
                                inner_int_thl_cloudy += h_hat * weights[m_h] * sqpi_inv
                                inner_int_T_cloudy += temp_m * weights[m_h] * sqpi_inv
                        outer_int_ql    += inner_int_ql    * weights[m_q] * sqpi_inv
                        outer_int_T     += inner_int_T     * weights[m_q] * sqpi_inv
                        outer_int_alpha += inner_int_alpha * weights[m_q] * sqpi_inv
                        outer_int_cf    += inner_int_cf    * weights[m_q] * sqpi_inv

                        outer_int_qt_cloudy += inner_int_qt_cloudy * weights[m_q] * sqpi_inv
                        outer_int_thl_cloudy += outer_int_thl_cloudy * weights[m_q] * sqpi_inv
                        outer_int_T_cloudy += outer_int_T_cloudy * weights[m_q] * sqpi_inv
                    EnvVar.QL.values[k]    = outer_int_ql
                    EnvVar.B.values[k] = g * (outer_int_alpha - self.Ref.alpha0_half[k])/self.Ref.alpha0_half[k] - GMV_B.values[k]
                    EnvVar.T.values[k]  = outer_int_T
                    EnvVar.CF.values[k]    = outer_int_cf
                    self.qt_cloudy[k] = outer_int_qt_cloudy
                    self.thl_cloudy[k] = outer_int_thl_cloudy
                    self.t_cloudy[k] = outer_int_T_cloudy
        elif EnvVar.H.name == 's':
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    sd_q = sqrt(EnvVar.QTvar.values[k])
                    sd_h = sqrt(EnvVar.Hvar.values[k])
                    corr = fmax(fmin(EnvVar.HQTcov.values[k]/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)
                    # limit sd_q to prevent negative qt_hat
                    sd_q_lim = (1e-10 - EnvVar.QT.values[k])/(sqrt2 * abscissas[0])
                    sd_q = fmin(sd_q, sd_q_lim)
                    qt_var = sd_q * sd_q
                    sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h
                    outer_int_alpha = 0.0
                    outer_int_T = 0.0
                    outer_int_ql = 0.0
                    outer_int_cf = 0.0
                    outer_int_qt_cloudy = 0.0
                    outer_int_thl_cloudy = 0.0
                    outer_int_T_cloudy = 0.0
                    for m_q in xrange(self.quadrature_order):
                        qt_hat    = EnvVar.QT.values[k] + sqrt2 * sd_q * abscissas[m_q]
                        mu_h_star = EnvVar.H.values[k]  + sqrt2 * corr * sd_h * abscissas[m_q]
                        inner_int_T     = 0.0
                        inner_int_ql    = 0.0
                        inner_int_alpha = 0.0
                        inner_int_cf    = 0.0
                        inner_int_qt_cloudy = 0.0
                        inner_int_T_cloudy = 0.0
                        inner_int_thl_cloudy = 0.0
                        for m_h in xrange(self.quadrature_order):
                            h_hat = sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star
                            sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k], qt_hat, h_hat)
                            temp_m = sa.T
                            ql_m = sa.ql
                            qv_m = EnvVar.QT.values[k] - ql_m
                            alpha_m = alpha_c(self.Ref.p0_half[k], temp_m, qt_hat, qv_m)
                            thetal_m = t_to_thetali_c(self.Ref.p0_half[k], temp_m, qt_hat, ql_m, 0.0)
                            inner_int_ql    += ql_m    * weights[m_h] * sqpi_inv
                            inner_int_T     += temp_m  * weights[m_h] * sqpi_inv
                            inner_int_alpha += alpha_m * weights[m_h] * sqpi_inv
                            if ql_m  > ql_threshold:
                                inner_int_cf += weights[m_h] * sqpi_inv
                                inner_int_qt_cloudy += qt_hat * weights[m_h] * sqpi_inv
                                inner_int_thl_cloudy += thetal_m * weights[m_h] * sqpi_inv
                                inner_int_T_cloudy += temp_m * weights[m_h] * sqpi_inv
                        outer_int_ql    += inner_int_ql    * weights[m_q] * sqpi_inv
                        outer_int_T     += inner_int_T     * weights[m_q] * sqpi_inv
                        outer_int_alpha += inner_int_alpha * weights[m_q] * sqpi_inv
                        outer_int_cf    += inner_int_cf    * weights[m_q] * sqpi_inv

                        outer_int_qt_cloudy += inner_int_qt_cloudy * weights[m_q] * sqpi_inv
                        outer_int_thl_cloudy += outer_int_thl_cloudy * weights[m_q] * sqpi_inv
                        outer_int_T_cloudy += outer_int_T_cloudy * weights[m_q] * sqpi_inv
                    EnvVar.QL.values[k]    = outer_int_ql
                    EnvVar.B.values[k] = g * (outer_int_alpha - self.Ref.alpha0_half[k])/self.Ref.alpha0_half[k] - GMV_B.values[k]
                    EnvVar.T.values[k]  = outer_int_T
                    EnvVar.CF.values[k]    = outer_int_cf
                    self.qt_cloudy[k] = outer_int_qt_cloudy
                    self.thl_cloudy[k] = outer_int_thl_cloudy
                    self.t_cloudy[k] = outer_int_T_cloudy
        return

    cpdef satadjust(self, EnvironmentVariables EnvVar, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw

            eos_struct sa
            double qv, alpha

        if GMV.use_scalar_var:
            self.eos_update_SA_sgs(EnvVar, GMV.B)
        else:
            with nogil:
                for k in xrange(gw,self.Gr.nzg-gw):
                    sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k], EnvVar.QT.values[k], EnvVar.H.values[k])
                    EnvVar.QL.values[k] = sa.ql
                    EnvVar.T.values[k] = sa.T
                    qv = EnvVar.QT.values[k] - EnvVar.QL.values[k]
                    alpha = alpha_c(self.Ref.p0_half[k], EnvVar.T.values[k], EnvVar.QT.values[k], qv)
                    EnvVar.B.values[k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) - GMV.B.values[k]
                    EnvVar.THL.values[k] = t_to_thetali_c(self.Ref.p0_half[k], EnvVar.T.values[k], EnvVar.QT.values[k],
                                                          EnvVar.QL.values[k], 0.0)
                    if EnvVar.QL.values[k] > 0.0:
                        EnvVar.CF.values[k] = 1.0
                    else:
                        EnvVar.CF.values[k] = 0.0
                    self.thl_cloudy[k] = EnvVar.THL.values[k]
                    self.qt_cloudy[k] = EnvVar.QT.values[k]
                    self.t_cloudy[k] = EnvVar.T.values[k]
        return