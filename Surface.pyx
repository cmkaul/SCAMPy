#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from thermodynamic_functions cimport latent_heat, cpm_c, exner_c, qv_star_t, sd_c, sv_c, pv_star
from surface_functions import entropy_flux, compute_ustar
from Variables cimport GridMeanVariables



cdef class SurfaceBase:
    def __init__(self):
        return
    cpdef initialize(self):
        return

    cpdef update(self, GridMeanVariables GMV):
        return






cdef class SurfaceFixedFlux(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    cpdef initialize(self):
        return

    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t gw = self.Gr.gw
            double rho_tflux =  self.shf /(cpm_c(self.qsurface))
            double windspeed = np.maximum(np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw]), 0.01)
            double cp_ = cpm_c(GMV.QT.values[gw])
            double lv = latent_heat(GMV.T.values[gw])

        self.rho_qtflux = self.lhf/(latent_heat(self.Tsurface))

        if GMV.H.name == 'thetal':
            self.rho_hflux = rho_tflux / exner_c(self.Ref.Pg)
        elif GMV.H.name == 's':
            self.rho_hflux = entropy_flux(rho_tflux/self.Ref.rho0[gw-1],self.rho_qtflux/self.Ref.rho0[gw-1],
                                          self.Ref.p0_half[gw], GMV.T.values[gw], GMV.QT.values[gw])

        self.bflux = (g * self.Ref.alpha0[gw-1]/cp_/GMV.T.values[gw]
                       * (self.shf + (eps_vi-1.0) * cp_ * GMV.T.values[gw] * self.lhf /lv))

        if not self.ustar_fixed:
            self.ustar = compute_ustar(windspeed, self.bflux, self.zrough, self.Gr.z_half[gw])

        self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        self.rho_uflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.U.values[gw]
        self.rho_vflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.V.values[gw]
        return


# Cases such as Rico which provide values of transfer coefficients
cdef class SurfaceFixedCoeffs(SurfaceBase):
    def __init__(self):
        SurfaceBase.__init__(self)
        return
    cpdef initialize(self):
        cdef:
            double pvg = pv_star(self.Tsurface)
            double pdg = self.Ref.Pg - pvg
        self.qsurface = qv_star_t(self.Ref.Pg, self.Tsurface)
        self.s_surface = (1.0-self.qsurface) * sd_c(pdg, self.Tsurface) + self.qsurface * sv_c(pvg,self.Tsurface)
        return

    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t gw = self.Gr.gw
            double windspeed = np.maximum(np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw]), 0.01)
            double cp_ = cpm_c(GMV.QT.values[gw])
            double lv = latent_heat(GMV.T.values[gw])
            double pv, pd, sv, sd


        self.rho_qtflux = -self.cq * windspeed * (GMV.QT.values[gw] - self.qsurface) * self.Ref.rho0[gw-1]
        self.lhf = lv * self.rho_qtflux

        if GMV.H.name == 'thetal':
            self.rho_hflux = -self.ch * windspeed * (GMV.H.values[gw] - self.Tsurface/exner_c(self.Ref.Pg)) * self.Ref.rho0[gw-1]
            self.shf = cp_  * self.rho_hflux
        elif GMV.H.name == 's':
            self.rho_hflux =  -self.ch * windspeed * (GMV.H.values[gw] - self.s_surface) * self.Ref.rho0[gw-1]
            pv = pv_star(GMV.T.values[gw])
            pd = self.Ref.p0_half[gw] - pv
            sv = sv_c(pv,GMV.T.values[gw])
            sd = sd_c(pd, GMV.T.values[gw])
            self.shf = (self.rho_hflux - self.lhf/lv * (sv-sd)) * GMV.T.values[gw]

        ## where I left off

        self.bflux = (g * self.Ref.alpha0[gw-1]/cp_/GMV.T.values[gw]
                       * (self.shf + (eps_vi-1.0) * cp_ * GMV.T.values[gw] * self.lhf /lv))


        self.ustar =  np.sqrt(self.cm) * windspeed

        self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        self.rho_uflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.U.values[gw]
        self.rho_vflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.V.values[gw]
        return
