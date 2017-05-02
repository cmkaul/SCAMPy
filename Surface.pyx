#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from thermodynamic_functions cimport latent_heat, cpm_c, exner_c
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
            double tendency_factor = self.Ref.alpha0_half[gw]/self.Ref.alpha0[gw-1]*self.Gr.dzi
            double rho_tflux =  self.shf /(cpm_c(self.qsurface))

        self.rho_qtflux = self.lhf/(latent_heat(self.Tsurface))


        if GMV.H.name == 'thetal':
            self.rho_hflux = rho_tflux / exner_c(self.Ref.Pg)
        elif GMV.H.name == 's':
            self.rho_hflux = entropy_flux(rho_tflux/self.Ref.rho0[gw-1],self.rho_qtflux/self.Ref.rho0[gw-1], self.Ref.p0_half[gw], GMV.T.values[gw], GMV.QT.values[gw])

        cdef:
            double windspeed = np.maximum(np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw]), 0.01)
            double cp_ = cpm_c(GMV.QT.values[gw])
            double lv = latent_heat(GMV.T.values[gw])

        self.b_flux = (g * self.Ref.alpha0_half[gw]/cp_/GMV.T.values[gw]
                       * (self.shf + (eps_vi-1.0) * cp_ * GMV.T.values[gw] * self.lhf /lv))
        if not self.ustar_fixed:
            self.ustar = compute_ustar(windspeed, self.bflux, self.zrough, self.Gr.z_half[gw])

        self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.b_flux /vkb

        return



