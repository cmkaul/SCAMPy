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

        # GMV.QT.tendencies[gw] += self.qtflux * tendency_factor
        if GMV.H.name == 'thetal':
            self.rho_hflux = rho_tflux / exner_c(self.Ref.Pg)
            # GMV.H.tendencies[gw] += self.rho_hflux * tendency_factor
        elif GMV.H.name == 's':
            self.rho_hflux = entropy_flux(rho_tflux/self.Ref.rho0[gw-1],self.rho_qtflux/self.Ref.rho0[gw-1], self.Ref.p0_half[gw], GMV.T.values[gw], GMV.QT.values[gw])
            # GMV.H.tendencies[gw] += self.rho_hflux * tendency_factor

        cdef:
            double windspeed = np.maximum(np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw]), 0.01)


        if not self.ustar_fixed:
            self.ustar = compute_ustar(windspeed, self.bflux, self.zrough, self.Gr.z_half[gw])

        # GMV.U.tendencies[gw] -= self.ustar * self.ustar/windspeed * GMV.U.values[gw] * tendency_factor
        # GMV.V.tendencies[gw] -= self.ustar * self.ustar/windspeed * GMV.V.values[gw] * tendency_factor


        return





# cdef class SurfaceMom:
#     def __init__(self,namelist):
#         try:
#             self.ustar = namelist['surface']['ustar']
#         except:
#             self.ustar = 0.0
#         return
#     cpdef update(self, Variables.):
#         cdef double windspeed = np.maximum(np.sqrt(PV.u[0]*PV.u[0] + PV.v[0] *PV.v[0]), 0.01)
#         self.u_flux = -self.ustar * self.ustar/windspeed * PV.u[0]
#         self.v_flux = -self.ustar * self.ustar/windspeed * PV.v[0]
#
#         return


