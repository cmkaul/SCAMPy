#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from thermodynamic_functions cimport latent_heat, cpm_c, exner_c, qv_star_t, sd_c, sv_c, pv_star, theta_rho_c
from surface_functions cimport entropy_flux, compute_ustar, buoyancy_flux
from turbulence_functions cimport get_wstar, get_inversion
from Variables cimport GridMeanVariables
from libc.math cimport cbrt,fabs



cdef class SurfaceBase:
    def __init__(self, paramlist):
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']
        return
    cpdef initialize(self):
        return

    cpdef update(self, GridMeanVariables GMV):
        return
    cpdef free_convection_windspeed(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, gw = self.Gr.gw
            Py_ssize_t kmin = gw, kmax = self.Gr.nzg-gw
            double zi, wstar, qv
            double [:] theta_rho = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        # Need to get theta_rho
        with nogil:
            for k in xrange(self.Gr.nzg):
                qv = GMV.QT.values[k] - GMV.QL.values[k]
                theta_rho[k] = theta_rho_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)
        zi = get_inversion(&theta_rho[0], &GMV.U.values[0], &GMV.V.values[0], &self.Gr.z_half[0], kmin, kmax, self.Ri_bulk_crit)
        wstar = get_wstar(self.bflux, zi) # yair here zi in TRMM should be adjusted
        self.windspeed = np.sqrt(self.windspeed*self.windspeed  + (1.2 *wstar)*(1.2 * wstar) )
        return


cdef class SurfaceFixedFlux(SurfaceBase):
    def __init__(self,paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    cpdef initialize(self):
        return

    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, gw = self.Gr.gw
            double rho_tflux =  self.shf /(cpm_c(self.qsurface))

        self.windspeed = np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw])
        self.rho_qtflux = self.lhf/(latent_heat(self.Tsurface))

        if GMV.H.name == 'thetal':
            self.rho_hflux = rho_tflux / exner_c(self.Ref.Pg)
        elif GMV.H.name == 's':
            self.rho_hflux = entropy_flux(rho_tflux/self.Ref.rho0[gw-1],self.rho_qtflux/self.Ref.rho0[gw-1],
                                          self.Ref.p0_half[gw], GMV.T.values[gw], GMV.QT.values[gw])
        self.bflux = buoyancy_flux(self.shf, self.lhf, GMV.T.values[gw], GMV.QT.values[gw],self.Ref.alpha0[gw-1]  )

        if not self.ustar_fixed:
            # Correction to windspeed for free convective cases (Beljaars, QJRMS (1994), 121, pp. 255-270)
            # Value 1.2 is empirical, but should be O(1)
            if self.windspeed < 0.1:  # Limit here is heuristic
                if self.bflux > 0.0:
                   self.free_convection_windspeed(GMV)
                else:
                    print('WARNING: Low windspeed + stable conditions, need to check ustar computation')
                    print('self.bflux ==>',self.bflux )
                    print('self.shf ==>',self.shf)
                    print('self.lhf ==>',self.lhf)
                    print('GMV.U.values[gw] ==>',GMV.U.values[gw])
                    print('GMV.v.values[gw] ==>',GMV.V.values[gw])
                    print('GMV.QT.values[gw] ==>',GMV.QT.values[gw])
                    print('self.Ref.alpha0[gw-1] ==>',self.Ref.alpha0[gw-1])

            self.ustar = compute_ustar(self.windspeed, self.bflux, self.zrough, self.Gr.z_half[gw])

        self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb
        self.rho_uflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / self.windspeed * GMV.U.values[gw]
        self.rho_vflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / self.windspeed * GMV.V.values[gw]
        return
    cpdef free_convection_windspeed(self, GridMeanVariables GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return


# Cases such as Rico which provide values of transfer coefficients
cdef class SurfaceFixedCoeffs(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
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


        self.bflux = buoyancy_flux(self.shf, self.lhf, GMV.T.values[gw], GMV.QT.values[gw],self.Ref.alpha0[gw-1]  )


        self.ustar =  np.sqrt(self.cm) * windspeed
        # CK--testing this--EDMF scheme checks greater or less than zero,
        if fabs(self.bflux) < 1e-10:
            self.obukhov_length = 0.0
        else:
            self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        self.rho_uflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.U.values[gw]
        self.rho_vflux = - self.Ref.rho0[gw-1] *  self.ustar * self.ustar / windspeed * GMV.V.values[gw]
        return
    cpdef free_convection_windspeed(self, GridMeanVariables GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return

cdef class SurfaceMoninObukhov(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    cpdef initialize(self):
        return
    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, gw = self.Gr.gw

        self.windspeed = np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw])

        return

    cpdef free_convection_windspeed(self, GridMeanVariables GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return