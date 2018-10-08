#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from Variables cimport GridMeanVariables, VariablePrognostic
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin

cdef class ForcingBase:
    def __init__(self):
        return
    cpdef initialize(self, GridMeanVariables GMV):
        self.subsidence = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dTdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dqtdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.ug = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.vg = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    cpdef update(self, GridMeanVariables GMV):
        return
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
        for k in xrange(gw, self.Gr.nzg-gw):
            U.tendencies[k] -= self.coriolis_param * (self.vg[k] - V.values[k])
            V.tendencies[k] += self.coriolis_param * (self.ug[k] - U.values[k])
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return


cdef class ForcingNone(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, GridMeanVariables GMV):
        ForcingBase.initialize(self, GMV)
        return
    cpdef update(self, GridMeanVariables GMV):
        return
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return



cdef class ForcingStandard(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, GridMeanVariables GMV):
        ForcingBase.initialize(self, GMV)
        return
    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
        if self.apply_subsidence:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                # Apply large-scale subsidence tendencies
                GMV.H.tendencies[k] -= (GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.subsidence[k]
                GMV.QT.tendencies[k] -= (GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.subsidence[k]


        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        return

# cdef class ForcingRadiative(ForcingBase): # yair - added to avoid zero subsidence
#     def __init__(self):
#         ForcingBase.__init__(self)
#         return
#     cpdef initialize(self, GridMeanVariables GMV):
#         ForcingBase.initialize(self, GMV)
#         return
#     cpdef update(self, GridMeanVariables GMV):
#         cdef:
#             Py_ssize_t k
#             double qv
#
#         for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
#             # Apply large-scale horizontal advection tendencies
#             qv = GMV.QT.values[k] - GMV.QL.values[k]
#             GMV.H.tendencies[k] += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv,
#                                                                 GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
#             GMV.QT.tendencies[k] += self.dqtdt[k]
#
#
#         return
#
#     cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
#         ForcingBase.coriolis_force(self, U, V)
#         return
#     cpdef initialize_io(self, NetCDFIO_Stats Stats):
#         return
#     cpdef io(self, NetCDFIO_Stats Stats):
#         return


cdef class ForcingDYCOMS_RF01(ForcingBase):

    def __init__(self):
        ForcingBase.__init__(self)
        return

    cpdef initialize(self, GridMeanVariables GMV):
        ForcingBase.initialize(self, GMV)

        self.alpha_z    = 1.
        self.kappa      = 85.
        self.F0         = 70.
        self.F1         = 22.
        self.divergence = 3.75e-6  # divergence is defined twice: here and in initialize_forcing method of DYCOMS_RF01 case class
                                   # where it is used to initialize large scale subsidence

        self.f_rad = np.zeros((self.Gr.nzg + 1), dtype=np.double, order='c') # radiative flux at cell edges
        return

    cpdef calculate_radiation(self, GridMeanVariables GMV):
        """
        see eq. 3 in Stevens et. al. 2005 DYCOMS paper
        """
        cdef:
            Py_ssize_t k

            double zi
            double rhoi

        # find zi (level of 8.0 g/kg isoline of qt)
        for k in xrange(self.Gr.gw, self.Gr.nzg - self.Gr.gw):
            if (GMV.QT.values[k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                zi     = self.Gr.z[idx_zi]
                rhoi   = self.Ref.rho0[idx_zi]
                break

        # cloud-top cooling
        q_0 = 0.0
    
        self.f_rad = np.zeros((self.Gr.nzg + 1), dtype=np.double, order='c')
        self.f_rad[self.Gr.nzg] = self.F0 * np.exp(-q_0)
        for k in xrange(self.Gr.nzg - 1, -1, -1):
            q_0           += self.kappa * self.Ref.rho0_half[k] * GMV.QL.values[k] * self.Gr.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[0] += self.F1 * np.exp(-q_1)
        for k in xrange(1, self.Gr.nzg + 1):
            q_1           += self.kappa * self.Ref.rho0_half[k - 1] * GMV.QL.values[k - 1] * self.Gr.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in xrange(0, self.Gr.nzg):
            if self.Gr.z[k] > zi:
                cbrt_z         = cbrt(self.Gr.z[k] - zi)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)
        # condition at the top
        cbrt_z                   = cbrt(self.Gr.z[k] + self.Gr.dz - zi)
        self.f_rad[self.Gr.nzg] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)

        for k in xrange(self.Gr.gw, self.Gr.nzg - self.Gr.gw):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / self.Gr.dz / self.Ref.rho0_half[k] / dycoms_cp

        return

    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V):
        ForcingBase.coriolis_force(self, U, V)
        return

    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv

        self.calculate_radiation(GMV)

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k]  += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            GMV.H.tendencies[k]  -= (GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.subsidence[k]
            GMV.QT.tendencies[k] -= (GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('rad_dTdt', self.dTdt[ self.Gr.gw     : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('rad_flux', self.f_rad[self.Gr.gw + 1 : self.Gr.nzg - self.Gr.gw + 1])
        return
