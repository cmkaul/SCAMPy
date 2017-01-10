import numpy as np
import pylab as plt
include "parameters.pxi"
import cython

from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
cimport Surface
cimport Forcing
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport exner_c, t_to_entropy_c, latent_heat, cpm_c, thetali_c

def CasesFactory(namelist):
    if namelist['meta']['casename'] == 'Soares':
        return Soares()
    elif namelist['meta']['casename'] == 'Bomex':
        return Bomex()


cdef class CasesBase:
    def __init__(self):
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref ):
        return
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV ):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_ts('Tsurface')
        Stats.add_ts('shf')
        Stats.add_ts('lhf')
        Stats.add_ts('ustar')
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_ts('Tsurface', self.Sur.Tsurface)
        Stats.write_ts('shf', self.Sur.shf)
        Stats.write_ts('lhf', self.Sur.lhf)
        Stats.write_ts('ustar', self.Sur.ustar)
        return
    cpdef update_surface(self, GridMeanVariables GMV):
        return


cdef class Soares(CasesBase):
    def __init__(self):
        self.casename = 'Soares2004'
        self.Sur = Surface.SurfaceFixedFlux()
        self.Fo = Forcing.ForcingStandard()
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1000.0 * 100.0
        Ref.qtg = 4.5e-3
        Ref.Tg = 300.0
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] theta = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            double ql = 0.0, qi = 0.0

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            if Gr.z_half[k] <= 1350.0:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4* Gr.z_half[k]/1000.0
                theta[k] = 300.0

            else:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4 * 1.35 - 9.4e-4 * (Gr.z_half[k]-1350.0)/1000.0
                theta[k] = 300.0 + 2.0 * (Gr.z_half[k]-1350.0)/1000.0
            GMV.U.values[k] = 0.01

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = theta[k]
                GMV.T.values[k] =  theta[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = theta[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = theta[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        plt.figure(1)
        plt.plot(GMV.QT.values, Gr.z_half)
        plt.show()



        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref ):
        self.Sur.zrough = 1.0e-4
        self.Sur.Tsurface = 300.0
        self.Sur.qsurface = 5e-3
        self.Sur.lhf = 2.5e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 6.0e-2 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = False
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux = g * ( 6.0e-2/self.Sur.Tsurface + (eps_vi -1.0)* 2.5e-5)
        self.Sur.initialize()

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV):
        self.Sur.update(GMV)
        return

cdef class Bomex(CasesBase):
    def __init__(self):
        self.casename = 'Bomex'
        self.Sur = Surface.SurfaceFixedFlux()
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.015e5  #Pressure at ground
        Ref.Tg = 300.4  #Temperature at ground
        Ref.qtg = 0.02245   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Bomex is cloud-free

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set Thetal profile
            if Gr.z_half[k] <= 520.:
                thetal[k] = 298.7
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                thetal[k] = 298.7 + (Gr.z_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000:
                thetal[k] = 302.4 + (Gr.z_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
            if Gr.z_half[k] > 2000.0:
                thetal[k] = 308.2 + (Gr.z_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

            #Set qt profile
            if Gr.z_half[k] <= 520:
                GMV.QT.values[k] = (17.0 + (Gr.z_half[k]) * (16.3-17.0)/520.0)/1000.0
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                GMV.QT.values[k] = (16.3 + (Gr.z_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000.0:
                GMV.QT.values[k] = (10.7 + (Gr.z_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
            if Gr.z_half[k] > 2000.0:
                GMV.QT.values[k] = (4.2 + (Gr.z_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0


            #Set u profile
            if Gr.z_half[k] <= 700.0:
                GMV.U.values[k] = -8.75
            if Gr.z_half[k] > 700.0:
                GMV.U.values[k] = -8.75 + (Gr.z_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        plt.figure(1)
        plt.plot(GMV.QT.values, Gr.z_half)
        plt.figure(2)
        plt.plot(GMV.H.values, Gr.z_half)
        plt.show()

        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = 299.1 * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux = (g * ((8.0e-3 + (eps_vi-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (eps_vi-1) * 22.45e-3))))
        self.Sur.initialize()
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Set large-scale cooling
            if Gr.z_half[k] <= 1500.0:
                self.Fo.dTdt[k] =  (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])
            else:
                self.Fo.dTdt[k] = (-2.0/(3600 * 24.0) + (Gr.z_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])
            # Set large-scale drying
            if Gr.z_half[k] <= 300.0:
                self.Fo.dqtdt[k] = -1.2e-8   #kg/(kg * s)
            if Gr.z_half[k] > 300.0 and Gr.z_half[k] <= 500.0:
                self.Fo.dqtdt[k] = -1.2e-8 + (Gr.z_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

            #Set large scale subsidence
            if Gr.z_half[k] <= 1500.0:
                self.Fo.subsidence[k] = 0.0 + Gr.z_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
            if Gr.z_half[k] > 1500.0 and Gr.z_half[k] <= 2100.0:
                self.Fo.subsidence[k] = -0.65/100 + (Gr.z_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV):
        self.Sur.update(GMV)
        return










