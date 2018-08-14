#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True

import sys
import numpy as np
import cython
import pylab as plt
from Grid cimport Grid
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from ReferenceState cimport ReferenceState

from thermodynamic_functions cimport eos_struct, eos, t_to_entropy_c, t_to_thetali_c, \
    eos_first_guess_thetal, eos_first_guess_entropy, alpha_c, buoyancy_c

cdef class VariablePrognostic:
    def __init__(self,nz_tot,loc, kind, bc, name, units):
        # Value at the current timestep
        self.values = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Value at the next timestep, used for calculating turbulence tendencies
        self.new = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.mf_update = np.zeros((nz_tot,),dtype=np.double, order='c')
        self.tendencies = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Placement on staggered grid
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return

    cpdef zero_tendencies(self, Grid Gr):
        cdef:
            Py_ssize_t k
        with nogil:
            for k in xrange(Gr.nzg):
                self.tendencies[k] = 0.0
        return

    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1


        if self.bc == 'sym':
            for k in xrange(Gr.gw):
                self.values[start_high + k +1] = self.values[start_high  - k]
                self.values[start_low - k] = self.values[start_low + 1 + k]

                self.mf_update[start_high + k +1] = self.mf_update[start_high  - k]
                self.mf_update[start_low - k] = self.mf_update[start_low + 1 + k]

                self.new[start_high + k +1] = self.new[start_high  - k]
                self.new[start_low - k] = self.new[start_low + 1 + k]



        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0

            self.mf_update[start_high] = 0.0
            self.mf_update[start_low] = 0.0

            self.new[start_high] = 0.0
            self.new[start_low] = 0.0

            for k in xrange(1,Gr.gw):
                self.values[start_high+ k] = -self.values[start_high - k ]
                self.values[start_low- k] = -self.values[start_low + k  ]

                self.mf_update[start_high+ k] = -self.mf_update[start_high - k ]
                self.mf_update[start_low- k] = -self.mf_update[start_low + k  ]

                self.new[start_high+ k] = -self.new[start_high - k ]
                self.new[start_low- k] = -self.new[start_low + k  ]

        return

cdef class VariableDiagnostic:

    def __init__(self,nz_tot,loc, kind, bc, name, units):
        # Value at the current timestep
        self.values = np.zeros((nz_tot,),dtype=np.double, order='c')
        # Placement on staggered grid
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return
    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw


        if self.bc == 'sym':
            for k in xrange(Gr.gw):
                self.values[start_high + k] = self.values[start_high  - 1]
                self.values[start_low - k] = self.values[start_low + 1]


        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0
            for k in xrange(1,Gr.gw):
                self.values[start_high+ k] = 0.0  #-self.values[start_high - k ]
                self.values[start_low- k] = 0.0 #-self.values[start_low + k ]


        return



cdef class GridMeanVariables:
    def __init__(self, namelist, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref

        self.U = VariablePrognostic(Gr.nzg, 'half', 'velocity', 'sym','u', 'm/s' )
        self.V = VariablePrognostic(Gr.nzg, 'half', 'velocity','sym', 'v', 'm/s' )
        # Just leave this zero for now!
        self.W = VariablePrognostic(Gr.nzg, 'full', 'velocity','sym', 'v', 'm/s' )

        # Create thermodynamic variables
        self.QT = VariablePrognostic(Gr.nzg, 'half', 'scalar','sym', 'qt', 'kg/kg')
        self.QR = VariablePrognostic(Gr.nzg, 'half', 'scalar','sym', 'qr', 'kg/kg')

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = VariablePrognostic(Gr.nzg, 'half', 'scalar', 'sym','s', 'J/kg/K' )
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = VariablePrognostic(Gr.nzg, 'half', 'scalar', 'sym','thetal', 'K')
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal
        else:
            sys.exit('Did not recognize thermal variable ' + namelist['thermodynamics']['thermal_variable'])

        # Diagnostic Variables--same class as the prognostic variables, but we append to diagnostics list
        # self.diagnostics_list  = []
        self.QL = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'ql', 'kg/kg')
        self.T = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'temperature', 'K')
        self.B = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'buoyancy', 'm^2/s^3')
        self.THL = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 'thetal','K')

        # TKE
        if  namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            self.use_tke = True
        else:
            self.use_tke = False

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False
            print('Defaulting to non-calculation of scalar variances')

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        except:
            self.EnvThermo_scheme = 'sa_mean'
            print('Defaulting to saturation adjustment with respect to environmental means')

        #Now add the 2nd moment variables
        if self.use_tke:
            self.TKE = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'tke','m^2/s^2' )

        if self.calc_scalar_var:
            self.QTvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'qt_var','kg^2/kg^2' )
            if namelist['thermodynamics']['thermal_variable'] == 'entropy':
                self.Hvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym', 's_var', '(J/kg/K)^2')
                self.HQTcov = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym' ,'s_qt_covar', '(J/kg/K)(kg/kg)' )
            elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
                self.Hvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar', 'sym' ,'thetal_var', 'K^2')
                self.HQTcov = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym' ,'thetal_qt_covar', 'K(kg/kg)' )

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar = VariableDiagnostic(Gr.nzg, 'half', 'scalar','sym', 'thatav_var','K^2' )

        return

    cpdef zero_tendencies(self):
        self.U.zero_tendencies(self.Gr)
        self.V.zero_tendencies(self.Gr)
        self.QT.zero_tendencies(self.Gr)
        self.QR.zero_tendencies(self.Gr)
        self.H.zero_tendencies(self.Gr)
        return

    cpdef update(self,  TimeStepping TS):
        cdef:
            Py_ssize_t  k
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
                self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
                self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
                self.QT.values[k] +=  self.QT.tendencies[k] * TS.dt
                self.QR.values[k] +=  self.QR.tendencies[k] * TS.dt


        self.U.set_bcs(self.Gr)
        self.V.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)
        self.QT.set_bcs(self.Gr)
        self.QR.set_bcs(self.Gr)

        if self.use_tke:
            self.TKE.set_bcs(self.Gr)

        if self.calc_scalar_var:
            self.QTvar.set_bcs(self.Gr)
            self.Hvar.set_bcs(self.Gr)
            self.HQTcov.set_bcs(self.Gr)

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar.set_bcs(self.Gr)

        self.zero_tendencies()
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('u_mean')
        Stats.add_profile('v_mean')
        Stats.add_profile('qt_mean')
        Stats.add_profile('qr_mean')
        if self.H.name == 's':
            Stats.add_profile('s_mean')
            Stats.add_profile('thetal_mean')
        elif self.H.name == 'thetal':
            Stats.add_profile('thetal_mean')

        Stats.add_profile('temperature_mean')
        Stats.add_profile('buoyancy_mean')
        Stats.add_profile('ql_mean')
        if self.use_tke:
            Stats.add_profile('tke_mean')
        if self.calc_scalar_var:
            Stats.add_profile('Hvar_mean')
            Stats.add_profile('QTvar_mean')
            Stats.add_profile('HQTcov_mean')

        Stats.add_ts('lwp')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            double [:] arr = self.U.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw]
            double lwp = 0.0
            Py_ssize_t k
        Stats.write_profile('u_mean', arr)
        Stats.write_profile('v_mean',self.V.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('qt_mean',self.QT.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('ql_mean',self.QL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('qr_mean',self.QR.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('temperature_mean',self.T.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('buoyancy_mean',self.B.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 's':
            Stats.write_profile('s_mean',self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('thetal_mean',self.THL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        elif self.H.name == 'thetal':
            Stats.write_profile('thetal_mean',self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.use_tke:
            Stats.write_profile('tke_mean',self.TKE.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_scalar_var:
            Stats.write_profile('Hvar_mean',self.Hvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('QTvar_mean',self.QTvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('HQTcov_mean',self.HQTcov.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            lwp += self.Ref.rho0_half[k]*self.QL.values[k]*self.Gr.dz
        Stats.write_ts('lwp', lwp)

        return

    cpdef satadjust(self):
        cdef:
            Py_ssize_t k
            eos_struct sa
            double alpha, qv, qt, h, p0

        with nogil:
            for k in xrange(self.Gr.nzg):
                h = self.H.values[k]
                qt = self.QT.values[k]
                p0 = self.Ref.p0_half[k]
                sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, p0, qt, h )
                self.QL.values[k] = sa.ql
                self.T.values[k] = sa.T
                qv = qt - sa.ql
                self.THL.values[k] = t_to_thetali_c(p0, sa.T, qt, sa.ql,0.0)
                alpha = alpha_c(p0, sa.T, qt, qv)
                self.B.values[k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)

        return
