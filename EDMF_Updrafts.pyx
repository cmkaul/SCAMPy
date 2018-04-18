#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
from thermodynamic_functions cimport  *
from microphysics_functions cimport  *
import cython
cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from libc.math cimport fmax, fmin
import pylab as plt


cdef class UpdraftVariable:
    def __init__(self, nu, nz, loc, kind, name, units):
        self.values = np.zeros((nu,nz),dtype=np.double, order='c')
        self.old = np.zeros((nu,nz),dtype=np.double, order='c')  # needed for prognostic updrafts
        self.new = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.tendencies = np.zeros((nu,nz),dtype=np.double, order='c')
        self.flux = np.zeros((nu,nz),dtype=np.double, order='c')
        self.bulkvalues = np.zeros((nz,), dtype=np.double, order = 'c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid.Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        n_updrafts = np.shape(self.values)[0]

        if self.name == 'w':
            for i in xrange(n_updrafts):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in xrange(1,Gr.gw):
                    self.values[i,start_high+ k] = -self.values[i,start_high - k ]
                    self.values[i,start_low- k] = -self.values[i,start_low + k  ]
        else:
            for k in xrange(Gr.gw):
                for i in xrange(n_updrafts):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]

        return


cdef class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Grid.Grid Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t i, k

        self.W = UpdraftVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )
        self.Area = UpdraftVariable(nu, nzg, 'full', 'scalar', 'area_fraction','[-]' )
        self.QT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.QR = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qr','kg/kg' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )

        self.THL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T = UpdraftVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B = UpdraftVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )

        if namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            try:
                use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
            except:
                use_steady_updrafts = False
            if use_steady_updrafts:
                self.prognostic = False
            else:
                self.prognostic = True
            self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        else:
            self.prognostic = False
            self.updraft_fraction = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']

        self.cloud_base = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_top = np.zeros((nu,), dtype=np.double, order='c')


        return

    cpdef initialize(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):

                    self.W.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    if self.prognostic:
                        self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                    else:
                        self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = GMV.QT.values[k]
                    self.QL.values[i,k] = GMV.QL.values[k]
                    self.QR.values[i,k] = GMV.QR.values[k]
                    self.H.values[i,k] = GMV.H.values[k]
                    self.T.values[i,k] = GMV.T.values[k]
                    self.B.values[i,k] = 0.0
                self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.QR.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_qr')
        if self.H.name == 'thetal':
            Stats.add_profile('updraft_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_s')
        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_ts('cloud_cover')
        Stats.add_ts('cloud_base')
        Stats.add_ts('cloud_top')

        return

    cpdef set_means(self, GridMeanVariables GMV):

        cdef:
            Py_ssize_t i, k

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.QR.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.Area.bulkvalues[k] > 1.0e-20:
                    for i in xrange(self.n_updrafts):
                        self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                        self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                        self.QR.bulkvalues[k] += self.Area.values[i,k] * self.QR.values[i,k]/self.Area.bulkvalues[k]
                        self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                        self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                        self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                        self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                            /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))
                else:
                    self.QT.bulkvalues[k] = GMV.QT.values[k]
                    self.QR.bulkvalues[k] = GMV.QR.values[k]
                    self.QL.bulkvalues[k] = 0.0
                    self.H.bulkvalues[k] = GMV.H.values[k]
                    self.T.bulkvalues[k] = GMV.T.values[k]
                    self.B.bulkvalues[k] = 0.0
                    self.W.bulkvalues[k] = 0.0

        return
    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_new_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.new[i,k] = self.W.values[i,k]
                    self.Area.new[i,k] = self.Area.values[i,k]
                    self.QT.new[i,k] = self.QT.values[i,k]
                    self.QL.new[i,k] = self.QL.values[i,k]
                    self.QR.new[i,k] = self.QR.values[i,k]
                    self.H.new[i,k] = self.H.values[i,k]
                    self.THL.new[i,k] = self.THL.values[i,k]
                    self.T.new[i,k] = self.T.values[i,k]
                    self.B.new[i,k] = self.B.values[i,k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_old_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.old[i,k] = self.W.values[i,k]
                    self.Area.old[i,k] = self.Area.values[i,k]
                    self.QT.old[i,k] = self.QT.values[i,k]
                    self.QL.old[i,k] = self.QL.values[i,k]
                    self.QR.old[i,k] = self.QR.values[i,k]
                    self.H.old[i,k] = self.H.values[i,k]
                    self.THL.old[i,k] = self.THL.values[i,k]
                    self.T.old[i,k] = self.T.values[i,k]
                    self.B.old[i,k] = self.B.values[i,k]
        return
    # quick utility to set "tmp" arrays with values in the "new" arrays
    cpdef set_values_with_new(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.values[i,k] = self.W.new[i,k]
                    self.Area.values[i,k] = self.Area.new[i,k]
                    self.QT.values[i,k] = self.QT.new[i,k]
                    self.QL.values[i,k] = self.QL.new[i,k]
                    self.QR.values[i,k] = self.QR.new[i,k]
                    self.H.values[i,k] = self.H.new[i,k]
                    self.THL.values[i,k] = self.THL.new[i,k]
                    self.T.values[i,k] = self.T.new[i,k]
                    self.B.values[i,k] = self.B.new[i,k]
        return


    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            double cloud_cover= 0.0
            double cloud_base = 99999.0
            double cloud_top = -99999.0
            Py_ssize_t k

        Stats.write_profile('updraft_area', self.Area.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_w', self.W.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_qt', self.QT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_ql', self.QL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_qr', self.QR.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 'thetal':
            Stats.write_profile('updraft_thetal', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('updraft_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            #Stats.write_profile('updraft_thetal', self.THL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_temperature', self.T.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_buoyancy', self.B.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            if self.QL.bulkvalues[k] >0.0:
                cloud_cover = fmax(cloud_cover, self.Area.bulkvalues[k])
                cloud_base = fmin(cloud_base,self.Gr.z_half[k])
                cloud_top = fmax(cloud_top, self.Gr.z_half[k])
        Stats.write_ts('cloud_cover', cloud_cover)
        Stats.write_ts('cloud_base', cloud_base)
        Stats.write_ts('cloud_top', cloud_top)

        return

    cpdef get_cloud_base_top(self):
        cdef Py_ssize_t i, k

        for i in xrange(self.n_updrafts):
            self.cloud_base[i] = 99999.0
            self.cloud_top[i] = -99999.0
            for k in xrange(self.Gr.gw,self.Gr.nzg-self.Gr.gw):
                if self.QL.values[i,k] > 0.0:
                    self.cloud_base[i] = fmin(self.cloud_base[i], self.Gr.z_half[k])
                    self.cloud_top[i] = fmax(self.cloud_top[i], self.Gr.z_half[k])


        return

cdef class UpdraftThermodynamics:
    def __init__(self, n_updraft, Grid.Grid Gr, ReferenceState.ReferenceState Ref, UpdraftVariables UpdVar):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        return
    cpdef satadjust(self, UpdraftVariables UpdVar):
        #Update T, QL
        cdef:
            Py_ssize_t k, i
            eos_struct sa

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k],
                             UpdVar.QT.values[i,k], UpdVar.H.values[i,k])
                    UpdVar.QL.values[i,k] = sa.ql
                    UpdVar.T.values[i,k] = sa.T
        return

    cpdef buoyancy(self,  UpdraftVariables UpdVar, EnvironmentVariables EnvVar,GridMeanVariables GMV, bint extrap):
        cdef:
            Py_ssize_t k, i
            double alpha, qv, qt, t, h
            Py_ssize_t gw = self.Gr.gw

        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)


        if not extrap:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.nzg):
                        qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                        alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                        UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
        else:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        if UpdVar.Area.values[i,k] > 1e-3:
                            qt = UpdVar.QT.values[i,k]
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            h = UpdVar.H.values[i,k]
                            t = UpdVar.T.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)

                        else:
                            sa = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                                     qt, h)
                            qt -= sa.ql
                            qv = qt
                            t = sa.T
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
                for i in xrange(self.n_updraft):
                    GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
                for i in xrange(self.n_updraft):
                    UpdVar.B.values[i,k] -= GMV.B.values[k]
                EnvVar.B.values[k] -= GMV.B.values[k]

        return


#Implements a simple "microphysics" that clips excess humidity above a user-specified level
cdef class UpdraftMicrophysics:
    def __init__(self, paramlist, n_updraft, Grid.Grid Gr, ReferenceState.ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.prec_source_h = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_h_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.prec_source_qt_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        return

    cpdef compute_sources(self, UpdraftVariables UpdVar):
        """
        Compute precipitation source terms for QT, QR and H
        """
        cdef:
            Py_ssize_t k, i
            double tmp_qr

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    tmp_qr = acnv_instant(UpdVar.QL.values[i,k], UpdVar.QT.values[i,k], self.max_supersaturation,\
                                          UpdVar.T.values[i,k], self.Ref.p0_half[k])
                    self.prec_source_qt[i,k] = -tmp_qr
                    self.prec_source_h[i,k]  = rain_source_to_thetal(tmp_qr, self.Ref.p0_half[k], UpdVar.T.values[i,k]) 

        self.prec_source_h_tot  = np.sum(np.multiply(self.prec_source_h,  UpdVar.Area.values), axis=0)
        self.prec_source_qt_tot = np.sum(np.multiply(self.prec_source_qt, UpdVar.Area.values), axis=0)

        return

    cpdef update_updraftvars(self, UpdraftVariables UpdVar):
        """
        Apply precipitation source terms to QL, QR and H
        """
        cdef:
            Py_ssize_t k, i

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    UpdVar.QT.values[i,k] += self.prec_source_qt[i,k]
                    UpdVar.QL.values[i,k] += self.prec_source_qt[i,k]
                    UpdVar.QR.values[i,k] -= self.prec_source_qt[i,k]
                    UpdVar.H.values[i,k] += self.prec_source_h[i,k]
        return

    cdef void compute_update_combined_local_thetal(self, double p0, double T, double *qt, double *ql, double *qr, double *h,
                                               Py_ssize_t i, Py_ssize_t k) nogil :

        # Language note: array indexing must be used to dereference pointers in Cython. * notation (C-style dereferencing)
        # is reserved for packing tuples

        tmp_qr =  acnv_instant(ql[0], qt[0], self.max_supersaturation, T, p0)
        self.prec_source_qt[i,k] = -tmp_qr
        self.prec_source_h[i,k]  = rain_source_to_thetal(tmp_qr, p0, T) 

        qt[0] += self.prec_source_qt[i,k]
        ql[0] += self.prec_source_qt[i,k]
        qr[0] -= self.prec_source_qt[i,k]
        h[0]  += self.prec_source_h[i,k]

        return
