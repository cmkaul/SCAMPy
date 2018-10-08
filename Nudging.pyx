#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from ForcingReference cimport ForcingReferenceBase
from Variables cimport GridMeanVariables
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal
import pylab as plt
cdef class NudgingBase:
    def __init__(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Gr = Gr
        self.Ref = Ref
        self.relax_coeff = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal

        return
    cpdef update(self, GridMeanVariables GMV):
        return

cdef class NudgingStandard(NudgingBase):
    def __init__(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, ForcingReferenceBase FoRef):
        NudgingBase.__init__(self,Gr,Ref,GMV)
        self.t_ref = np.interp(Ref.p0_half, np.flipud(FoRef.pressure),
                               np.flipud(FoRef.temperature))
        self.qt_ref =  np.interp(Ref.p0_half, np.flipud(FoRef.pressure),
                                 np.flipud(FoRef.qt))
        self.u_ref =  np.interp(Ref.p0_half, np.flipud(FoRef.pressure),
                                np.flipud(FoRef.u))
        self.v_ref =  np.interp(Ref.p0_half, np.flipud(FoRef.pressure),
                                np.flipud(FoRef.v))
        self.qt_tendency = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.t_tendency = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.h_tendency = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.u_tendency = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.v_tendency = np.zeros((Gr.nzg,), dtype=np.double, order='c')



        return
    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.qt_tendency[k] = self.relax_coeff[k] * (self.qt_ref[k] - GMV.QT.values[k])
                self.t_tendency[k] = self.relax_coeff[k] * (self.t_ref[k] - GMV.T.values[k])
                qv = GMV.QT.values[k] - GMV.QL.values[k]
                self.h_tendency[k] = self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], self.qt_tendency[k], self.t_tendency[k])
                GMV.QT.tendencies[k] += self.qt_tendency[k]
                GMV.H.tendencies[k] += self.h_tendency[k]
        if self.nudge_uv:
                with nogil:
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        self.u_tendency[k] = self.relax_coeff[k] * (self.u_ref[k] - GMV.U.values[k])
                        self.v_tendency[k] = self.relax_coeff[k] * (self.v_ref[k] - GMV.V.values[k])
                        GMV.U.tendencies[k] += self.u_tendency[k]
                        GMV.V.tendencies[k] += self.v_tendency[k]

        #
        # plt.figure('nudge')
        # plt.plot(self.t_tendency[self.Gr.gw:self.Gr.nzg-self.Gr.gw], self.Ref.p0_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        # plt.plot(self.h_tendency[self.Gr.gw:self.Gr.nzg-self.Gr.gw], self.Ref.p0_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        # plt.gca().invert_yaxis()
        return





 # self.Nud = Nudging.NudgingStandard(Gr,self.FoRef)