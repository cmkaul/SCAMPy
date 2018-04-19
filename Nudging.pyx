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
        return





 # self.Nud = Nudging.NudgingStandard(Gr,self.FoRef)