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
cdef class NudgingBase:
    def __init__(self):
        return
    cpdef update(self):
        return

cdef class NudgingStandard(NudgingBase):
    def __init__(self, Grid Gr, ReferenceState Ref, ForcingReferenceBase FoRef):
        self.relax_coeff = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.t_ref = np.interp(Ref.p0_half, np.flipud(self.FoRef.pressure), np.flipud(self.FoRef.temperature))
        self.qt_ref =  np.interp(Ref.p0_half, np.flipud(self.FoRef.pressure), np.flipud(self.FoRef.qt))
        self.u_ref =  np.interp(Ref.p0_half, np.flipud(self.FoRef.pressure), np.flipud(self.FoRef.u))
        self.v_ref =  np.interp(Ref.p0_half, np.flipud(self.FoRef.pressure), np.flipud(self.FoRef.v))
        return
    cpdef update(self)





 # self.Nud = Nudging.NudgingStandard(Gr,self.FoRef)