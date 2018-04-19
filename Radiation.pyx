#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from TimeStepping cimport TimeStepping
cdef class RadiationBase:
    def __init__(self):
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
cdef class RadiationNone(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return
cdef class RadiationRRTM(RadiationBase):
    def __init__(self):
        RadiationBase.__init__(self)
        return
    cpdef initialize(self, Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        return
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS):
        return