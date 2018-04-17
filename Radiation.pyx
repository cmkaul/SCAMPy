#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
from Grid cimport Grid
from ReferenceState cimport ReferenceState

cdef class RadiationBase:
    def __init__(self,Grid Gr, ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        return
cdef class RadiationNone(RadiationBase):
    def __init__(self,Grid Gr, ReferenceState Ref):
        RadiationBase.__init__(self,Gr,Ref)
        return
cdef class RadiationRRTM(RadiationBase):
    def __init__(self,Grid Gr, ReferenceState Ref):
        RadiationBase.__init__(self,Gr,Ref)
        return