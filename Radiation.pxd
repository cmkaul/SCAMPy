from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from TimeStepping cimport TimeStepping

cdef class RadiationBase:
    cdef:
        Grid Gr
        ReferenceState Ref
        double srf_lw_up
        double srf_lw_down
        double srf_sw_up
        double srf_sw_down
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
cdef class RadiationRRTM(RadiationBase):
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
