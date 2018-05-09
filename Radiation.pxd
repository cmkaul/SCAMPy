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
        double toa_lw_up
        double toa_sw_up
        double toa_sw_down
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
cdef class RadiationRRTM(RadiationBase):
    cdef:
        double co2_factor
        int dyofyr
        double adjes
        double scon
        double coszen
        double adif
        double adir
        double [:]
        double [:] o3vmr
        double [:] co2vmr
        double [:] ch4vmr
        double [:] n2ovmr
        double [:] o2vmr
        double [:] cfc11vmr
        double [:] cfc12vmr
        double [:] cfc22vmr
        double [:] ccl4vmr




    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, TimeStepping TS)
