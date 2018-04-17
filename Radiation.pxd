from Grid cimport  Grid
from ReferenceState cimport ReferenceState

cdef class RadiationBase:
    cdef:
        Grid Gr
        ReferenceState Ref
        double srf_lw_up
        double srf_lw_down
        double srf_sw_up
        double srf_sw_down

cdef class RadiationNone(RadiationBase)

cdef class RadiationRRTM(RadiationBase)
