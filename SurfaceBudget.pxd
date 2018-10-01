from TimeStepping cimport TimeStepping
from Surface cimport SurfaceBase
from Radiation cimport RadiationBase

cdef class SurfaceBudget:
    cdef:
        bint constant_sst
        double ocean_heat_flux
        double slab_depth
        double fixed_sst_time

    cpdef update(self, RadiationBase Ra, SurfaceBase Sur, TimeStepping TS)