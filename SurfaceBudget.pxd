from TimeStepping cimport TimeStepping
from Surface cimport SurfaceBase
from Radiation cimport RadiationBase

cdef class SurfaceBudget:
    cpdef update(self, RadiationBase Ra, SurfaceBase Sur, TimeStepping TS)