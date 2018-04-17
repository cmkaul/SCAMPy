#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
from TimeStepping cimport TimeStepping
from Surface cimport SurfaceBase
from Radiation cimport RadiationBase

include "parameters.pxi"

cdef class SurfaceBudget:
    def __init__(self, namelist):

        try:
            self.constant_sst = namelist['surface_budget']['constant_sst']
        except:
            self.constant_sst = False

        try:
            self.ocean_heat_flux = namelist['surface_budget']['ocean_heat_flux']
        except:
            self.ocean_heat_flux = 0.0
        try:
            self.slab_depth = namelist['surface_budget']['slab_depth']
        except:
            self.slab_depth = 1.0 # default to 1 m slab ocean
        # Allow spin up time with fixed sst
        try:
            self.fixed_sst_time = namelist['surface_budget']['fixed_sst_time']
        except:
            self.fixed_sst_time = 0.0

        return

    cpdef update(self, RadiationBase Ra, SurfaceBase Sur, TimeStepping TS):
        cdef:
            double net_flux, tendency
        if self.constant_sst:
            return
        net_flux = -self.ocean_heat_flux - Ra.srf_lw_up - Ra.srf_sw_up \
                   - Sur.shf - Sur.lhf + Ra.srf_lw_down + Ra.srf_sw_down
        tendency = net_flux/cl/liquid_density/self.slab_depth
        Sur.Tsurface += tendency *TS.dt
        return


