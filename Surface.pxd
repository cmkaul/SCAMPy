from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from thermodynamic_functions cimport latent_heat,cpm_c

cdef class SurfaceBase:
    cdef:
        double zrough
        bint interactive_zrough
        double Tsurface
        double qsurface
        double shf
        double lhf
        double cm
        double ch
        double cq
        double bflux
        double windspeed
        double ustar
        double rho_qtflux
        double rho_hflux
        double rho_uflux
        double rho_vflux
        double obukhov_length
        double Ri_bulk_crit
        bint ustar_fixed
        Grid Gr
        ReferenceState Ref
    cpdef initialize(self)
    cpdef update(self, GridMeanVariables GMV)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceFixedFlux(SurfaceBase):
    cpdef initialize(self)
    cpdef update(self, GridMeanVariables GMV)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)


cdef class SurfaceFixedCoeffs(SurfaceBase):
    cdef:
        double s_surface
    cpdef initialize(self)
    cpdef update(self, GridMeanVariables GMV)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)

cdef class SurfaceMoninObukhov(SurfaceBase):
    cpdef initialize(self)
    cpdef update(self, GridMeanVariables GMV)
    cpdef free_convection_windspeed(self, GridMeanVariables GMV)
