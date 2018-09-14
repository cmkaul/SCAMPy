from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables
from TimeStepping cimport TimeStepping
from EDMF_Updrafts cimport UpdraftVariables
from EDMF_Environment cimport EnvironmentVariables
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
        double [:] gm_T_tendency
        double [:] gm_heating_rate_lw
        double [:] gm_heating_rate_sw
        double [:] gm_uflux_lw
        double [:] gm_dflux_lw
        double [:] gm_uflux_sw
        double [:] gm_dflux_sw
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface)

cdef class RadiationNone(RadiationBase):
    cpdef initialize(self, Grid Gr, ReferenceState Ref)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface)
cdef class RadiationRRTM(RadiationBase):
    cdef:
        bint compute_on_subdomains
        double co2_factor
        int dyofyr
        double adjes
        double scon
        double coszen
        double adif
        double adir
        double [:] pi_full
        double [:] p_full
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
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                 EnvironmentVariables EnvVar, TimeStepping TS, double Tsurface)
    cpdef zero_fluxes(self)