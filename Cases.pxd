from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
from Surface cimport SurfaceBase
from Forcing cimport ForcingBase
from NetCDFIO cimport  NetCDFIO_Stats
from TimeStepping cimport  TimeStepping
from Radiation cimport RadiationBase
from ForcingReference cimport ForcingReferenceBase
from SurfaceBudget cimport SurfaceBudget
from Nudging cimport NudgingBase
from EDMF_Environment cimport EnvironmentVariables
from EDMF_Updrafts cimport UpdraftVariables

cdef class CasesBase:
    cdef:
        str casename
        str inversion_option
        SurfaceBase Sur
        ForcingBase Fo
        RadiationBase Ra
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class Soares(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class Bomex(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class life_cycle_Tan2018(CasesBase):
    cdef:
        double shf0
        double lhf0
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class Rico(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class TRMM_LBA(CasesBase):
    cdef:
        double [:] rad_time
        double [:,:] rad

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class ARM_SGP(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class GATE_III(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)


# Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus
# Stevens et al
# Monthly Weather Review 2005
# url: https://journals.ametsoc.org/doi/abs/10.1175/MWR2930.1
# doi: 10.1175/MWR2930.1

cdef class DYCOMS_RF01(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
cdef class ZGILS(CasesBase):
    cdef:
        Py_ssize_t location
        bint adjust_qt_adv_co2
        bint adjust_subsidence_co2
        double co2_factor
        ForcingReferenceBase FoRef
        SurfaceBudget SurBud
        NudgingBase Nud
        double t_adv_max
        double qt_adv_max
        double alpha_h
        double tau_relax_inverse
        double h_bl

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_radiation(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV, NetCDFIO_Stats Stats)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_radiation(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
    cpdef update(self, GridMeanVariables GMV, UpdraftVariables UpdVar,
                           EnvironmentVariables EnvVar, TimeStepping TS)
