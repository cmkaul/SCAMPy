from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
from Surface cimport SurfaceBase
from Forcing cimport ForcingBase
from NetCDFIO cimport  NetCDFIO_Stats
from TimeStepping cimport  TimeStepping

cdef class CasesBase:
    cdef:
        str casename
        str inversion_option
        SurfaceBase Sur
        ForcingBase Fo
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

cdef class Soares(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)

cdef class Bomex(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)

cdef class life_cycle_Tan2018(CasesBase):
    cdef:
        double shf0
        double lhf0
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

cdef class Rico(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

cdef class TRMM_LBA(CasesBase):
    cdef:
        double [:] rad_time
        double [:,:] rad

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)

cdef class ARM_SGP(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS)

cdef class GATE_III(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

cdef class DYCOMS_RF01(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

cdef class GABLS(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)

# Still not fully implemented in Cases.pyx - Ignacio
cdef class SP(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS)


