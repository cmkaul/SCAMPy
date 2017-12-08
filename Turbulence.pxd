cimport EDMF_Updrafts
cimport EDMF_Environment
from Grid cimport Grid
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport  SurfaceBase
from ReferenceState cimport  ReferenceState
from Cases cimport CasesBase
from TimeStepping cimport  TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from turbulence_functions cimport entr_struct

cdef class ParameterizationBase:
    cdef:
        double [:] turbulence_tendency
        double zi
        Grid Gr
        ReferenceState Ref
        VariableDiagnostic KM
        VariableDiagnostic KH
        double wstar
        double prandtl_number
        double Ri_bulk_crit
        bint extrapolate_buoyancy
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS, ReferenceState Ref)
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef compute_eddy_diffusivities_similarity(self, GridMeanVariables GMV, CasesBase Case)
    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV)


cdef class SimilarityED(ParameterizationBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS, ReferenceState Ref )
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV)


