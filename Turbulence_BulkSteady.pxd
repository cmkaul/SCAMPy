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
from Turbulence cimport ParameterizationBase


cdef class EDMF_BulkSteady(ParameterizationBase):
    cdef:
        double dt
        Py_ssize_t n_updrafts
        EDMF_Updrafts.UpdraftVariables UpdVar
        EDMF_Updrafts.UpdraftMicrophysics UpdMicro
        EDMF_Updrafts.UpdraftThermodynamics UpdThermo
        EDMF_Environment.EnvironmentVariables EnvVar
        entr_struct (*entr_detr_fp) (double z, double z_half, double zi) nogil
        bint const_area
        bint use_local_micro
        double surface_area
        double entrainment_factor
        double detrainment_factor
        double [:,:] entr_w
        double [:,:] entr_sc
        double [:,:] detr_w
        double [:,:] detr_sc
        double [:] area_surface_bc
        double [:] h_surface_bc
        double [:] qt_surface_bc
        double [:] w_surface_bc
        double [:,:] m # mass flux
        double [:] massflux_h
        double [:] massflux_qt
        double [:] massflux_tendency_h
        double [:] massflux_tendency_qt
        double [:] diffusive_flux_h
        double [:] diffusive_flux_qt
        double [:] diffusive_tendency_h
        double [:] diffusive_tendency_qt
        double surface_scalar_coeff
        double w_entr_coeff
        double w_buoy_coeff
        double max_area_factor

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS )
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals)
    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case)
    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case)
    cpdef solve_updraft_velocity(self)
    cpdef solve_area_fraction(self, GridMeanVariables GMV)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV)
    cpdef apply_updraft_microphysics(self)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)

    cpdef update_GMV_MF_implicitMF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED_implicitMF(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
