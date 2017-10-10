cimport EDMF_Updrafts
cimport EDMF_Environment
from Grid cimport Grid
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport  SurfaceBase
from ReferenceState cimport  ReferenceState
from Cases cimport CasesBase
from TimeStepping cimport  TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from turbulence_functions cimport entr_struct, entr_in_struct
from Turbulence cimport ParameterizationBase


cdef class EDMF_PrognosticTKE(ParameterizationBase):
    cdef:
        Py_ssize_t n_updrafts
        EDMF_Updrafts.UpdraftVariables UpdVar
        EDMF_Updrafts.UpdraftMicrophysics UpdMicro
        EDMF_Updrafts.UpdraftThermodynamics UpdThermo
        EDMF_Environment.EnvironmentVariables EnvVar
        EDMF_Environment.EnvironmentThermodynamics EnvThermo
        entr_struct (*entr_detr_fp) (entr_in_struct entr_in) nogil
        bint const_area
        bint use_local_micro
        bint similarity_diffusivity
        bint use_steady_updrafts
        double surface_area
        double minimum_area
        double entrainment_factor
        double detrainment_factor
        double vel_pressure_coeff
        double vel_buoy_coeff
        double dt_upd
        double [:,:] entr_sc
        double [:,:] detr_sc
        double [:] area_surface_bc
        double [:] h_surface_bc
        double [:] qt_surface_bc
        double [:] w_surface_bc
        double [:,:] m # mass flux
        double [:] massflux_h
        double [:] massflux_qt
        double [:] massflux_tke
        double [:] massflux_tendency_h
        double [:] massflux_tendency_qt
        double [:] diffusive_flux_h
        double [:] diffusive_flux_qt
        double [:] diffusive_tendency_h
        double [:] diffusive_tendency_qt
        double [:] mixing_length
        double [:] tke_buoy
        double [:] tke_dissipation
        double [:] tke_entr_gain
        double [:] tke_detr_loss
        double [:] tke_shear
        double max_area_factor
        double tke_ed_coeff
        double tke_diss_coeff

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS )
    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case)
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef compute_mixing_length(self, double obukhov_length)
    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef reset_surface_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals)
    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case)
    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_tke(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef initialize_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef compute_tke_buoy(self, GridMeanVariables GMV)
    cpdef compute_tke_dissipation(self)
    cpdef compute_tke_entr(self)
    cpdef compute_tke_detr(self)
    cpdef compute_tke_shear(self, GridMeanVariables GMV)
    # cpdef update_tke_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_tke_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS)
    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV)

