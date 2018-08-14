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
        bint use_local_micro
        bint similarity_diffusivity
        bint use_steady_updrafts
        bint calc_scalar_var
        double surface_area
        double minimum_area
        double entrainment_factor
        double detrainment_factor
        double vel_pressure_coeff # used by diagnostic plume option; now calc'ed from Tan et al 2018 coefficient set
        double vel_buoy_coeff # used by diagnostic plume option; now calc'ed from Tan et al 2018 coefficient set
        double pressure_buoy_coeff # Tan et al. 2018: coefficient alpha_b in Eq. 30
        double pressure_drag_coeff # Tan et al. 2018: coefficient alpha_d in Eq. 30
        double pressure_plume_spacing # Tan et al. 2018: coefficient r_d in Eq. 30
        double dt_upd
        double [:,:] entr_sc
        double [:,:] detr_sc
        double [:,:] updraft_pressure_sink
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
        double [:] tke_pressure
        double max_area_factor
        double tke_ed_coeff
        double tke_diss_coeff

        #double [:] Hvar
        #double [:] QTvar
        double [:] Hvar_shear
        double [:] QTvar_shear
        double [:] Hvar_entr_gain
        double [:] QTvar_entr_gain
        double [:] Hvar_detr_loss
        double [:] QTvar_detr_loss
        double [:] Hvar_diss_coeff
        double [:] QTvar_diss_coeff
        double [:] HQTcov
        double [:] HQTcov_shear
        double [:] HQTcov_entr_gain
        double [:] HQTcov_detr_loss
        double [:] HQTcov_diss_coeff
        double [:] Hvar_dissipation
        double [:] QTvar_dissipation
        double [:] HQTcov_dissipation
        double [:] Hvar_rain
        double [:] QTvar_rain
        double [:] HQTcov_rain

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case)
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef compute_mixing_length(self, double obukhov_length)
    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef reset_surface_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef reset_surface_covariance(self, GridMeanVariables GMV, CasesBase Case)
    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals)
    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case)
    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_tke(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_covariance(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef initialize_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef initialize_covariance(self, GridMeanVariables GMV, CasesBase Case)
    cpdef cleanup_covariance(self, GridMeanVariables GMV)
    cpdef compute_tke_buoy(self, GridMeanVariables GMV)
    cpdef compute_tke_dissipation(self)
    cpdef compute_covariance_dissipation(self)
    cpdef compute_tke_entr(self)
    cpdef compute_covariance_entr(self)
    cpdef compute_tke_detr(self)
    cpdef compute_covariance_detr(self)
    cpdef compute_tke_shear(self, GridMeanVariables GMV)
    cpdef compute_covariance_shear(self, GridMeanVariables GMV)
    cpdef compute_tke_pressure(self)
    cpdef compute_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV)
    cpdef update_tke_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS)
    cpdef update_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS)
    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV)
    cdef get_GMV_TKE(self, EDMF_Updrafts.UpdraftVariable au, EDMF_Updrafts.UpdraftVariable wu,
                      EDMF_Environment.EnvironmentVariable we, EDMF_Environment.EnvironmentVariable tke_e,
                      double *gmv_w, double *gmv_tke)
    cdef get_env_tke_from_GMV(self, EDMF_Updrafts.UpdraftVariable au, EDMF_Updrafts.UpdraftVariable wu,
                      EDMF_Environment.EnvironmentVariable we, EDMF_Environment.EnvironmentVariable tke_e,
                      double *gmv_w, double *gmv_tke)

    cdef get_GMV_CoVar(self, EDMF_Updrafts.UpdraftVariable au,
                        EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e,
                        EDMF_Environment.EnvironmentVariable covar_e,
                       double *gmv_phi, double *gmv_psi, double *gmv_covar)

    cdef get_env_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Environment.EnvironmentVariable covar_e,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar)

