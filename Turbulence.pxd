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
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS )
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef compute_eddy_diffusivities_similarity(self, GridMeanVariables GMV, CasesBase Case)


cdef class EDMF_PrognosticTKE(ParameterizationBase):
    cdef:
        Py_ssize_t n_updrafts
        EDMF_Updrafts.UpdraftVariables UpdVar
        EDMF_Updrafts.UpdraftMicrophysics UpdMicro
        EDMF_Updrafts.UpdraftThermodynamics UpdThermo
        EDMF_Environment.EnvironmentVariables EnvVar
        EDMF_Environment.EnvironmentThermodynamics EnvThermo
        Py_ssize_t updraft_iterations
        entr_struct (*entr_detr_fp) (double z, double z_half, double zi) nogil
        bint const_area
        bint use_local_micro
        bint similarity_diffusivity
        double surface_area
        double [:,:] entr_w
        double [:,:] entr_sc
        double [:,:] entrB
        double [:,:] entrL
        double [:,:] detr_w
        double [:,:] detr_sc
        double [:,:] detrB
        double [:,:] detrL
        double [:] init_sc_upd
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
        Py_ssize_t wu_option
        double updraft_fraction
        double updraft_exponent
        double wu_min
        Py_ssize_t au_optL
        Py_ssize_t au_optB
        Py_ssize_t au_optB_wu
        Py_ssize_t au_optB_srf
        double au_optB1_frac
        double updraft_surface_height
        double tke_ed_coeff
        double w_entr_coeff
        double w_buoy_coeff
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
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals)
    cpdef compute_entrainment_detrainment(self)
    cpdef solve_updraft_velocity(self,  TimeStepping TS)
    cpdef solve_area_fraction(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)

    cpdef update_GMV_MF_implicitMF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED_implicitMF(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_diagnostic_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef compute_tke(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_tke_buoy(self, GridMeanVariables GMV)
    cpdef compute_tke_dissipation(self, TimeStepping TS)
    cpdef compute_tke_entr_detr(self)
    cpdef compute_tke_shear(self, GridMeanVariables GMV)
    cpdef update_tke_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_tke_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS)







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
    cpdef compute_entrainment_detrainment(self)
    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case)
    cpdef solve_updraft_velocity(self)
    cpdef solve_area_fraction(self, GridMeanVariables GMV)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)

    cpdef update_GMV_MF_implicitMF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED_implicitMF(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)


cdef class SimilarityED(ParameterizationBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS )
    cpdef update_inversion(self, GridMeanVariables GMV, option)


