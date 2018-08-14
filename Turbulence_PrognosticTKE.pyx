#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
cimport  EDMF_Updrafts
from Grid cimport Grid
cimport EDMF_Environment
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport SurfaceBase
from Cases cimport  CasesBase
from ReferenceState cimport  ReferenceState
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport  *
from turbulence_functions cimport *
from utility_functions cimport *
from libc.math cimport fmax, sqrt, exp, pow, cbrt, fmin
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class EDMF_PrognosticTKE(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self, paramlist,  Gr, Ref)
        # Set the number of updrafts (1)
        try:
            self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        except:
            self.n_updrafts = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft')
        try:
            self.use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
        except:
            self.use_steady_updrafts = False
        try:
            self.use_local_micro = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        except:
            self.use_local_micro = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to local (level-by-level) microphysics')

        try:
            self.use_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['use_scalar_var']
        except:
            self.use_scalar_var = False
            print('Defaulting to non-calculation of scalar variances')

        try:
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'dry':
                self.entr_detr_fp = entr_detr_dry
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_w':
                self.entr_detr_fp = entr_detr_inverse_w
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'b_w2':
                self.entr_detr_fp = entr_detr_b_w2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'buoyancy_sorting':
                self.entr_detr_fp = entr_detr_buoyancy_sorting
            else:
                print('Turbulence--EDMF_PrognosticTKE: Entrainment rate namelist option is not recognized')
        except:
            self.entr_detr_fp = entr_detr_b_w2
            print('Turbulence--EDMF_PrognosticTKE: defaulting to cloudy entrainment formulation')
        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False
            print('Turbulence--EDMF_PrognosticTKE: defaulting to TKE-based eddy diffusivity')

        try:
            self.extrapolate_buoyancy = namelist['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy']
        except:
            self.extrapolate_buoyancy = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to extrapolation of updraft buoyancy along a pseudoadiabat')


        # Get values from paramlist
        # set defaults at some point?
        self.surface_area = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
        self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']
        self.max_area_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        self.pressure_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        # "Legacy" coefficients used by the steady updraft routine
        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff

        # Need to code up as paramlist option?
        self.minimum_area = 1e-3

        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist,paramlist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = EDMF_Updrafts.UpdraftMicrophysics(paramlist, self.n_updrafts, Gr, Ref)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EDMF_Environment.EnvironmentVariables(namelist,Gr)
        # Create the class for environment thermodynamics
        self.EnvThermo = EDMF_Environment.EnvironmentThermodynamics(namelist, paramlist, Gr, Ref, self.EnvVar)

        # Entrainment rates
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        #self.press = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Pressure term in updraft vertical momentum equation
        self.updraft_pressure_sink = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # environmental tke source terms
        self.tke_buoy = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_dissipation = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_entr_gain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_detr_loss = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_shear = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_pressure = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        if self.use_scalar_var:
            #self.Hvar = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            #self.QTvar = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            #self.HQTcov = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.Hvar_dissipation = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.QTvar_dissipation = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.HQTcov_dissipation = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.Hvar_entr_gain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.QTvar_entr_gain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.HQTcov_entr_gain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.Hvar_detr_loss = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.QTvar_detr_loss = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.HQTcov_detr_loss = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.Hvar_shear = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.QTvar_shear = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.HQTcov_shear = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.Hvar_rain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.QTvar_rain = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            self.HQTcov_rain = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # Near-surface BC of updraft area fraction
        self.area_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')

       # Mass flux tendencies of mean scalars (for output)
        self.massflux_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')


       # (Eddy) diffusive tendencies of mean scalars (for output)
        self.diffusive_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')

       # Vertical fluxes for output
        self.massflux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_tke = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        return

    cpdef initialize(self, GridMeanVariables GMV):
        self.UpdVar.initialize(GMV)
        return

    # Initialize the IO pertaining to this class
    cpdef initialize_io(self, NetCDFIO_Stats Stats):

        self.UpdVar.initialize_io(Stats)
        self.EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('massflux')
        Stats.add_profile('massflux_h')
        Stats.add_profile('massflux_qt')
        Stats.add_profile('massflux_tendency_h')
        Stats.add_profile('massflux_tendency_qt')
        Stats.add_profile('diffusive_flux_h')
        Stats.add_profile('diffusive_flux_qt')
        Stats.add_profile('diffusive_tendency_h')
        Stats.add_profile('diffusive_tendency_qt')
        Stats.add_profile('total_flux_h')
        Stats.add_profile('total_flux_qt')
        Stats.add_profile('massflux_tke')
        Stats.add_profile('mixing_length')
        Stats.add_profile('tke_buoy')
        Stats.add_profile('tke_dissipation')
        Stats.add_profile('tke_entr_gain')
        Stats.add_profile('tke_detr_loss')
        Stats.add_profile('tke_shear')
        Stats.add_profile('tke_pressure')
        Stats.add_profile('updraft_qt_precip')
        Stats.add_profile('updraft_thetal_precip')

        if self.use_scalar_var:
            Stats.add_profile('Hvar_dissipation')
            Stats.add_profile('QTvar_dissipation')
            Stats.add_profile('HQTcov_dissipation')
            Stats.add_profile('Hvar_entr_gain')
            Stats.add_profile('QTvar_entr_gain')
            Stats.add_profile('HQTcov_entr_gain')
            Stats.add_profile('Hvar_detr_loss')
            Stats.add_profile('QTvar_detr_loss')
            Stats.add_profile('HQTcov_detr_loss')
            Stats.add_profile('Hvar_shear')
            Stats.add_profile('QTvar_shear')
            Stats.add_profile('HQTcov_shear')
            Stats.add_profile('Hvar_rain')
            Stats.add_profile('QTvar_rain')
            Stats.add_profile('HQTcov_rain')

        return

    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg-self.Gr.gw
            double [:] mean_entr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_detr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] massflux = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_h = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_qt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                mf_h[k] = interp2pt(self.massflux_h[k], self.massflux_h[k-1])
                mf_qt[k] = interp2pt(self.massflux_qt[k], self.massflux_qt[k-1])
                massflux[k] = interp2pt(self.m[0,k], self.m[0,k-1])
                if self.UpdVar.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        mean_entr_sc[k] += self.UpdVar.Area.values[i,k] * self.entr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_detr_sc[k] += self.UpdVar.Area.values[i,k] * self.detr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]

        Stats.write_profile('entrainment_sc', mean_entr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('detrainment_sc', mean_detr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux', massflux[self.Gr.gw:self.Gr.nzg-self.Gr.gw ])
        Stats.write_profile('massflux_h', mf_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_qt', mf_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_h', self.massflux_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_qt', self.massflux_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_flux_h', self.diffusive_flux_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_flux_qt', self.diffusive_flux_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_h', self.diffusive_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_qt', self.diffusive_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('total_flux_h', np.add(mf_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw],
                                                   self.diffusive_flux_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw]))
        Stats.write_profile('total_flux_qt', np.add(mf_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw],
                                                    self.diffusive_flux_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw]))
        Stats.write_profile('massflux_tke', self.massflux_tke[kmin-1:kmax-1])
        Stats.write_profile('mixing_length', self.mixing_length[kmin:kmax])
        Stats.write_profile('tke_buoy', self.tke_buoy[kmin:kmax])
        self.compute_tke_dissipation()
        Stats.write_profile('tke_dissipation', self.tke_dissipation[kmin:kmax])
        Stats.write_profile('tke_entr_gain', self.tke_entr_gain[kmin:kmax])
        self.compute_tke_detr()
        Stats.write_profile('tke_detr_loss', self.tke_detr_loss[kmin:kmax])
        Stats.write_profile('tke_shear', self.tke_shear[kmin:kmax])
        Stats.write_profile('tke_pressure', self.tke_pressure[kmin:kmax])
        Stats.write_profile('updraft_qt_precip', self.UpdMicro.prec_source_qt_tot[kmin:kmax])
        Stats.write_profile('updraft_thetal_precip', self.UpdMicro.prec_source_h_tot[kmin:kmax])

        if self.use_scalar_var:
            #Stats.write_profile('Hvar', self.Hvar[kmin:kmax])
            #Stats.write_profile('QTvar', self.QTvar[kmin:kmax])
            #Stats.write_profile('HQTcov', self.HQTcov[kmin:kmax])
            self.compute_covariance_dissipation()
            Stats.write_profile('Hvar_dissipation', self.Hvar_dissipation[kmin:kmax])
            Stats.write_profile('QTvar_dissipation', self.QTvar_dissipation[kmin:kmax])
            Stats.write_profile('HQTcov_dissipation', self.HQTcov_dissipation[kmin:kmax])
            Stats.write_profile('Hvar_entr_gain', self.Hvar_entr_gain[kmin:kmax])
            Stats.write_profile('QTvar_entr_gain', self.QTvar_entr_gain[kmin:kmax])
            Stats.write_profile('HQTcov_entr_gain', self.HQTcov_entr_gain[kmin:kmax])
            self.compute_covariance_detr()
            Stats.write_profile('Hvar_detr_loss', self.Hvar_detr_loss[kmin:kmax])
            Stats.write_profile('QTvar_detr_loss', self.QTvar_detr_loss[kmin:kmax])
            Stats.write_profile('HQTcov_detr_loss', self.HQTcov_detr_loss[kmin:kmax])
            Stats.write_profile('Hvar_shear', self.Hvar_shear[kmin:kmax])
            Stats.write_profile('QTvar_shear', self.QTvar_shear[kmin:kmax])
            Stats.write_profile('HQTcov_shear', self.HQTcov_shear[kmin:kmax])
            Stats.write_profile('Hvar_rain', self.Hvar_rain[kmin:kmax])
            Stats.write_profile('QTvar_rain', self.QTvar_rain[kmin:kmax])
            Stats.write_profile('HQTcov_rain', self.HQTcov_rain[kmin:kmax])

        return



    # Perform the update of the scheme

    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg - self.Gr.gw

        self.update_inversion(GMV, Case.inversion_option)

        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        if TS.nstep == 0:
            self.initialize_tke(GMV, Case)
            if self.use_scalar_var:
                self.initialize_covariance(GMV, Case)
            with nogil:
                for k in xrange(self.Gr.nzg):
                    self.EnvVar.TKE.values[k] = GMV.TKE.values[k]
                    if self.use_scalar_var:
                        self.EnvVar.Hvar.values[k] = GMV.Hvar.values[k]
                        self.EnvVar.QTvar.values[k] = GMV.QTvar.values[k]
                        self.EnvVar.HQTcov.values[k] = GMV.HQTcov.values[k]

        self.decompose_environment(GMV, 'values')

        if self.use_steady_updrafts:
            self.compute_diagnostic_updrafts(GMV, Case)
        else:
            self.compute_prognostic_updrafts(GMV, Case, TS)

        # TODO -maybe not needed? - both diagnostic and prognostic updrafts end with decompose_environment
        # But in general ok here without thermodynamics because MF doesnt depend directly on buoyancy
        self.decompose_environment(GMV, 'values')

        self.update_GMV_MF(GMV, TS)
        # (###)
        # decompose_environment +  EnvThermo.satadjust + UpdThermo.buoyancy should always be used together
        # This ensures that:
        #   - the buoyancy of updrafts and environment is up to date with the most recent decomposition,
        #   - the buoyancy of updrafts and environment is updated such that
        #     the mean buoyancy with repect to reference state alpha_0 is zero.
        self.decompose_environment(GMV, 'mf_update')
        self.EnvThermo.satadjust(self.EnvVar, True)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.compute_eddy_diffusivities_tke(GMV, Case)

        self.update_GMV_ED(GMV, Case, TS)
        self.compute_tke(GMV, Case, TS)
        if self.use_scalar_var:
            self.compute_covariance(GMV, Case, TS)

        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)

        return

    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        cdef:
            Py_ssize_t iter_
            double time_elapsed = 0.0

        self.UpdVar.set_new_with_values()
        self.UpdVar.set_old_with_values()
        self.set_updraft_surface_bc(GMV, Case)
        self.dt_upd = np.minimum(TS.dt, 0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            self.compute_entrainment_detrainment(GMV, Case)
            self.solve_updraft_velocity_area(GMV,TS)
            self.solve_updraft_scalars(GMV, Case, TS)
            self.UpdVar.set_values_with_new()
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
            # (####)
            # TODO - see comment (###)
            # It would be better to have a simple linear rule for updating environment here
            # instead of calling EnvThermo saturation adjustment scheme for every updraft.
            # If we are using quadratures this is expensive and probably unnecessary.
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.satadjust(self.EnvVar, False)
            self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)
        return

    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz
            double dzi = self.Gr.dzi
            eos_struct sa
            entr_struct ret
            entr_in_struct input
            double a,b,c, w, w_km,  w_mid, w_low, denom, arg
            double entr_w, detr_w, B_k, area_k, w2

        self.set_updraft_surface_bc(GMV, Case)
        self.compute_entrainment_detrainment(GMV, Case)


        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.values[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.values[i,gw] = self.qt_surface_bc[i]
                # Find the cloud liquid content
                sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[gw],
                         self.UpdVar.QT.values[i,gw], self.UpdVar.H.values[i,gw])
                self.UpdVar.QL.values[i,gw] = sa.ql
                self.UpdVar.T.values[i,gw] = sa.T
                self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[gw], self.UpdVar.T.values[i,gw],
                                                                   &self.UpdVar.QT.values[i,gw], &self.UpdVar.QL.values[i,gw],
                                                                   &self.UpdVar.QR.values[i,gw], &self.UpdVar.H.values[i,gw],
                                                                   i, gw)
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    denom = 1.0 + self.entr_sc[i,k] * dz
                    self.UpdVar.H.values[i,k] = (self.UpdVar.H.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.H.values[k])/denom
                    self.UpdVar.QT.values[i,k] = (self.UpdVar.QT.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.QT.values[k])/denom


                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                    self.UpdVar.QL.values[i,k] = sa.ql
                    self.UpdVar.T.values[i,k] = sa.T
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[k], self.UpdVar.T.values[i,k],
                                                                       &self.UpdVar.QT.values[i,k], &self.UpdVar.QL.values[i,k],
                                                                       &self.UpdVar.QR.values[i,k], &self.UpdVar.H.values[i,k],
                                                                       i, k)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.QR.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)
        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        # Solve updraft velocity equation
        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.W.values[i, self.Gr.gw-1] = self.w_surface_bc[i]
                self.entr_sc[i,gw] = 2.0 /dz
                self.detr_sc[i,gw] = 0.0
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    area_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    if area_k >= self.minimum_area:
                        w_km = self.UpdVar.W.values[i,k-1]
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        w2 = ((self.vel_buoy_coeff * B_k + 0.5 * w_km * w_km * dzi)
                              /(0.5 * dzi +entr_w + self.vel_pressure_coeff/sqrt(fmax(area_k,self.minimum_area))))
                        if w2 > 0.0:
                            self.UpdVar.W.values[i,k] = sqrt(w2)
                        else:
                            self.UpdVar.W.values[i,k:] = 0
                            break
                    else:
                        self.UpdVar.W.values[i,k:] = 0




        self.UpdVar.W.set_bcs(self.Gr)

        cdef double au_lim
        with nogil:
            for i in xrange(self.n_updrafts):
                au_lim = self.max_area_factor * self.area_surface_bc[i]
                self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                w_mid = 0.5* (self.UpdVar.W.values[i,gw])
                for k in xrange(gw+1, self.Gr.nzg):
                    w_low = w_mid
                    w_mid = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                    if w_mid > 0.0:
                        if self.entr_sc[i,k]>(0.9/dz):
                            self.entr_sc[i,k] = 0.9/dz

                        self.UpdVar.Area.values[i,k] = (self.Ref.rho0_half[k-1]*self.UpdVar.Area.values[i,k-1]*w_low/
                                                        (1.0-(self.entr_sc[i,k]-self.detr_sc[i,k])*dz)/w_mid/self.Ref.rho0_half[k])
                        # # Limit the increase in updraft area when the updraft decelerates
                        if self.UpdVar.Area.values[i,k] >  au_lim:
                            self.UpdVar.Area.values[i,k] = au_lim
                            self.detr_sc[i,k] =(self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1]
                                                * w_low / au_lim / w_mid / self.Ref.rho0_half[k] + self.entr_sc[i,k] * dz -1.0)/dz
                    else:
                        # the updraft has terminated so set its area fraction to zero at this height and all heights above
                        self.UpdVar.Area.values[i,k] = 0.0
                        self.UpdVar.H.values[i,k] = GMV.H.values[k]
                        self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                        self.UpdVar.QR.values[i,k] = GMV.QR.values[k]
                        #TODO wouldnt it be more consistent to have here?
                        #self.UpdVar.QL.values[i,k] = GMV.QL.values[k]
                        #self.UpdVar.T.values[i,k] = GMV.T.values[k]
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                        self.UpdVar.QL.values[i,k] = sa.ql
                        self.UpdVar.T.values[i,k] = sa.T

        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.UpdVar.Area.set_bcs(self.Gr)

        self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h, self.UpdVar.Area.values), axis=0)
        self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt, self.UpdVar.Area.values), axis=0)

        return

    cpdef compute_tke(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed D_T
            self.compute_mixing_length(Case.Sur.obukhov_length)

        self.compute_tke_buoy(GMV)
        self.compute_tke_entr()
        self.compute_tke_shear(GMV)
        self.compute_tke_pressure()

        self.reset_surface_tke(GMV, Case)
        self.update_tke_ED(GMV, Case, TS)


        return

    cpdef initialize_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            double ws= self.wstar, us = Case.Sur.ustar, zs = self.zi, z
        # Similarity profile initialization of TKE
        # Need to consider what to do when neutral/stable
        if ws > 0.0:
            with nogil:
                for k in xrange(self.Gr.nzg):
                    z = self.Gr.z_half[k]
                    GMV.TKE.values[k] = ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))


        self.reset_surface_tke(GMV, Case)
        self.compute_mixing_length(Case.Sur.obukhov_length)
        return

    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    cpdef compute_mixing_length(self, double obukhov_length):

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double tau =  get_mixing_tau(self.zi, self.wstar)
            double l1, l2, z_
            double grad, grad2, H


        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                l1 = tau * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                z_ = self.Gr.z_half[k]
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_
                self.mixing_length[k] = fmax( 1.0/(1.0/fmax(l1,1e-10) + 1.0/l2), 1e-3)
        return

    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double lm
            double we_half

        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)
        else:
            self.compute_mixing_length(Case.Sur.obukhov_length)
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    lm = self.mixing_length[k]
                    self.KM.values[k] = self.tke_ed_coeff * lm * sqrt(fmax(self.EnvVar.TKE.values[k],0.0) )
                    self.KH.values[k] = self.KM.values[k] / self.prandtl_number

        return

    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case):

        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        cdef:
            Py_ssize_t i, gw = self.Gr.gw
            double zLL = self.Gr.z_half[gw]
            double ustar = Case.Sur.ustar, oblength = Case.Sur.obukhov_length
            double alpha0LL  = self.Ref.alpha0_half[gw]
            double qt_var = get_surface_variance(Case.Sur.rho_qtflux*alpha0LL,
                                                 Case.Sur.rho_qtflux*alpha0LL, ustar, zLL, oblength)
            double h_var = get_surface_variance(Case.Sur.rho_hflux*alpha0LL,
                                                 Case.Sur.rho_hflux*alpha0LL, ustar, zLL, oblength)

            double a_ = self.surface_area/self.n_updrafts
            double surface_scalar_coeff

        # with nogil:
        for i in xrange(self.n_updrafts):
            surface_scalar_coeff= percentile_bounds_mean_norm(1.0-self.surface_area+i*a_,
                                                                   1.0-self.surface_area + (i+1)*a_ , 1000)

            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.h_surface_bc[i] = (GMV.H.values[gw] + surface_scalar_coeff * sqrt(h_var))
            self.qt_surface_bc[i] = (GMV.QT.values[gw] + surface_scalar_coeff * sqrt(qt_var))
        return

    cpdef reset_surface_tke(self, GridMeanVariables GMV, CasesBase Case):
        GMV.TKE.values[self.Gr.gw] = get_surface_tke(Case.Sur.ustar,
                                                     self.wstar,
                                                     self.Gr.z_half[self.Gr.gw],
                                                     Case.Sur.obukhov_length)
        return


    cpdef reset_surface_covariance(self, GridMeanVariables GMV, CasesBase Case):

        cdef:
            double zLL = self.Gr.z_half[self.Gr.gw]
            double ustar = Case.Sur.ustar, oblength = Case.Sur.obukhov_length
            double flux1 = Case.Sur.rho_hflux * self.Ref.alpha0_half[self.Gr.gw]
            double flux2 = Case.Sur.rho_qtflux * self.Ref.alpha0_half[self.Gr.gw]

        GMV.Hvar.values[self.Gr.gw] =  get_surface_variance(flux1, flux1, ustar, zLL, oblength)
        GMV.QTvar.values[self.Gr.gw] = get_surface_variance(flux2, flux2, ustar, zLL, oblength)
        GMV.HQTcov.values[self.Gr.gw] = get_surface_variance(flux1,flux2, ustar, zLL, oblength)

        return


    # Find values of environmental variables by subtracting updraft values from grid mean values
    # whichvals used to check which substep we are on--correspondingly use 'GMV.SomeVar.value' (last timestep value)
    # or GMV.SomeVar.mf_update (GMV value following massflux substep)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals):

        # first make sure the 'bulkvalues' of the updraft variables are updated
        self.UpdVar.set_means(GMV)

        cdef:
            Py_ssize_t k, gw = self.Gr.gw
            double val1, val2, au_full
            double Hvar_e, QTvar_e, HQTcov_e, Hvar_u, QTvar_u, HQTcov_u
        if whichvals == 'values':

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1
                    self.EnvVar.QT.values[k] = val1 * GMV.QT.values[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.values[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W--interpolate area fraction to the "full" grid points
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

            self.get_GMV_TKE(self.UpdVar.Area,self.UpdVar.W, self.EnvVar.W, self.EnvVar.TKE,
                             &GMV.W.values[0], &GMV.TKE.values[0])

            if self.use_scalar_var:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar,
                                 &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,
                                 &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov,
                                 &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])


        elif whichvals == 'mf_update':
            # same as above but replace GMV.SomeVar.values with GMV.SomeVar.mf_update

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1

                    self.EnvVar.QT.values[k] = val1 * GMV.QT.mf_update[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.mf_update[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

            self.get_GMV_TKE(self.UpdVar.Area,self.UpdVar.W, self.EnvVar.W, self.EnvVar.TKE,
                             &GMV.W.values[0], &GMV.TKE.values[0])

            if self.use_scalar_var:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar,
                                 &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,
                                 &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov,
                                 &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])

        return

    cdef get_GMV_TKE(self, EDMF_Updrafts.UpdraftVariable au, EDMF_Updrafts.UpdraftVariable wu,
                      EDMF_Environment.EnvironmentVariable we, EDMF_Environment.EnvironmentVariable tke_e,
                      double *gmv_w, double *gmv_tke):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double interp_w_diff

        with nogil:
            for k in xrange(self.Gr.nzg):
                interp_w_diff = interp2pt(we.values[k-1]-gmv_w[k-1],we.values[k]-gmv_w[k])
                gmv_tke[k] = ae[k] * interp_w_diff * interp_w_diff + ae[k] * tke_e.values[k]
                for i in xrange(self.n_updrafts):
                    interp_w_diff = interp2pt(wu.values[i,k-1]-gmv_w[k-1],wu.values[i,k]-gmv_w[k])
                    gmv_tke[k] += au.values[i,k] *interp_w_diff * interp_w_diff
        return



    cdef get_env_tke_from_GMV(self, EDMF_Updrafts.UpdraftVariable au, EDMF_Updrafts.UpdraftVariable wu,
                      EDMF_Environment.EnvironmentVariable we, EDMF_Environment.EnvironmentVariable tke_e,
                      double *gmv_w, double *gmv_tke):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double interp_w_diff

        with nogil:
            for k in xrange(self.Gr.nzg):
                if ae[k] > 0.0:
                    interp_w_diff = interp2pt(we.values[k-1]-gmv_w[k-1],we.values[k]-gmv_w[k])
                    tke_e.values[k] = gmv_tke[k] - ae[k] * interp_w_diff * interp_w_diff

                    for i in xrange(self.n_updrafts):
                        interp_w_diff = interp2pt(wu.values[i,k-1]-gmv_w[k-1],wu.values[i,k]-gmv_w[k])
                        tke_e.values[k] -= au.values[i,k] *interp_w_diff * interp_w_diff
                    tke_e.values[k] = tke_e.values[k]/ae[k]
                else:
                    tke_e.values[k] = 0.0
        return


    # Note: this assumes all variables are defined on half levels not full levels (i.e. phi, psi are not w)
    cdef get_GMV_CoVar(self, EDMF_Updrafts.UpdraftVariable au,
                        EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e,
                        EDMF_Environment.EnvironmentVariable covar_e,
                       double *gmv_phi, double *gmv_psi, double *gmv_covar):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double phi_diff, psi_diff

        with nogil:
            for k in xrange(self.Gr.nzg):
                phi_diff = phi_e.values[k]-gmv_phi[k]
                psi_diff = psi_e.values[k]-gmv_psi[k]
                gmv_covar[k] = ae[k] * phi_diff * psi_diff + ae[k] * covar_e.values[k]
                for i in xrange(self.n_updrafts):
                    phi_diff = phi_u.values[i,k]-gmv_phi[k]
                    psi_diff = psi_u.values[i,k]-gmv_psi[k]
                    gmv_covar[k] += au.values[i,k] * phi_diff * psi_diff
        return



    cdef get_env_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Environment.EnvironmentVariable covar_e,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double phi_diff, psi_diff

        with nogil:
            for k in xrange(self.Gr.nzg):
                if ae[k] > 0.0:
                    phi_diff = phi_e.values[k]-gmv_phi[k]
                    psi_diff = psi_e.values[k] - gmv_psi[k]
                    covar_e.values[k] = gmv_covar[k] - ae[k] * phi_diff * psi_diff
                    for i in xrange(self.n_updrafts):
                        phi_diff = phi_u.values[i,k]-gmv_phi[k]
                        psi_diff = psi_u.values[i,k] - gmv_psi[k]
                        covar_e.values[k] -= au.values[i,k] * phi_diff * psi_diff
                    covar_e.values[k] = covar_e.values[k]/ae[k]
                else:
                    covar_e.values[k] = 0.0
        return




    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            entr_struct ret
            entr_in_struct input
            eos_struct sa


        self.UpdVar.get_cloud_base_top_cover()

        input.wstar = self.wstar

        with nogil:
            for i in xrange(self.n_updrafts):
                input.zi = self.UpdVar.cloud_base[i]
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    input.b = self.UpdVar.B.values[i,k]
                    input.w = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                    input.z = self.Gr.z_half[k]
                    input.af = self.UpdVar.Area.values[i,k]
                    input.tke = self.EnvVar.TKE.values[k]
                    input.ml = self.mixing_length[k]
                    input.qt_env = self.EnvVar.QT.values[k]
                    input.ql_env = self.EnvVar.QL.values[k]
                    input.H_env = self.EnvVar.H.values[k]
                    input.b_env = self.EnvVar.B.values[k]
                    input.w_env = self.EnvVar.W.values[k]
                    input.H_up = self.UpdVar.H.values[i,k]
                    input.qt_up = self.UpdVar.QT.values[i,k]
                    input.ql_up = self.UpdVar.QL.values[i,k]
                    input.p0 = self.Ref.p0_half[k]
                    input.alpha0 = self.Ref.alpha0_half[k]
                    input.tke_ed_coeff  = self.tke_ed_coeff
                    input.T_mean = (self.EnvVar.T.values[k]+self.UpdVar.T.values[i,k])/2
                    input.L = 20000.0 # need to define the scale of the GCM grid resolution
                    ret = self.entr_detr_fp(input)
                    self.entr_sc[i,k] = ret.entr_sc * self.entrainment_factor
                    self.detr_sc[i,k] = ret.detr_sc * self.detrainment_factor

        return

    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dzi = self.Gr.dzi
            double dti_ = 1.0/self.dt_upd
            double dt_ = 1.0/dti_
            double whalf_kp, whalf_k
            double a1, a2 # groupings of terms in area fraction discrete equation
            double au_lim
            double anew_k, a_k, a_km, entr_w, detr_w, B_k, entr_term, detr_term, rho_ratio
            double adv, buoy, exch, press, press_buoy, press_drag # groupings of terms in velocity discrete equation

        with nogil:
            for i in xrange(self.n_updrafts):
                self.entr_sc[i,gw] = 2.0 * dzi
                self.detr_sc[i,gw] = 0.0
                self.UpdVar.W.new[i,gw-1] = self.w_surface_bc[i]
                self.UpdVar.Area.new[i,gw] = self.area_surface_bc[i]
                au_lim = self.area_surface_bc[i] * self.max_area_factor

                for k in range(gw, self.Gr.nzg-gw):

                    # First solve for updated area fraction at k+1
                    whalf_kp = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k+1])
                    whalf_k = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    adv = -self.Ref.alpha0_half[k+1] * dzi *( self.Ref.rho0_half[k+1] * self.UpdVar.Area.values[i,k+1] * whalf_kp
                                                              -self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * whalf_k)
                    entr_term = self.UpdVar.Area.values[i,k+1] * whalf_kp * (self.entr_sc[i,k+1] )
                    detr_term = self.UpdVar.Area.values[i,k+1] * whalf_kp * (- self.detr_sc[i,k+1])


                    self.UpdVar.Area.new[i,k+1]  = fmax(dt_ * (adv + entr_term + detr_term) + self.UpdVar.Area.values[i,k+1], 0.0)
                    if self.UpdVar.Area.new[i,k+1] > au_lim:
                        self.UpdVar.Area.new[i,k+1] = au_lim
                        if self.UpdVar.Area.values[i,k+1] > 0.0:
                            self.detr_sc[i,k+1] = (((au_lim-self.UpdVar.Area.values[i,k+1])* dti_ - adv -entr_term)/(-self.UpdVar.Area.values[i,k+1]  * whalf_kp))
                        else:
                            # this detrainment rate won't affect scalars but would affect velocity
                            self.detr_sc[i,k+1] = (((au_lim-self.UpdVar.Area.values[i,k+1])* dti_ - adv -entr_term)/(-au_lim  * whalf_kp))

                    # Now solve for updraft velocity at k
                    rho_ratio = self.Ref.rho0[k-1]/self.Ref.rho0[k]
                    anew_k = interp2pt(self.UpdVar.Area.new[i,k], self.UpdVar.Area.new[i,k+1])
                    if anew_k >= self.minimum_area:
                        a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                        a_km = interp2pt(self.UpdVar.Area.values[i,k-1], self.UpdVar.Area.values[i,k])
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        adv = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * self.UpdVar.W.values[i,k] * dzi
                               - self.Ref.rho0[k-1] * a_km * self.UpdVar.W.values[i,k-1] * self.UpdVar.W.values[i,k-1] * dzi)
                        exch = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k]
                                * (entr_w * self.EnvVar.W.values[k] - detr_w * self.UpdVar.W.values[i,k] ))
                        buoy= self.Ref.rho0[k] * a_k * B_k
                        press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                        press_drag = -1.0 * self.Ref.rho0[k] * a_k * (self.pressure_drag_coeff/self.pressure_plume_spacing
                                                                     * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])**2.0/sqrt(fmax(a_k,self.minimum_area)))
                        press = press_buoy + press_drag
                        self.updraft_pressure_sink[i,k] = press
                        self.UpdVar.W.new[i,k] = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * dti_
                                                  -adv + exch + buoy + press)/(self.Ref.rho0[k] * anew_k * dti_)

                        if self.UpdVar.W.new[i,k] <= 0.0:
                            self.UpdVar.W.new[i,k:] = 0.0
                            self.UpdVar.Area.new[i,k+1:] = 0.0
                            break
                    else:
                        self.UpdVar.W.new[i,k:] = 0.0
                        self.UpdVar.Area.new[i,k+1:] = 0.0
                        # keep this in mind if we modify updraft top treatment!
                        self.updraft_pressure_sink[i,k:] = 0.0
                        break
                    # the above lines were replaced by the followings to allow integration above negative w
                    # the model output is sensitive to the choice of value inthe condition : <= 0.01
                    #     if self.UpdVar.W.new[i,k] <= 0.01:
                    #         self.UpdVar.W.new[i,k] = 0.0
                    #         self.UpdVar.Area.new[i,k+1] = 0.0
                    #         #break
                    # else:
                    #     self.UpdVar.W.new[i,k] = 0.0
                    #     self.UpdVar.Area.new[i,k+1] = 0.0
                    #     #break
        # plt.figure('area')
        # plt.plot(self.UpdVar.Area.new[0,:], self.Gr.z_half)
        # plt.show()

        return

    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            double dzi = self.Gr.dzi
            double dti_ = 1.0/self.dt_upd
            double m_k, m_km
            Py_ssize_t gw = self.Gr.gw
            double H_entr, QT_entr
            double c1, c2, c3, c4
            eos_struct sa
            double qt_var, h_var

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.new[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.new[i,gw] = self.qt_surface_bc[i]
                self.UpdVar.QR.new[i,gw] = 0.0 #TODO

                if self.use_local_micro:
                    # do saturation adjustment
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp,
                             self.Ref.p0_half[gw], self.UpdVar.QT.new[i,gw], self.UpdVar.H.new[i,gw])
                    self.UpdVar.QL.new[i,gw] = sa.ql
                    self.UpdVar.T.new[i,gw] = sa.T
                    # remove precipitation (update QT, QL and H)
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[gw], self.UpdVar.T.new[i,gw],
                                                                       &self.UpdVar.QT.new[i,gw], &self.UpdVar.QL.new[i,gw],
                                                                       &self.UpdVar.QR.new[i,gw], &self.UpdVar.H.new[i,gw],
                                                                       i, gw)

                # starting from the bottom do entrainment at each level
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    H_entr = self.EnvVar.H.values[k]
                    QT_entr = self.EnvVar.QT.values[k]

                    # write the discrete equations in form:
                    # c1 * phi_new[k] = c2 * phi[k] + c3 * phi[k-1] + c4 * phi_entr
                    if self.UpdVar.Area.new[i,k] >= self.minimum_area:
                        m_k = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k]
                               * interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k]))
                        m_km = (self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1]
                               * interp2pt(self.UpdVar.W.values[i,k-2], self.UpdVar.W.values[i,k-1]))
                        c1 = self.Ref.rho0_half[k] * self.UpdVar.Area.new[i,k] * dti_
                        c2 = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * dti_
                              - m_k * (dzi + self.detr_sc[i,k]))
                        c3 = m_km * dzi
                        c4 = m_k * self.entr_sc[i,k]

                        self.UpdVar.H.new[i,k] =  (c2 * self.UpdVar.H.values[i,k]  + c3 * self.UpdVar.H.values[i,k-1]
                                                   + c4 * H_entr)/c1
                        self.UpdVar.QT.new[i,k] = (c2 * self.UpdVar.QT.values[i,k] + c3 * self.UpdVar.QT.values[i,k-1]
                                                   + c4* QT_entr)/c1
                    else:
                        self.UpdVar.H.new[i,k] = GMV.H.values[k]
                        self.UpdVar.QT.new[i,k] = GMV.QT.values[k]

                    # find new temperature
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.new[i,k], self.UpdVar.H.new[i,k])
                    self.UpdVar.QL.new[i,k] = sa.ql
                    self.UpdVar.T.new[i,k] = sa.T

                    if self.use_local_micro:
                        # remove precipitation (pdate QT, QL and H)
                        self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[k], self.UpdVar.T.new[i,k],
                                                                       &self.UpdVar.QT.new[i,k], &self.UpdVar.QL.new[i,k],
                                                                       &self.UpdVar.QR.new[i,k], &self.UpdVar.H.new[i,k],
                                                                       i, k)

        if self.use_local_micro:
            # save the total source terms for H and QT due to precipitation
            # TODO - add QR source
            self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h,
                                                                 self.UpdVar.Area.values), axis=0)
            self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt,
                                                                  self.UpdVar.Area.values), axis=0)
        else:
            # Compute the updraft microphysical sources (precipitation)
            #after the entrainment loop is finished
            self.UpdMicro.compute_sources(self.UpdVar)
            # Update updraft variables with microphysical source tendencies
            self.UpdMicro.update_updraftvars(self.UpdVar)

        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.QR.set_bcs(self.Gr)
        return

    # After updating the updraft variables themselves:
    # 1. compute the mass fluxes (currently not stored as class members, probably will want to do this
    # for output purposes)
    # 2. Apply mass flux tendencies and updraft microphysical tendencies to GMV.SomeVar.Values (old time step values)
    # thereby updating to GMV.SomeVar.mf_update
    # mass flux tendency is computed as 1st order upwind

    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t gw = self.Gr.gw
            double mf_tend_h=0.0, mf_tend_qt=0.0
            double env_h_interp, env_qt_interp
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        with nogil:
            for i in xrange(self.n_updrafts):
                self.m[i,gw-1] = 0.0
                for k in xrange(self.Gr.gw, self.Gr.nzg-1):
                    self.m[i,k] = ((self.UpdVar.W.values[i,k] - self.EnvVar.W.values[k] )* self.Ref.rho0[k]
                                   * interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1]))

        self.massflux_h[gw-1] = 0.0
        self.massflux_qt[gw-1] = 0.0
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                self.massflux_h[k] = 0.0
                self.massflux_qt[k] = 0.0
                env_h_interp = interp2pt(self.EnvVar.H.values[k], self.EnvVar.H.values[k+1])
                env_qt_interp = interp2pt(self.EnvVar.QT.values[k], self.EnvVar.QT.values[k+1])
                for i in xrange(self.n_updrafts):
                    self.massflux_h[k] += self.m[i,k] * (interp2pt(self.UpdVar.H.values[i,k],
                                                                   self.UpdVar.H.values[i,k+1]) - env_h_interp )
                    self.massflux_qt[k] += self.m[i,k] * (interp2pt(self.UpdVar.QT.values[i,k],
                                                                    self.UpdVar.QT.values[i,k+1]) - env_qt_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variables
        with nogil:

            for k in xrange(self.Gr.gw, self.Gr.nzg):
                mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
                mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdMicro.prec_source_h_tot[k]
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k]

                #No mass flux tendency for U, V
                GMV.U.mf_update[k] = GMV.U.values[k]
                GMV.V.mf_update[k] = GMV.V.values[k]
                # Prepare the output
                self.massflux_tendency_h[k] = mf_tend_h
                self.massflux_tendency_qt[k] = mf_tend_qt


        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)
        GMV.QR.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

        return

    # Update the grid mean variables with the tendency due to eddy diffusion
    # Km and Kh have already been updated
    # 2nd order finite differences plus implicit time step allows solution with tridiagonal matrix solver
    # Update from GMV.SomeVar.mf_update to GMV.SomeVar.new
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double dzi = self.Gr.dzi
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] =  self.EnvVar.QT.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.QT.new[k+gw] = GMV.QT.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])
            # get the diffusive flux
            self.diffusive_flux_qt[gw] = interp2pt(Case.Sur.rho_qtflux, -rho_ae_K_m[gw] * dzi *(self.EnvVar.QT.values[gw+1]-self.EnvVar.QT.values[gw]) )
            for k in xrange(self.Gr.gw+1, self.Gr.nzg-self.Gr.gw):
                self.diffusive_flux_qt[k] = -0.5 * self.Ref.rho0_half[k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.QT.values[k+1]-self.EnvVar.QT.values[k-1])

        # Solve H
        with nogil:
            for k in xrange(nz):
                x[k] = self.EnvVar.H.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.H.new[k+gw] = GMV.H.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])
                self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
            # get the diffusive flux
            self.diffusive_flux_h[gw] = interp2pt(Case.Sur.rho_hflux, -rho_ae_K_m[gw] * dzi *(self.EnvVar.H.values[gw+1]-self.EnvVar.H.values[gw]) )
            for k in xrange(self.Gr.gw+1, self.Gr.nzg-self.Gr.gw):
                self.diffusive_flux_h[k] = -0.5 * self.Ref.rho0_half[k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.H.values[k+1]-self.EnvVar.H.values[k-1])

        # Solve U
        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.U.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_uflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.U.new[k+gw] = x[k]

        # Solve V
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.V.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_vflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.V.new[k+gw] = x[k]

        GMV.QT.set_bcs(self.Gr)
        GMV.QR.set_bcs(self.Gr)
        GMV.H.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

        return

    cpdef compute_tke_buoy(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double d_alpha_thetal_dry, d_alpha_qt_dry
            double d_alpha_thetal_cloudy, d_alpha_qt_cloudy
            double d_alpha_thetal_total, d_alpha_qt_total
            double lh, prefactor, cpm
            double qt_dry, th_dry, t_cloudy, qv_cloudy, qt_cloudy, th_cloudy
            double grad_thl_minus=0.0, grad_qt_minus=0.0, grad_thl_plus=0, grad_qt_plus=0
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        # Note that source terms at the gw grid point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]

                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_minus = grad_thl_plus
                grad_qt_minus = grad_qt_plus
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi

                prefactor = Rd * exner_c(self.Ref.p0_half[k])/self.Ref.p0_half[k]

                d_alpha_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
                d_alpha_qt_dry = prefactor * th_dry * (eps_vi-1.0)

                if self.EnvVar.CF.values[k] > 0.0:
                    d_alpha_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_alpha_qt_cloudy = (lh / cpm / t_cloudy * d_alpha_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_alpha_thetal_cloudy = 0.0
                    d_alpha_qt_cloudy = 0.0

                d_alpha_thetal_total = (self.EnvVar.CF.values[k] * d_alpha_thetal_cloudy
                                        + (1.0-self.EnvVar.CF.values[k]) * d_alpha_thetal_dry)
                d_alpha_qt_total = (self.EnvVar.CF.values[k] * d_alpha_qt_cloudy
                                    + (1.0-self.EnvVar.CF.values[k]) * d_alpha_qt_dry)

                # TODO - check
                self.tke_buoy[k] = g / self.Ref.alpha0_half[k] * ae[k] * self.Ref.rho0_half[k] \
                                   * ( \
                                       - self.KH.values[k] * interp2pt(grad_thl_plus, grad_thl_minus) * d_alpha_thetal_total \
                                       - self.KH.values[k] * interp2pt(grad_qt_plus,  grad_qt_minus)  * d_alpha_qt_total\
                                     )
        return


    # Note we need mixing length again here....
    cpdef compute_tke_dissipation(self):
        cdef:
            Py_ssize_t k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
            #tke diss coeff

        # does mixing length need to be recomputed? Even if no change in GMV.TKE, if updraft area changes
        # this would change environmental tke (assuming it is still adjusted based on tke)
        # first pass...assume we can re-use
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_dissipation[k] = (self.Ref.rho0_half[k] * ae[k] * pow(fmax(self.EnvVar.TKE.values[k],0), 1.5)
                                           /fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
        return

    # Note updrafts' (total) entrainment rate = environment's (total) detrainment rate
    # and updrafts' (total) detrainment rate = environment's (total) entrainment rate
    # Therefore, fractional xxtrainment rates must be properly multiplied by the right component mass fluxes
    # Here we use the terminology entrainment/detrainment relative to the __environment__
    cpdef compute_tke_entr(self):
        cdef:
            Py_ssize_t i, k
            double w_u, w_e

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_entr_gain[k] = 0.0
                w_e = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                for i in xrange(self.n_updrafts):
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    self.tke_entr_gain[k] +=   self.UpdVar.Area.values[i,k] * w_u * self.detr_sc[i,k] * (w_u - w_e) * (w_u - w_e)
                self.tke_entr_gain[k] *= 0.5 * self.Ref.rho0_half[k]
        return


    cpdef compute_tke_detr(self):
        cdef:
            Py_ssize_t i, k
            double w_u

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_detr_loss[k] = 0.0
                for i in xrange(self.n_updrafts):
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    self.tke_detr_loss[k] += self.UpdVar.Area.values[i,k] * w_u * self.entr_sc[i,k]
                self.tke_detr_loss[k] *= self.Ref.rho0_half[k] * self.EnvVar.TKE.values[k]
        return

    cpdef compute_tke_shear(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double du_high = 0.0
            double dv_high = 0.0
            double dw_high = 2.0 * self.EnvVar.W.values[gw]  * self.Gr.dzi
            double du_low, dv_low, dw_low,

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                du_low = du_high
                dv_low = dv_high
                dw_low = dw_high
                du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi
                dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi
                dw_high = (self.EnvVar.W.values[k+1] - self.EnvVar.W.values[k]) * self.Gr.dzi
                self.tke_shear[k] =( self.Ref.rho0_half[k] * ae[k] * self.KM.values[k] *
                                    ( pow(interp2pt(du_low, du_high),2.0) +  pow(interp2pt(dv_low, dv_high),2.0)
                                      + pow(interp2pt(dw_low, dw_high),2.0)))
        return

    cpdef compute_tke_pressure(self):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double wu_half, we_half
            double press_buoy, press_drag

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.tke_pressure[k] = 0.0
                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    we_half = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                    press_buoy= (-1.0 * self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k]
                                 * self.UpdVar.B.values[i,k] * self.pressure_buoy_coeff)
                    press_drag = (-1.0 * self.Ref.rho0_half[k] * sqrt(self.UpdVar.Area.values[i,k])
                                  * (self.pressure_drag_coeff/self.pressure_plume_spacing* (wu_half -we_half)**2.0))
                    self.tke_pressure[k] += (we_half - wu_half) * (press_buoy + press_drag)
        return

    cpdef update_tke_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS):
        cdef:
            Py_ssize_t k, kk, i
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double dzi = self.Gr.dzi
            double dti = TS.dti
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] ae_old = np.subtract(np.ones((nzg,),dtype=np.double, order='c'), np.sum(self.UpdVar.Area.old,axis=0))
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] whalf = np.zeros((nzg,),dtype=np.double, order='c')
            double  D_env = 0.0
            double wu_half, we_half,a_half, tke_0_surf,press_k



        with nogil:
            for k in xrange(1,nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]
                whalf[k] = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
        wu_half = interp2pt(self.UpdVar.W.bulkvalues[gw-1], self.UpdVar.W.bulkvalues[gw])
        GMV.TKE.values[gw] = get_surface_tke(Case.Sur.ustar, self.wstar, self.Gr.z_half[gw], Case.Sur.obukhov_length)
        self.get_env_tke_from_GMV(self.UpdVar.Area, self.UpdVar.W, self.EnvVar.W,
                                  self.EnvVar.TKE, &GMV.W.values[0], &GMV.TKE.values[0])

        tke_0_surf = self.EnvVar.TKE.values[gw]


        with nogil:
            for kk in xrange(nz):
                k = kk+gw
                D_env = 0.0

                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]

                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k] * self.tke_diss_coeff * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))/fmax(self.mixing_length[k],1.0))
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * self.EnvVar.TKE.values[k] * dti
                         + self.tke_shear[k] + self.tke_buoy[k] + self.tke_entr_gain[k] + self.tke_pressure[k])
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            x[0] = tke_0_surf

            b[nz-1] += c[nz-1]
            c[nz-1] = 0.0
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for kk in xrange(nz):
                k = kk + gw
                self.EnvVar.TKE.values[k] = fmax(x[kk], 0.0)
                wu_half = interp2pt(self.UpdVar.W.bulkvalues[k-1], self.UpdVar.W.bulkvalues[k])

        self.get_GMV_TKE(self.UpdVar.Area,self.UpdVar.W, self.EnvVar.W, self.EnvVar.TKE,
                             &GMV.W.values[0], &GMV.TKE.values[0])

        return

    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv, alpha


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.QL.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QL.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QL.values[k])

                # TODO - change to prognostic?
                GMV.QR.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QR.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QR.values[k])

                GMV.T.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.T.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.T.values[k])
                qv = GMV.QT.values[k] - GMV.QL.values[k]

                GMV.THL.values[k] = t_to_thetali_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k],
                                                   GMV.QL.values[k], 0.0)


                GMV.B.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.B.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.B.values[k])

        return



    cpdef compute_covariance(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        #if TS.nstep > 0: #TODO - its already initialized in t=0 here
        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(Case.Sur.obukhov_length)

        self.compute_covariance_entr()
        self.compute_covariance_shear(GMV)
        self.compute_covariance_rain(TS, GMV)
        self.reset_surface_covariance(GMV, Case)
        self.update_covariance_ED(GMV, Case, TS)
        self.cleanup_covariance(GMV)
        #else: #TODO - its initialized in the update calls
        #    self.initialize_covariance(GMV, Case)

        return


    cpdef initialize_covariance(self, GridMeanVariables GMV, CasesBase Case):

        cdef:
            Py_ssize_t k

            double ws= self.wstar, us = Case.Sur.ustar, zs = self.zi, z

        self.reset_surface_covariance(GMV, Case)

        with nogil:
            for k in xrange(self.Gr.nzg):
                z = self.Gr.z_half[k]
                GMV.Hvar.values[k]   = GMV.Hvar.values[self.Gr.gw]   * GMV.TKE.values[k] / GMV.TKE.values[self.Gr.gw]
                GMV.QTvar.values[k]  = GMV.QTvar.values[self.Gr.gw]  * GMV.TKE.values[k] / GMV.TKE.values[self.Gr.gw]
                GMV.HQTcov.values[k] = GMV.HQTcov.values[self.Gr.gw] * GMV.TKE.values[k] / GMV.TKE.values[self.Gr.gw]

        self.compute_mixing_length(Case.Sur.obukhov_length)

        return

    cpdef cleanup_covariance(self, GridMeanVariables GMV):
        cdef:
            double tmp_eps = 1e-18

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if GMV.Hvar.values[k] < tmp_eps:
                    GMV.Hvar.values[k] = 0.0
                if GMV.QTvar.values[k] < tmp_eps:
                    GMV.QTvar.values[k] = 0.0
                if self.EnvVar.Hvar.values[k] < tmp_eps:
                    self.EnvVar.Hvar.values[k] = 0.0
                if self.EnvVar.QTvar.values[k] < tmp_eps:
                    self.EnvVar.QTvar.values[k] = 0.0


    cpdef compute_covariance_shear(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double dH_high = 0.0
            double dQT_high = 0.0
            double dH_low, dQT_low

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                dH_low = dH_high
                dH_high = (self.EnvVar.H.values[k+1] - self.EnvVar.H.values[k]) * self.Gr.dzi
                dQT_low = dQT_high
                dQT_high = (self.EnvVar.QT.values[k+1] - self.EnvVar.QT.values[k]) * self.Gr.dzi
                self.Hvar_shear[k] = 2.0*(self.Ref.rho0_half[k] * ae[k] * self.KH.values[k] *pow(interp2pt(dH_low, dH_high),2.0))
                self.QTvar_shear[k] = 2.0*(self.Ref.rho0_half[k] * ae[k] * self.KH.values[k] *pow(interp2pt(dQT_low, dQT_high),2.0))
                self.HQTcov_shear[k] = 2.0*(self.Ref.rho0_half[k] * ae[k] * self.KH.values[k] *interp2pt(dH_low, dH_high)*interp2pt(dQT_low, dQT_high))
        return

    cpdef compute_covariance_entr(self):
        cdef:
            Py_ssize_t i, k
            double H_u, H_env, QT_u, QT_env

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.Hvar_entr_gain[k] = 0.0
                self.QTvar_entr_gain[k] = 0.0
                self.HQTcov_entr_gain[k] = 0.0
                H_env = self.EnvVar.H.values[k]
                QT_env = self.EnvVar.QT.values[k]
                for i in xrange(self.n_updrafts):
                    H_u = self.UpdVar.H.values[i,k]
                    QT_u = self.UpdVar.QT.values[i,k]
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    self.Hvar_entr_gain[k] +=   self.UpdVar.Area.values[i,k] * w_u * self.detr_sc[i,k] * (H_u - H_env) * (H_u - H_env)
                    self.QTvar_entr_gain[k] +=   self.UpdVar.Area.values[i,k] * w_u * self.detr_sc[i,k] * (QT_u - QT_env) * (QT_u - QT_env)
                    self.HQTcov_entr_gain[k] +=   self.UpdVar.Area.values[i,k] * w_u * self.detr_sc[i,k] * (H_u - H_env) * (QT_u - QT_env)
                self.Hvar_entr_gain[k] *= self.Ref.rho0_half[k]
                self.QTvar_entr_gain[k] *= self.Ref.rho0_half[k]
                self.HQTcov_entr_gain[k] *= self.Ref.rho0_half[k]
        return

    cpdef compute_covariance_detr(self):
        cdef:
            Py_ssize_t i, k

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.Hvar_detr_loss[k] = 0.0
                self.QTvar_detr_loss[k] = 0.0
                self.HQTcov_detr_loss[k] = 0.0
                for i in xrange(self.n_updrafts):
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    self.Hvar_detr_loss[k]   += self.UpdVar.Area.values[i,k] * w_u * self.entr_sc[i,k]
                    self.QTvar_detr_loss[k]  += self.UpdVar.Area.values[i,k] * w_u * self.entr_sc[i,k]
                    self.HQTcov_detr_loss[k] += self.UpdVar.Area.values[i,k] * w_u * self.entr_sc[i,k]
                self.Hvar_detr_loss[k]   *= self.Ref.rho0_half[k] * self.EnvVar.Hvar.values[k]
                self.QTvar_detr_loss[k]  *= self.Ref.rho0_half[k] * self.EnvVar.QTvar.values[k]
                self.HQTcov_detr_loss[k] *= self.Ref.rho0_half[k] * self.EnvVar.HQTcov.values[k]
        return

    cpdef compute_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i, k
            # TODO defined again in compute_covariance_shear and compute_covaraince
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.Hvar_rain[k]   = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.Hvar_rain_dt[k]   * TS.dti
                self.QTvar_rain[k]  = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.QTvar_rain_dt[k]  * TS.dti
                self.HQTcov_rain[k] = self.Ref.rho0_half[k] * ae[k] *      self.EnvThermo.HQTcov_rain_dt[k] * TS.dti

        return

    cpdef update_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS):
        cdef:
            Py_ssize_t k, kk, i
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double dzi = self.Gr.dzi
            double dti = TS.dti
            double alpha0LL  = self.Ref.alpha0_half[self.Gr.gw]
            double zLL = self.Gr.z_half[self.Gr.gw]
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] ae_old = np.subtract(np.ones((nzg,),dtype=np.double, order='c'), np.sum(self.UpdVar.Area.old,axis=0))
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] whalf = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] Hhalf = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] QThalf = np.zeros((nzg,),dtype=np.double, order='c')

            double  D_env = 0.0
            double Hu_half, He_half, a_half, QTu_half, QTe_half
            double wu_half, we_half, Hvar_0_surf, QTvar_0_surf, HQTcov_0_surf


        with nogil:
            for k in xrange(1,nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]
                whalf[k] = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                Hhalf[k] = self.EnvVar.H.values[k]
                QThalf[k] = self.EnvVar.QT.values[k]
        wu_half = interp2pt(self.UpdVar.W.bulkvalues[gw-1], self.UpdVar.W.bulkvalues[gw])
        Hu_half = self.UpdVar.H.bulkvalues[gw]
        QTu_half = self.UpdVar.QT.bulkvalues[gw]

        GMV.Hvar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_hflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        GMV.QTvar.values[gw] = get_surface_variance(Case.Sur.rho_qtflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        GMV.HQTcov.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        self.get_env_covar_from_GMV(self.UpdVar.Area, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H,
                                  self.EnvVar.Hvar, &GMV.H.values[0], &GMV.H.values[0], &GMV.Hvar.values[0])
        self.get_env_covar_from_GMV(self.UpdVar.Area, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT,
                                  self.EnvVar.QTvar, &GMV.QT.values[0], &GMV.QT.values[0], &GMV.QTvar.values[0])
        self.get_env_covar_from_GMV(self.UpdVar.Area, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT,
                                  self.EnvVar.HQTcov, &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])




        # BC  at the surface
        Hvar_0_surf = self.EnvVar.Hvar.values[gw]
        QTvar_0_surf = self.EnvVar.QTvar.values[gw]
        HQTcov_0_surf = self.EnvVar.HQTcov.values[gw]

        # run tridiagonal solver for Hvar
        with nogil:
            for kk in xrange(nz):
                k = kk+gw
                D_env = 0.0

                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]

                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k]  *sqrt(fmax(self.EnvVar.TKE.values[k],0))/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * self.EnvVar.Hvar.values[k] * dti
                         + self.Hvar_shear[k] + self.Hvar_entr_gain[k] + self.Hvar_rain[k])

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            x[0] = Hvar_0_surf

            b[nz-1] += c[nz-1]
            c[nz-1] = 0.0
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for kk in xrange(nz):
                k = kk + gw
                self.EnvVar.Hvar.values[k] = x[kk]
                GMV.Hvar.values[k] = (ae[k] * (self.EnvVar.Hvar.values[k] + (Hhalf[k]-GMV.H.values[k]) * (Hhalf[k]-GMV.H.values[k]))
                                  + self.UpdVar.Area.bulkvalues[k] * (self.UpdVar.H.bulkvalues[k]-GMV.H.values[k])  * (self.UpdVar.H.bulkvalues[k]-GMV.H.values[k]))

        self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar,
                             &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])

        # run tridiagonal solver for  QTvar
        with nogil:
            for kk in xrange(nz):
                k = kk+gw
                D_env = 0.0

                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]

                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k] * pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * self.EnvVar.QTvar.values[k] * dti
                         + self.QTvar_shear[k] + self.QTvar_entr_gain[k] + self.QTvar_rain[k])

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            x[0] = QTvar_0_surf

            b[nz-1] += c[nz-1]
            c[nz-1] = 0.0
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for kk in xrange(nz):
                k = kk + gw
                #self.EnvVar.Hvar.values[k] = fmax(x[kk], 0.0)
                self.EnvVar.QTvar.values[k] = x[kk]
                GMV.QTvar.values[k] = (ae[k] * (self.EnvVar.QTvar.values[k] + (QThalf[k]-GMV.QT.values[k]) * (QThalf[k]-GMV.QT.values[k]))
                                  + self.UpdVar.Area.bulkvalues[k] * (self.UpdVar.QT.bulkvalues[k]-GMV.QT.values[k])  * (self.UpdVar.QT.bulkvalues[k]-GMV.QT.values[k]))

        self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,
                             &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])

        # run tridiagonal solver for HQTcov
        with nogil:
            for kk in xrange(nz):
                k = kk+gw
                D_env = 0.0

                for i in xrange(self.n_updrafts):
                    wu_half = self.UpdVar.W.values[i,k]
                    D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]

                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k]
                         * pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * self.EnvVar.HQTcov.values[k] * dti
                         + self.HQTcov_shear[k] + self.HQTcov_entr_gain[k] + self.HQTcov_rain[k])

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            x[0] = HQTcov_0_surf

            b[nz-1] += c[nz-1]
            c[nz-1] = 0.0
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for kk in xrange(nz):
                k = kk + gw
                self.EnvVar.HQTcov.values[k] = x[kk]
                GMV.HQTcov.values[k] = (ae[k] * (self.EnvVar.HQTcov.values[k] + (Hhalf[k]-GMV.H.values[k]) * (QThalf[k]-GMV.QT.values[k]))
                                  + self.UpdVar.Area.bulkvalues[k] * (self.UpdVar.H.bulkvalues[k]-GMV.H.values[k])  * (self.UpdVar.QT.bulkvalues[k]-GMV.QT.values[k]))

        self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov,
                             &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])

        return

    cpdef compute_covariance_dissipation(self):
        cdef:
            Py_ssize_t k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.Hvar_dissipation[k] = (self.Ref.rho0_half[k] * ae[k] * self.EnvVar.Hvar.values[k]
                                    *pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                self.QTvar_dissipation[k] = (self.Ref.rho0_half[k] * ae[k] * self.EnvVar.QTvar.values[k]
                                    *pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                self.HQTcov_dissipation[k] = (self.Ref.rho0_half[k] * ae[k] * self.EnvVar.HQTcov.values[k]
                                    *pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)

        return


