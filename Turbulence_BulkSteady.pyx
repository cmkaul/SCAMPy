#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import pylab as plt
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
from utility_functions cimport interp2pt, gaussian_mean
from libc.math cimport fmax, sqrt, exp, pow
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from Turbulence_BulkSteady cimport EDMF_BulkSteady






cdef class EDMF_BulkSteady(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self, paramlist, Gr, Ref)
        try:
            self.dt = namelist['time_stepping']['dt']
        except:
            self.dt = 1.0
        # Set the number of updrafts (1)
        try:
            self.n_updrafts = namelist['turbulence']['EDMF_BulkSteady']['updraft_number']
        except:
            self.n_updrafts = 1
            print('Turbulence--EDMF_BulkSteady: defaulting to single updraft')
        try:
            self.const_area = namelist['turbulence']['EDMF_BulkSteady']['constant_area']
        except:
            self.const_area = True
            print('Turbulence--EDMF_BulkSteady: defaulting to constant area fraction')
        try:
            self.use_local_micro = namelist['turbulence']['EDMF_BulkSteady']['use_local_micro']
        except:
            self.use_local_micro = True

        try:
            if namelist['turbulence']['EDMF_BulkSteady']['entrainment'] == 'cloudy':
                self.entr_detr_fp = entr_detr_cloudy
            elif namelist['turbulence']['EDMF_BulkSteady']['entrainment'] == 'dry':
                self.entr_detr_fp = entr_detr_dry
            elif namelist['turbulence']['EDMF_BulkSteady']['entrainment'] == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z

            else:
                print('Turbulence--EDMF_BulkSteady: Entrainment rate namelist option is not recognized')
        except:

            self.entr_detr_fp = entr_detr_cloudy
            print('Turbulence--EDMF_BulkSteady: defaulting to cloudy entrainment formulation')


        #
        self.extrapolate_buoyancy = False

        # Get parameters
        self.surface_area = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']
        self.surface_scalar_coeff = paramlist['turbulence']['EDMF_BulkSteady']['surface_scalar_coeff']
        self.w_entr_coeff = paramlist['turbulence']['EDMF_BulkSteady']['w_entr_coeff']
        self.w_buoy_coeff = paramlist['turbulence']['EDMF_BulkSteady']['w_buoy_coeff']
        self.max_area_factor = paramlist['turbulence']['EDMF_BulkSteady']['max_area_factor']
        self.entrainment_factor = paramlist['turbulence']['EDMF_BulkSteady']['entrainment_factor']
        self.detrainment_factor = paramlist['turbulence']['EDMF_BulkSteady']['detrainment_factor']


        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = EDMF_Updrafts.UpdraftMicrophysics(paramlist, self.n_updrafts, Gr, Ref)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EDMF_Environment.EnvironmentVariables(namelist,Gr)

        # Entrainment rates
        self.entr_w = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_w = np.zeros((self.n_updrafts,Gr.nzg),dtype=np.double,order='c')
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

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

        Stats.add_profile('entrainment_w')
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

        Stats.add_profile('precip_source_h')
        Stats.add_profile('precip_source_qt')

        return

    cpdef io(self, NetCDFIO_Stats Stats):

        cdef:
            Py_ssize_t k, i
            double [:] mean_entr_w = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_entr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_detr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')


        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        mean_entr_w[k] += self.UpdVar.Area.values[i,k] * self.entr_w[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_entr_sc[k] += self.UpdVar.Area.values[i,k] * self.entr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_detr_sc[k] += self.UpdVar.Area.values[i,k] * self.detr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]


        Stats.write_profile('entrainment_w', mean_entr_w[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('entrainment_sc', mean_entr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('detrainment_sc', mean_detr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('massflux', np.sum(self.m[:,self.Gr.gw-1:self.Gr.nzg-self.Gr.gw -1], axis=0))
        Stats.write_profile('massflux_h', self.massflux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('massflux_qt', self.massflux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])

        Stats.write_profile('massflux_tendency_h', self.massflux_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_qt', self.massflux_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('diffusive_flux_h', self.diffusive_flux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('diffusive_flux_qt', self.diffusive_flux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1])
        Stats.write_profile('diffusive_tendency_h', self.diffusive_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_qt', self.diffusive_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('total_flux_h', np.add(self.massflux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1],
                                                   self.diffusive_flux_h[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1]))
        Stats.write_profile('total_flux_qt', np.add(self.massflux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1],
                                                    self.diffusive_flux_qt[self.Gr.gw-1:self.Gr.nzg-self.Gr.gw-1]))

        Stats.write_profile('precip_source_h', np.divide(self.UpdMicro.prec_source_h_tot[self.Gr.gw:self.Gr.nzg-self.Gr.gw],self.dt))
        Stats.write_profile('precip_source_qt', np.divide(self.UpdMicro.prec_source_qt_tot[self.Gr.gw:self.Gr.nzg-self.Gr.gw],self.dt))

        return

    # Perform the update of the scheme
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS ):
        cdef:
            Py_ssize_t kmin= self.Gr.gw
            Py_ssize_t kmax =self.Gr.nzg-self.Gr.gw

        # Perform the environmental/updraft decomposition
        ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV,Case)
        self.decompose_environment(GMV, 'values')


        # Solve updraft equations
        self.compute_entrainment_detrainment(GMV, Case)
        self.set_updraft_surface_bc(GMV, Case)


        self.solve_updraft_scalars(GMV)
        self.UpdThermo.buoyancy(self.UpdVar, GMV, self.extrapolate_buoyancy)



        self.solve_updraft_velocity()
        self.solve_area_fraction(GMV)
        self.apply_updraft_microphysics()


        # Update GMV.mf_update with mass-flux tendencies and updraft source terms
        self.update_GMV_MF(GMV, TS)

        # Compute the decomposition based on the updated updraft variables
        self.decompose_environment(GMV, 'mf_update')

        # Compute the eddy diffusion term with the updated environmental values
        self.update_GMV_ED(GMV, Case, TS)


        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)


        # PLOTS
        # plt.figure('Updated  W')
        # plt.plot(self.UpdVar.W.values[0,kmin-1:kmax], self.Gr.z[kmin-1:kmax], label='upd')
        # plt.plot(self.EnvVar.W.values[kmin:kmax],self.Gr.z_half[kmin:kmax], label='env')
        # plt.legend(loc=0)
        #
        # plt.figure('Updated Updraft Area')
        # plt.plot(self.UpdVar.Area.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax], label='upd')
        #
        # plt.figure('Updated QT')
        # plt.plot(self.UpdVar.QT.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='upd')
        # plt.plot(self.EnvVar.QT.values[kmin:kmax],self.Gr.z_half[kmin:kmax],label='env')
        # plt.plot(GMV.QT.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        # plt.plot(GMV.QT.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        # plt.plot(GMV.QT.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        # plt.legend(loc=0)
        # #
        # #
        # plt.figure('Viscosity')
        # plt.plot(self.KM.values[kmin:kmax], self.Gr.z_half[kmin:kmax],label='KM')
        # plt.plot(self.KH.values[kmin:kmax], self.Gr.z[kmin:kmax],label='KH')
        #
        # plt.legend(loc=0)
        # #
        #
        # plt.figure('Updated H')
        # plt.plot(self.UpdVar.H.values[0,kmin:kmax], self.Gr.z_half[kmin:kmax],label='upd')
        # plt.plot(self.EnvVar.H.values[kmin:kmax], self.Gr.z_half[kmin:kmax],label='env')
        # plt.plot(GMV.H.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        # plt.plot(GMV.H.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        # plt.plot(GMV.H.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        # plt.legend(loc=0)
        # #
        # plt.figure('Updated U')
        # plt.plot(GMV.U.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        # plt.plot(GMV.U.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        # plt.plot(GMV.U.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        # plt.legend(loc=0)
        # #
        # plt.figure('Updated V')
        # plt.plot(GMV.V.values[kmin:kmax], self.Gr.z_half[kmin:kmax], label='values')
        # plt.plot(GMV.V.mf_update[kmin:kmax], self.Gr.z_half[kmin:kmax], label='mf')
        # plt.plot(GMV.V.new[kmin:kmax], self.Gr.z_half[kmin:kmax], label='new')
        # plt.legend(loc=0)
        #
        # plt.show()



        return


    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    # Find values of environmental variables by subtracting updraft values from grid mean values
    # whichvals used to check which substep we are on--correspondingly use 'GMV.SomeVar.value' (last timestep value)
    # or GMV.SomeVar.mf_update (GMV value following massflux substep)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals):

        # first make sure the 'bulkvalues' of the updraft variables are updated
        self.UpdVar.set_means(GMV)

        cdef:
            Py_ssize_t k
            double val1, val2, au_full


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

        return

    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            entr_struct ret

        self.update_inversion(GMV, Case.inversion_option)

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    ret = self.entr_detr_fp(self.Gr.z[k], self.Gr.z_half[k], self.zi)
                    self.entr_w[i,k] = ret.entr_w * self.entrainment_factor
                    self.entr_sc[i,k] = ret.entr_sc * self.entrainment_factor
                    self.detr_w[i,k] = ret.detr_w * self.detrainment_factor
                    self.detr_sc[i,k] = ret.detr_sc * self.detrainment_factor

        return



    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case):

        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        cdef:
            Py_ssize_t i
            double e_srf = 3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar
            Py_ssize_t gw = self.Gr.gw

        with nogil:
            for i in xrange(self.n_updrafts):
                self.area_surface_bc[i] = self.surface_area/self.n_updrafts
                self.w_surface_bc[i] = 0.0
                self.h_surface_bc[i] = (GMV.H.values[gw] + self.surface_scalar_coeff
                                        * Case.Sur.rho_hflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])
                self.qt_surface_bc[i] = (GMV.QT.values[gw] + self.surface_scalar_coeff
                                         * Case.Sur.rho_qtflux/sqrt(e_srf) * self.Ref.alpha0_half[gw])

        return





    # solve the updraft velocity equation
    cpdef solve_updraft_velocity(self):
        cdef:
            Py_ssize_t k, i
            double a, b, c
            double dz = self.Gr.dz, dzi = self.Gr.dzi
            double w2, w, B_k, entr_w


        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.W.values[i, self.Gr.gw-1] = self.w_surface_bc[i]
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    w = self.UpdVar.W.values[i,k-1]
                    B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                    entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                    w2 = (self.w_buoy_coeff * B_k + 0.5 * w *w * dzi)/(0.5*dzi + self.w_entr_coeff * entr_w )
                    if w2 > 0.0:
                        self.UpdVar.W.values[i,k] = sqrt(w2)
                    else:
                        self.UpdVar.W.values[i,k:] = 0
                        break

        self.UpdVar.W.set_bcs(self.Gr)

        return

    cpdef solve_area_fraction(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, i
            double dzi= self.Gr.dzi
            Py_ssize_t gw = self.Gr.gw
            double w_k, w_km, m_km, a_lim

        if self.const_area:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                    for k in xrange(gw+1, self.Gr.nzg):

                        if self.UpdVar.W.values[i,k] > 0.0:
                            self.UpdVar.Area.values[i,k] = self.area_surface_bc[i]
                        else:
                            # the updraft has terminated so set its area fraction to zero at this height and all heights above
                            self.UpdVar.Area.values[i,k] =  self.area_surface_bc[i]
                            self.UpdVar.H.values[i,k] = GMV.H.values[k]
                            self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                            self.UpdVar.B.values[i,k] = 0.0
                            self.UpdVar.QL.values[i,k] = 0.0
                            self.UpdVar.T.values[i,k] = GMV.T.values[k]
                            self.UpdMicro.prec_source_h[i,k] = 0.0
                            self.UpdMicro.prec_source_qt[i,k] = 0.0

        else:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                    a_lim = self.max_area_factor*self.area_surface_bc[i]
                    w_k =interp2pt(self.UpdVar.W.values[i,gw], self.UpdVar.W.values[i,gw-1])

                    for k in xrange(gw+1, self.Gr.nzg):
                        w_km = w_k
                        w_k = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                        if w_k > 0.0:
                            m_km = self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1] * w_km
                            self.UpdVar.Area.values[i,k] =( m_km * dzi / (self.Ref.rho0_half[k] * w_k
                                                                          * (dzi-self.entr_sc[i,k]+ self.detr_sc[i,k])))
                            # # Limit the increase in updraft area when the updraft decelerates
                            if self.UpdVar.Area.values[i,k] >  a_lim:
                                self.detr_sc[i,k-1] = m_km * dzi /self.Ref.rho0_half[k]/w_k/a_lim + self.entr_sc[i,k] - dzi
                                self.UpdVar.Area.values[i,k] = a_lim
                        else:
                            # the updraft has terminated so set its area fraction to zero at this height and all heights above
                            self.UpdVar.Area.values[i,k] = 0.0
                            self.UpdVar.H.values[i,k] = GMV.H.values[k]
                            self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                            self.UpdVar.B.values[i,k] = 0.0
                            self.UpdVar.T.values[i,k] = GMV.T.values[k]
                            self.UpdVar.QL.values[i,k] = 0.0
                            self.UpdMicro.prec_source_h[i,k] = 0.0
                            self.UpdMicro.prec_source_qt[i,k] = 0.0


        self.UpdVar.Area.set_bcs(self.Gr)
        return

    cpdef solve_updraft_scalars(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, i
            double dz = self.Gr.dz
            Py_ssize_t gw = self.Gr.gw
            double denom
            eos_struct sa

        if self.use_local_micro:
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
                                                                       &self.UpdVar.H.values[i,gw], i, gw)

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
                                                                           &self.UpdVar.H.values[i,k], i, k)

        else:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.H.values[i,gw] = self.h_surface_bc[i]
                    self.UpdVar.QT.values[i,gw] = self.qt_surface_bc[i]
                    # Find the cloud liquid content
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[gw],
                             self.UpdVar.QT.values[i,gw], self.UpdVar.H.values[i,gw])
                    self.UpdVar.QL.values[i,gw] = sa.ql
                    self.UpdVar.T.values[i,gw] = sa.T

                    for k in xrange(gw+1, self.Gr.nzg-gw):
                        denom = 1.0 + self.entr_sc[i,k] * dz
                        self.UpdVar.H.values[i,k] = (self.UpdVar.H.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.H.values[k])/denom
                        self.UpdVar.QT.values[i,k] = (self.UpdVar.QT.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.QT.values[k])/denom
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                        self.UpdVar.QL.values[i,k] = sa.ql
                        self.UpdVar.T.values[i,k] = sa.T


        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)

        return

    cpdef apply_updraft_microphysics(self):
        if self.use_local_micro:
            self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h,
                                                                 self.UpdVar.Area.values), axis=0)
            self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt,
                                                                  self.UpdVar.Area.values), axis=0)
        else:
            # Compute the updraft microphysical sources-
            self.UpdMicro.compute_sources(self.UpdVar)
            # Update updraft variables with microphysical source tendencies
            self.UpdMicro.update_updraftvars(self.UpdVar)
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
            double gmv_h_interp
            double gmv_qt_interp
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        with nogil:
            for i in xrange(self.n_updrafts):
                self.m[i,gw-1] = 0.0
                for k in xrange(self.Gr.gw, self.Gr.nzg-1):
                    self.m[i,k] = self.UpdVar.W.values[i,k] * self.Ref.rho0[k] * interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1])

        self.massflux_h[gw-1] = 0.0
        self.massflux_qt[gw-1] = 0.0
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                self.massflux_h[k] = 0.0
                self.massflux_qt[k] = 0.0
                gmv_h_interp = interp2pt(GMV.H.values[k], GMV.H.values[k+1])
                gmv_qt_interp = interp2pt(GMV.QT.values[k], GMV.QT.values[k+1])
                for i in xrange(self.n_updrafts):
                    self.massflux_h[k] += self.m[i,k] * (interp2pt(self.UpdVar.H.values[i,k], self.UpdVar.H.values[i,k+1]) - gmv_h_interp )
                    self.massflux_qt[k] += self.m[i,k] * (interp2pt(self.UpdVar.QT.values[i,k], self.UpdVar.QT.values[i,k+1]) - gmv_qt_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variables
        with nogil:

            for k in xrange(self.Gr.gw, self.Gr.nzg):
                mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
                mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h #+ self.UpdMicro.prec_source_h_tot[k]
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt #+ self.UpdMicro.prec_source_qt_tot[k]

                # Horiontal velocities have no mass flux tendency (u_updraft = u_environment, v_updraft = v_environment)
                GMV.U.mf_update[k] = GMV.U.values[k]
                GMV.V.mf_update[k] = GMV.V.values[k]

                # Prepare the output
                self.massflux_tendency_h[k] = mf_tend_h
                self.massflux_tendency_qt[k] = mf_tend_qt

        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)
        return



    cpdef update_GMV_MF_implicitMF(self, GridMeanVariables GMV, TimeStepping TS):
        # cdef:
        #     Py_ssize_t k, i
        #     Py_ssize_t gw = self.Gr.gw
        #     double mf_tend_h=0.0, mf_tend_qt=0.0
        #     double gmv_h_interp
        #     double gmv_qt_interp
        # self.massflux_h[:] = 0.0
        # self.massflux_qt[:] = 0.0
        #
        # # Compute the mass flux and associated scalar fluxes
        # with nogil:
        #     for i in xrange(self.n_updrafts):
        #         self.m[i,gw-1] = 0.0
        #         for k in xrange(self.Gr.gw, self.Gr.nzg-1):
        #             self.m[i,k] = self.UpdVar.W.values[i,k] * self.Ref.rho0[k] * interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1])
        #
        # self.massflux_h[gw-1] = 0.0
        # self.massflux_qt[gw-1] = 0.0
        # with nogil:
        #     for k in xrange(gw, self.Gr.nzg-gw-1):
        #         self.massflux_h[k] = 0.0
        #         self.massflux_qt[k] = 0.0
        #         gmv_h_interp = 0.0#interp2pt(GMV.H.values[k], GMV.H.values[k+1])
        #         gmv_qt_interp = 0.0 #interp2pt(GMV.QT.values[k], GMV.QT.values[k+1])
        #         for i in xrange(self.n_updrafts):
        #             self.massflux_h[k] += self.m[i,k] * (interp2pt(self.UpdVar.H.values[i,k], self.UpdVar.H.values[i,k+1]) - gmv_h_interp )
        #             self.massflux_qt[k] += self.m[i,k] * (interp2pt(self.UpdVar.QT.values[i,k], self.UpdVar.QT.values[i,k+1]) - gmv_qt_interp )
        #
        # # Compute the  mass flux tendencies
        # # Adjust the values of the grid mean variables
        # with nogil:
        #
        #     for k in xrange(self.Gr.gw, self.Gr.nzg):
        #         mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
        #         mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
        #
        #         GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdMicro.prec_source_h_tot[k]
        #         GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k]
        #
        #         # Horiontal velocities have no mass flux tendency (u_updraft = u_environment, v_updraft = v_environment)
        #         GMV.U.mf_update[k] = GMV.U.values[k]
        #         GMV.V.mf_update[k] = GMV.V.values[k]
        #
        #         # Prepare the output
        #         self.massflux_tendency_h[k] = mf_tend_h
        #         self.massflux_tendency_qt[k] = mf_tend_qt
        #
        # GMV.H.set_bcs(self.Gr)
        # GMV.QT.set_bcs(self.Gr)
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
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            #double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] ae = np.ones((nzg,), dtype=np.double, order='c')
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, self.Gr.dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] =  GMV.QT.mf_update[k+gw] #self.EnvVar.QT.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.QT.new[k+gw] = x[k]  # GMV.QT.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])
                self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
                self.diffusive_flux_qt[k+gw] = -rho_ae_K_m[k+gw] * (GMV.QT.new[k+gw+1] - GMV.QT.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            self.diffusive_flux_qt[gw-1] = Case.Sur.rho_qtflux*self.Ref.alpha0_half[gw]

        # Solve H
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.H.mf_update[k+gw]   # self.EnvVar.H.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.H.new[k+gw] = x[k] #GMV.H.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])
                self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
                self.diffusive_flux_h[k+gw] =  -rho_ae_K_m[k+gw] * (GMV.H.new[k+gw+1] - GMV.H.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]


        # Solve U
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.U.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_uflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.U.new[k+gw] = x[k]
            #     self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
            #     self.diffusive_flux_h[k+gw] =  -rho_ae_K_m[k+gw] * (GMV.H.new[k+gw+1] - GMV.H.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            # self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]

        # Solve V
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.V.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_vflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.V.new[k+gw] = x[k]
            #     self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
            #     self.diffusive_flux_h[k+gw] =  -rho_ae_K_m[k+gw] * (GMV.H.new[k+gw+1] - GMV.H.new[k+gw]) * self.Gr.dzi * self.Ref.alpha0[k+gw]
            # self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]
        return

    cpdef update_GMV_ED_implicitMF(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        # cdef:
        #     Py_ssize_t k
        #     Py_ssize_t gw = self.Gr.gw
        #     Py_ssize_t nzg = self.Gr.nzg
        #     Py_ssize_t nz = self.Gr.nz
        #     double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        #     double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        #     double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        #     double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        #     #double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
        #     double [:] ae = np.ones((nzg,), dtype=np.double, order='c')
        #     double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
        #     double [:] massflux = np.sum(self.m,axis=0)
        #
        # with nogil:
        #     for k in xrange(nzg-1):
        #         rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]
        #
        # # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        # construct_tridiag_diffusion_implicitMF(nzg, gw, self.Gr.dzi, TS.dt, &rho_ae_K_m[0],&massflux[0], &self.Ref.rho0_half[0],&self.Ref.alpha0[0],
        #                             &ae[0], &a[0], &b[0], &c[0])
        #
        # # Solve QT
        # with nogil:
        #     for k in xrange(nz):
        #         x[k] =  GMV.QT.mf_update[k+gw] #self.EnvVar.QT.values[k+gw]
        #     x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        # tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        #
        # with nogil:
        #     for k in xrange(nz):
        #         GMV.QT.new[k+gw] = x[k]  # GMV.QT.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])
        #         self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
        #         self.diffusive_flux_qt[k+gw] = -self.KH.values[k+gw] * 0.5* (GMV.QT.new[k+gw+1] - GMV.QT.new[k+gw-1]) * self.Gr.dzi
        #     self.diffusive_flux_qt[gw-1] = Case.Sur.rho_qtflux*self.Ref.alpha0_half[gw]
        #
        # # Solve H
        # with nogil:
        #     for k in xrange(nz):
        #         x[k] = GMV.H.mf_update[k+gw]   # self.EnvVar.H.values[k+gw]
        #     x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * self.Gr.dzi * self.Ref.alpha0_half[gw]/ae[gw]
        # tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        #
        # with nogil:
        #     for k in xrange(nz):
        #         GMV.H.new[k+gw] = x[k] #GMV.H.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])
        #         self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
        #         self.diffusive_flux_h[k+gw] = -self.KH.values[k+gw] * 0.5* (GMV.H.new[k+gw+1] - GMV.H.new[k+gw-1]) * self.Gr.dzi
        #     self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]
        return




