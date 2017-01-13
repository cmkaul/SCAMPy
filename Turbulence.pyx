#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import pylab as plt
import numpy as np
include "parameters.pxi"
include "parameters_edmf.pxi"
import cython
cimport  EDMF_Updrafts
from Grid cimport Grid
cimport EDMF_Environment
from Variables cimport VariableDiagnostic, GridMeanVariables
from Surface cimport SurfaceBase
from Cases cimport  CasesBase
from ReferenceState cimport  ReferenceState
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport  *
from turbulence_functions cimport *
from utility_functions cimport interp2pt
from libc.math cimport fmax, sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

def ParameterizationFactory(namelist, Grid Gr, ReferenceState Ref):
    scheme = namelist['turbulence']['scheme']
    if scheme == 'EDMF_BulkSteady':
        return  EDMF_BulkSteady(namelist, Gr, Ref)
    if scheme == 'SimilarityED':
        return SimilarityED(namelist, Gr, Ref)
    else:
        print('Did not recognize parameterization ' + scheme)
        return


# A base class common to all turbulence parameterizations
cdef class ParameterizationBase:
    def __init__(self, Grid Gr, ReferenceState Ref):
        self.turbulence_tendency  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.Gr = Gr # grid class
        self.Ref = Ref # reference state class
        self.KM = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'diffusivity', 'm^2/s') # eddy viscosity
        self.KH = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'viscosity', 'm^2/s') # eddy diffusivity

        return
    cpdef initialize(self, GridMeanVariables GMV):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        return

    # Calculate the tendency of the grid mean variables due to turbulence as the difference between the values at the beginning
    # and  end of all substeps taken
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS ):
        cdef:
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t k

        with nogil:
            for k in xrange(gw,nzg-gw):
                GMV.H.tendencies[k] = (GMV.H.new[k] - GMV.H.values[k]) * TS.dti
                GMV.QT.tendencies[k] = (GMV.QT.new[k] - GMV.QT.values[k]) * TS.dti
        return

    # Update the diagnosis of the inversion height, using the maximum temperature gradient method
    cpdef update_inversion(self, GridMeanVariables GMV, option ):
        cdef:
            double maxgrad = 0.0
            double grad
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double qv = GMV.QT.values[gw] - GMV.QL.values[gw]
            double theta_rho_surf = theta_rho_c(self.Ref.p0_half[gw], GMV.T.values[gw], GMV.QT.values[gw], qv)
            double theta_rho_k

        if option == 'theta_rho':
            with nogil:
                for k in xrange(self.Gr.gw, self.Gr.gw + self.Gr.nz-1):
                    qv=GMV.QT.values[k] - GMV.QL.values[k]
                    theta_rho_k = theta_rho_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)
                    if theta_rho_k > theta_rho_surf:
                        self.zi = self.Gr.z_half[k]
                        break
        elif option == 'thetal_maxgrad':

            with nogil:
                for k in xrange(self.Gr.gw, self.Gr.gw + self.Gr.nz-1):
                    grad =  (GMV.THL.values[k+1] - GMV.THL.values[k])*self.Gr.dzi
                    if grad > maxgrad:
                        maxgrad = grad
                        self.zi = self.Gr.z[k]
        else:
            print('INVERSION HEIGHT OPTION NOT RECOGNIZED')


        print('Inversion height ', self.zi)




        return

    cpdef compute_wstar(self, CasesBase Case):
        self.wstar = np.max((g/300.0*Case.Sur.shf/cpm_c(Case.Sur.qsurface)*self.Ref.alpha0[self.Gr.gw-1] * self.zi)**(1.0/3.0), 0.0)
        return

    # Compute eddy diffusivities from similarity theory (Siebesma 2007)
    cpdef compute_eddy_diffusivities_similarity(self, GridMeanVariables GMV, CasesBase Case):
        self.update_inversion(GMV, Case.inversion_option)
        self.compute_wstar(Case)

        cdef:
            double ustar = Case.Sur.ustar
            double zzi
            double wstar = (g/300.0*Case.Sur.shf/cpm_c(Case.Sur.qsurface)*self.Ref.alpha0[self.Gr.gw-1] * self.zi)**(1.0/3.0)
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
        with nogil:
            for k in xrange(gw,nzg-gw):
                zzi = self.Gr.z_half[k]/self.zi
                if zzi <= 1.0:
                    self.KH.values[k] = vkb * ( (ustar/wstar)**3 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * wstar * self.zi
                    self.KM.values[k] = self.KH.values[k]/3.0
                else:
                    self.KH.values[k] = 0.0
                    self.KM.values[k] = 0.0

        # Set the boundary points at top and bottom of domain
        self.KH.set_bcs(self.Gr)
        self.KM.set_bcs(self.Gr)
        return



cdef class EDMF_BulkSteady(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, Grid Gr, ReferenceState Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self,  Gr, Ref)
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
            self.surface_area = namelist['turbulence']['EDMF_BulkSteady']['surface_area']
        except:
            self.surface_area = 0.1
            print('Turbulence--EDMF_BulkSteady: defaulting to 10% area fraction')
        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = EDMF_Updrafts.UpdraftMicrophysics(self.n_updrafts, Gr, Ref)

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

        return

    # Perform the update of the scheme
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS ):
        # Perform the environmental/updraft decomposition

        self.decompose_environment(GMV, 'values')


        # Solve updraft equations
        self.set_updraft_surface_bc(GMV, Case)


        self.solve_updraft_scalars(GMV)
        self.UpdThermo.buoyancy(self.UpdVar, GMV)

        self.compute_entrainment_detrainment()

        self.solve_updraft_velocity(TS)
        self.solve_area_fraction(GMV)


        # Compute the updraft microphysical sources--Here trying something different from Zhihong and integrating the
        # dynamic updraft equations upward, then applying precipitation adjustment instead of Zhihong's level-by-level
        # integrate/precipitate, integrate/precipitate approach
        self.UpdMicro.compute_sources(self.UpdVar)

        # Update updraft variables with microphysical source tendencies
        self.UpdMicro.update_updraftvars(self.UpdVar)

        # Update GMV.mf_update with mass-flux tendencies and updraft source terms
        self.update_GMV_MF(GMV, TS)

        # Compute the decomposition based on the updated updraft variables
        self.decompose_environment(GMV, 'mf_update')

        # Compute the eddy diffusion term with the updated environmental values
        ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV,Case)
        self.update_GMV_ED(GMV, Case, TS)



        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)
        return


    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    cpdef compute_wstar(self, CasesBase Case):
        ParameterizationBase.compute_wstar(self,Case)
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

    cpdef compute_entrainment_detrainment(self):
        cdef:
            Py_ssize_t k, cloud_base_index
            entr_struct ret
            bint above_cloudbase

        with nogil:
            for i in xrange(self.n_updrafts):
                cloud_base_index = self.Gr.nzg
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    if self.UpdVar.QL.values[i,k] > 1.0e-8:
                        cloud_base_index = k
                        break
                above_cloudbase = False
                for k in xrange(cloud_base_index):
                    ret = entr_detr(self.Gr.z[k], self.Gr.z_half[k], above_cloudbase)
                    self.entr_w[i,k] = ret.entr_w
                    self.entr_sc[i,k] = ret.entr_sc
                    self.detr_w[i,k] = ret.detr_w
                    self.detr_sc[i,k] = ret.detr_sc

                above_cloudbase = True
                for k in xrange(cloud_base_index, self.Gr.nzg):
                    ret = entr_detr(self.Gr.z[k], self.Gr.z_half[k], above_cloudbase)
                    self.entr_w[i,k] = ret.entr_w
                    self.entr_sc[i,k] = ret.entr_sc
                    self.detr_w[i,k] = ret.detr_w
                    self.detr_sc[i,k] = ret.detr_sc


        return



    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case):

        self.update_inversion(GMV, Case.inversion_option)
        self.compute_wstar(Case)

        cdef:
            Py_ssize_t i
            double e_srf = 3.75 * Case.Sur.ustar * Case.Sur.ustar + 0.2 * self.wstar * self.wstar
            Py_ssize_t gw = self.Gr.gw

        with nogil:
            for i in xrange(self.n_updrafts):
                self.area_surface_bc[i] = self.surface_area/self.n_updrafts
                self.w_surface_bc[i] = 0.0
                self.h_surface_bc[i] = GMV.H.values[gw] + 0.3 * Case.Sur.rho_hflux/sqrt(e_srf) * self.Ref.alpha0_half[gw]
                self.qt_surface_bc[i] = GMV.QT.values[gw] + 0.3 * Case.Sur.rho_qtflux/sqrt(e_srf) * self.Ref.alpha0_half[gw]

        return





    # solve the updraft velocity equation
    cpdef solve_updraft_velocity(self,TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            double a, b, c
            double dz = self.Gr.dz
            double w2, w


        # with nogil:
        #     for i in xrange(self.n_updrafts):
        #         self.UpdVar.W.values[i, self.Gr.gw-1] = self.w_surface_bc[i]
        #         for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
        #             a = 1.0 + 0.5 * dz * w_b * self.entr_w[i,k]
        #             b = dz * w_b * self.entr_w[i,k] * self.UpdVar.W.values[i,k-1]
        #             c = (0.5 * dz * w_b * self.entr_w[i,k] -1.0 ) * self.UpdVar.W.values[i,k-1]*self.UpdVar.W.values[i,k-1] - 2.0  * dz * w_a * self.UpdVar.B.values[i,k]
        #             if b*b - 4.0 *a *c > 0.0:
        #                 self.UpdVar.W.values[i,k] = fmax( (-b + sqrt(b*b-4.0*a*c))/(2.0*a), 0.0)
        #             else:
        #                 self.UpdVar.W.values[i,k] = 0.0
        #             if self.UpdVar.W.values[i,k] == 0.0:
        #                 self.UpdVar.W.values[i,k+1:]=0.0
        #                 break
        #

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.W.values[i, self.Gr.gw-1] = self.w_surface_bc[i]

                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    w = self.UpdVar.W.values[i,k-1]
                    w2 = w * w + 2.0 * dz *(w_a *self.UpdVar.B.values[i,k] + w_b *self.entr_w[i,k] * w * (self.EnvVar.W.values[k-1] - w))


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
            double dz= self.Gr.dz
            Py_ssize_t gw = self.Gr.gw
            double w_mid, w_low

        if self.const_area:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                    for k in xrange(gw+1, self.Gr.nzg-gw):

                        if self.UpdVar.W.values[i,k] > 0.0:
                            self.UpdVar.Area.values[i,k] = self.area_surface_bc[i]
                        else:
                            # the updraft has terminated so set its area fraction to zero at this height and all heights above
                            self.UpdVar.Area.values[i,k] =  self.area_surface_bc[i]
                            self.UpdVar.H.values[i,k] = GMV.H.values[k]
                            self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                            self.UpdVar.B.values[i,k] = 0.0

        else:
            with nogil:
                for i in xrange(self.n_updrafts):
                    self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                    # w_mid = 0.5* (self.UpdVar.W.values[i,gw] + self.UpdVar.W.values[i,gw-1])
                    w_mid = 0.5* (self.UpdVar.W.values[i,gw])
                    for k in xrange(gw+1, self.Gr.nzg-gw):
                        w_low = w_mid
                        w_mid = 0.5*(self.UpdVar.W.values[i,k]+self.UpdVar.W.values[i,k-1])
                        if self.UpdVar.W.values[i,k] > 0.0:
                            self.UpdVar.Area.values[i,k] = (self.Ref.rho0_half[k-1]*self.UpdVar.Area.values[i,k-1]*w_low/
                                                            (1.0-(self.entr_sc[i,k-1]-self.detr_sc[i,k-1])*dz)/w_mid/self.Ref.rho0_half[k])
                            # # Limit the increase in updraft area when the updraft decelerates
                            if self.UpdVar.Area.values[i,k] >  self.area_surface_bc[i]:
                                self.detr_sc[i,k-1] = (self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1]*w_low/
                                                   (dz * self.area_surface_bc[i] * w_mid * self.Ref.rho0_half[k]) + self.entr_sc[i,k-1] -1.0/dz)
                                self.detr_w[i,k-1] = self.detr_sc[i,k-1]
                                self.UpdVar.Area.values[i,k] = self.area_surface_bc[i]
                        else:
                            # the updraft has terminated so set its area fraction to zero at this height and all heights above
                            self.UpdVar.Area.values[i,k] = 0.0
                            self.UpdVar.H.values[i,k] = GMV.H.values[k]
                            self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                            self.UpdVar.B.values[i,k] = 0.0
                            self.UpdVar.QL.values[i,k] = 0.0


        self.UpdVar.Area.set_bcs(self.Gr)
        return

    cpdef solve_updraft_scalars(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, i
            double dz = self.Gr.dz
            Py_ssize_t gw = self.Gr.gw
            bint above_cloudbase = False
            eos_struct sa
            entr_struct ret


        with nogil:
            for i in xrange(self.n_updrafts):
                above_cloudbase = False
                self.UpdVar.H.values[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.values[i,gw] = self.qt_surface_bc[i]
                # Find the cloud liquid content
                sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[gw],
                         self.UpdVar.QT.values[i,gw], self.UpdVar.H.values[i,gw])
                self.UpdVar.QL.values[i,gw] = sa.ql
                self.UpdVar.T.values[i,gw] = sa.T
                #
                above_cloudbase = set_cloudbase_flag(self.UpdVar.QL.values[i,gw], above_cloudbase)
                ret = entr_detr(self.Gr.z[gw], self.Gr.z_half[gw], above_cloudbase)


                for k in xrange(gw+1, self.Gr.nzg-gw):
                    # self.UpdVar.H.values[i,k] = (0.5 * dz* ret.entr_sc* (GMV.H.values[k] + GMV.H.values[k-1] - self.UpdVar.H.values[i,k-1]) + self.UpdVar.H.values[i,k-1])/(1.0 + 0.5*dz*ret.entr_sc)
                    # self.UpdVar.QT.values[i,k] = (0.5 * dz* ret.entr_sc* (GMV.QT.values[k] + GMV.QT.values[k-1] - self.UpdVar.QT.values[i,k-1]) + self.UpdVar.QT.values[i,k-1])/(1.0 + 0.5*dz*ret.entr_sc)

                    self.UpdVar.H.values[i,k] = self.UpdVar.H.values[i,k-1] - ret.entr_sc * dz * (self.UpdVar.H.values[i,k-1]-self.EnvVar.H.values[k-1])
                    self.UpdVar.QT.values[i,k] = self.UpdVar.QT.values[i,k-1] - ret.entr_sc * dz * (self.UpdVar.QT.values[i,k-1]-self.EnvVar.QT.values[k-1])
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                    self.UpdVar.QL.values[i,k] = sa.ql
                    self.UpdVar.T.values[i,k] = sa.T

                    above_cloudbase = set_cloudbase_flag(sa.ql, above_cloudbase)
                    ret = entr_detr(self.Gr.z[k], self.Gr.z_half[k], above_cloudbase)


        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
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

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * (mf_tend_h + self.UpdMicro.prec_source_h_tot[k])
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * (mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k])

                # Prepare the output
                self.massflux_tendency_h[k] = mf_tend_h
                self.massflux_tendency_qt[k] = mf_tend_qt
                # self.massflux_h[k] = self.massflux_h[k] * self.Ref.alpha0_half[k]
                # self.massflux_qt[k] = self.massflux_qt[k] * self.Ref.alpha0_half[k]
        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)




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
                self.diffusive_flux_qt[k+gw] = -self.KH.values[k+gw] * 0.5* (GMV.QT.new[k+gw+1] - GMV.QT.new[k+gw-1]) * self.Gr.dzi
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
                self.diffusive_flux_h[k+gw] = -self.KH.values[k+gw] * 0.5* (GMV.H.new[k+gw+1] - GMV.H.new[k+gw-1]) * self.Gr.dzi
            self.diffusive_flux_h[gw-1] = Case.Sur.rho_hflux*self.Ref.alpha0_half[gw]


        return




cdef class SimilarityED(ParameterizationBase):
    def __init__(self, namelist, Grid Gr, ReferenceState Ref):
        ParameterizationBase.__init__(self, Gr, Ref)
        return
    cpdef initialize(self, GridMeanVariables GMV):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        return

    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS ):

        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)

        ParameterizationBase.compute_eddy_diffusivities_similarity(self, GMV, Case)

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double [:] a = np.zeros((nz,),dtype=np.double, order='c')
            double [:] b = np.zeros((nz,),dtype=np.double, order='c')
            double [:] c = np.zeros((nz,),dtype=np.double, order='c')
            double [:] x = np.zeros((nz,),dtype=np.double, order='c')
            double [:] dummy_ae = np.ones((nzg,),dtype=np.double, order='c')
            double [:] rho_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_K_m[k] = 0.5 * (self.KH.values[k]+ self.KH.values[k+1]) * self.Ref.rho0[k]


        # Matrix is the same for all variables that use the same eddy diffusivity
        construct_tridiag_diffusion(nzg, gw, self.Gr.dzi, TS.dt, &rho_K_m[0],
                                    &self.Ref.rho0_half[0], &dummy_ae[0] ,&a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.QT.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * self.Gr.dzi * self.Ref.alpha0_half[gw]

        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                GMV.QT.new[k+gw] = x[k]


        # Solve H
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.H.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * self.Gr.dzi * self.Ref.alpha0_half[gw]

        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                GMV.H.new[k+gw] = x[k]


        ParameterizationBase.update(self, GMV,Case, TS)

        return

    cpdef update_inversion(self, GridMeanVariables GMV, option ):
        ParameterizationBase.update_inversion(self, GMV, option)
        return

    cpdef compute_wstar(self, CasesBase Case):
        ParameterizationBase.compute_wstar(self,Case)
        return







cdef construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return


cdef tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c):
    cdef:
        double * scratch = <double*> PyMem_Malloc(nz * sizeof(double))
        Py_ssize_t i
        double m

    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    with nogil:
        for i in xrange(1,nz):
            m = 1.0/(b[i] - a[i] * scratch[i-1])
            scratch[i] = c[i] * m
            x[i] = (x[i] - a[i] * x[i-1])*m


        for i in xrange(nz-2,-1,-1):
            x[i] = x[i] - scratch[i] * x[i+1]


    PyMem_Free(scratch)
    return


