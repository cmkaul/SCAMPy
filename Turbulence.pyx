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
from utility_functions cimport interp2pt
from Turbulence_PrognosticTKE cimport EDMF_PrognosticTKE


def ParameterizationFactory(namelist, paramlist, Grid Gr, ReferenceState Ref):
    scheme = namelist['turbulence']['scheme']
    if scheme == 'EDMF_PrognosticTKE':
        return  EDMF_PrognosticTKE(namelist, paramlist, Gr, Ref)
    elif scheme == 'SimilarityED':
        return SimilarityED(namelist, paramlist, Gr, Ref)
    else:
        print('Did not recognize parameterization ' + scheme)
        return


# A base class common to all turbulence parameterizations
cdef class ParameterizationBase:
    def __init__(self, paramlist, Grid Gr, ReferenceState Ref):
        self.turbulence_tendency  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.Gr = Gr # grid class
        self.Ref = Ref # reference state class
        self.KM = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'diffusivity', 'm^2/s') # eddy viscosity
        self.KH = VariableDiagnostic(Gr.nzg,'half', 'scalar','sym', 'viscosity', 'm^2/s') # eddy diffusivity
        # get values from paramlist
        self.prandtl_number = paramlist['turbulence']['prandtl_number']
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']

        return
    cpdef initialize(self, GridMeanVariables GMV):
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        return

    # Calculate the tendency of the grid mean variables due to turbulence as the difference between the values at the beginning
    # and  end of all substeps taken
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t k

        with nogil:
            for k in xrange(gw,nzg-gw):
                GMV.H.tendencies[k] += (GMV.H.new[k] - GMV.H.values[k]) * TS.dti
                GMV.QT.tendencies[k] += (GMV.QT.new[k] - GMV.QT.values[k]) * TS.dti
                GMV.U.tendencies[k] += (GMV.U.new[k] - GMV.U.values[k]) * TS.dti
                GMV.V.tendencies[k] += (GMV.V.new[k] - GMV.V.values[k]) * TS.dti

        return

    # Update the diagnosis of the inversion height, using the maximum temperature gradient method
    cpdef update_inversion(self, GridMeanVariables GMV, option ):
        cdef:
            double [:] theta_rho = np.zeros((self.Gr.nzg,),dtype=np.double, order='c')
            double qv, grad,  maxgrad = 0.0
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t k, kmin = gw, kmax = self.Gr.nzg-gw

            double Ri_bulk_crit = 0.0

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                qv = GMV.QT.values[k] - GMV.QL.values[k]
                theta_rho[k] = theta_rho_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)


        if option == 'theta_rho':
            with nogil:
                for k in xrange(kmin,kmax):
                    if theta_rho[k] > theta_rho[kmin]:
                        self.zi = self.Gr.z_half[k]
                        break
        elif option == 'thetal_maxgrad':

            with nogil:
                for k in xrange(kmin, kmax):
                    grad =  (GMV.THL.values[k+1] - GMV.THL.values[k])*self.Gr.dzi
                    if grad > maxgrad:
                        maxgrad = grad
                        self.zi = self.Gr.z[k]
        elif option == 'critical_Ri':
            self.zi = get_inversion(&theta_rho[0], &GMV.U.values[0], &GMV.V.values[0], &self.Gr.z_half[0], kmin, kmax, self.Ri_bulk_crit)

        else:
            print('INVERSION HEIGHT OPTION NOT RECOGNIZED')

        # print('Inversion height ', self.zi)

        return



    # Compute eddy diffusivities from similarity theory (Siebesma 2007)
    cpdef compute_eddy_diffusivities_similarity(self, GridMeanVariables GMV, CasesBase Case):
        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        cdef:
            double ustar = Case.Sur.ustar
            double zzi
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz

        with nogil:
            for k in xrange(gw,nzg-gw):
                zzi = self.Gr.z_half[k]/self.zi
                if zzi <= 1.0:
                    if self.wstar<1e-6:
                        self.KH.values[k] = 0.0
                        self.KM.values[k] = 0.0
                    else:
                        self.KH.values[k] = vkb * ( (ustar/self.wstar)**3 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * self.wstar * self.zi
                        self.KM.values[k] = self.KH.values[k] * self.prandtl_number
                else:
                    self.KH.values[k] = 0.0
                    self.KM.values[k] = 0.0


        # Set the boundary points at top and bottom of domain
        self.KH.set_bcs(self.Gr)
        self.KM.set_bcs(self.Gr)
        return

    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV):
        return


#####################################################################################################################




cdef class SimilarityED(ParameterizationBase):
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref):
        self.extrapolate_buoyancy = False
        ParameterizationBase.__init__(self, paramlist, Gr, Ref)
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


        # Solve U
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.U.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_uflux * self.Gr.dzi * self.Ref.alpha0_half[gw]

        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                GMV.U.new[k+gw] = x[k]

        # Solve V
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.V.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_vflux * self.Gr.dzi * self.Ref.alpha0_half[gw]

        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                GMV.V.new[k+gw] = x[k]

        self.update_GMV_diagnostics(GMV)
        ParameterizationBase.update(self, GMV,Case, TS)

        return

    cpdef update_inversion(self, GridMeanVariables GMV, option ):
        ParameterizationBase.update_inversion(self, GMV, option)
        return


    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV):
        # Ideally would write this to be able to use an SGS condensation closure, but unless the need arises,
        # we will just do an all-or-nothing treatment as a placeholder


        GMV.satadjust()


        return
