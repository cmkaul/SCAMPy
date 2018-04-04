#!python
# cython: boundscheck=False
# cython: wraparound=True
# cython: initializedcheck=False
# cython: cdivision=True

#Adapated from PyCLES: https://github.com/pressel/pycles

from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats
cimport numpy as np
import numpy as np
import pylab as plt

from scipy.integrate import odeint
from thermodynamic_functions cimport t_to_entropy_c, eos_first_guess_entropy, eos, alpha_c
include 'parameters.pxi'


cdef class ReferenceState:
    def __init__(self, Grid Gr ):

        self.p0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.p0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        return




    def initialize(self, Grid Gr, NetCDFIO_Stats Stats):
        '''
        Initilize the reference profiles. The function is typically called from the case
        specific initialization fucntion defined in Initialization.pyx
        :param Gr: Grid class
        :param Thermodynamics: Thermodynamics class
        :param NS: StatsIO class
        :param Pa:  ParallelMPI class
        :return:
        '''

        self.sg = t_to_entropy_c(self.Pg, self.Tg, self.qtg, 0.0, 0.0)


        # Form a right hand side for integrating the hydrostatic equation to
        # determine the reference pressure
        ##_____________TO COMPILE______________
        def rhs(p, z):
            ret =  eos(t_to_entropy_c, eos_first_guess_entropy, np.exp(p),  self.qtg, self.sg)
            q_i = 0.0
            q_l = ret.ql
            T = ret.T
            return -g / (Rd * T * (1.0 - self.qtg + eps_vi * (self.qtg - q_l - q_i)))



        ##_____________TO COMPILE______________

        # Construct arrays for integration points
        z = np.array(Gr.z[Gr.gw - 1:-Gr.gw + 1])
        z_half = np.append([0.0], np.array(Gr.z_half[Gr.gw:-Gr.gw]))

        # We are integrating the log pressure so need to take the log of the
        # surface pressure
        p0 = np.log(self.Pg)

        p = np.zeros(Gr.nzg, dtype=np.double, order='c')
        p_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        # Perform the integration
        p[Gr.gw - 1:-Gr.gw +1] = odeint(rhs, p0, z, hmax=1.0)[:, 0]
        p_half[Gr.gw:-Gr.gw] = odeint(rhs, p0, z_half, hmax=1.0)[1:, 0]

        # Set boundary conditions
        p[:Gr.gw - 1] = p[2 * Gr.gw - 2:Gr.gw - 1:-1]
        p[-Gr.gw + 1:] = p[-Gr.gw - 1:-2 * Gr.gw:-1]

        p_half[:Gr.gw] = p_half[2 * Gr.gw - 1:Gr.gw - 1:-1]
        p_half[-Gr.gw:] = p_half[-Gr.gw - 1:-2 * Gr.gw - 1:-1]

        p = np.exp(p)
        p_half = np.exp(p_half)


        cdef double[:] p_ = p
        cdef double[:] p_half_ = p_half
        cdef double[:] temperature = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] temperature_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi_half = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv_half = np.zeros(Gr.nzg, dtype=np.double, order='c')

        # Compute reference state thermodynamic profiles
        #_____COMMENTED TO TEST COMPILATION_____________________
        for k in xrange(Gr.nzg):
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_[k], self.qtg, self.sg)
            temperature[k] = ret.T
            ql[k] = ret.ql
            qv[k] = self.qtg - (ql[k] + qi[k])
            alpha[k] = alpha_c(p_[k], temperature[k], self.qtg, qv[k])
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_half_[k], self.qtg, self.sg)
            temperature_half[k] = ret.T
            ql_half[k] = ret.ql
            qv_half[k] = self.qtg - (ql_half[k] + qi_half[k])
            alpha_half[k] = alpha_c(p_half_[k], temperature_half[k], self.qtg, qv_half[k])

        # Now do a sanity check to make sure that the Reference State entropy profile is uniform following
        # saturation adjustment
        cdef double s
        for k in xrange(Gr.nzg):
            s = t_to_entropy_c(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])
            if np.abs(s - self.sg)/self.sg > 0.01:
                print('Error in reference profiles entropy not constant !')
                print('Likely error in saturation adjustment')





        # print(np.array(Gr.extract_local_ghosted(alpha_half,2)))
        self.alpha0_half = alpha_half
        self.alpha0 = alpha
        self.p0 = p_
        self.p0_half = p_half
        self.rho0 = 1.0 / np.array(self.alpha0)
        self.rho0_half = 1.0 / np.array(self.alpha0_half)

        Stats.add_reference_profile('alpha0')
        Stats.write_reference_profile('alpha0', alpha[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('alpha0_half')
        Stats.write_reference_profile('alpha0_half', alpha_half[Gr.gw:-Gr.gw])


        Stats.add_reference_profile('p0')
        Stats.write_reference_profile('p0', p_[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('p0_half')
        Stats.write_reference_profile('p0_half', p_half[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('rho0')
        Stats.write_reference_profile('rho0', 1.0 / np.array(alpha[Gr.gw:-Gr.gw]))
        Stats.add_reference_profile('rho0_half')
        Stats.write_reference_profile('rho0_half', 1.0 / np.array(alpha_half[Gr.gw:-Gr.gw]))

        # Stats.add_reference_profile('temperature0', Gr, Pa)
        # Stats.write_reference_profile('temperature0', temperature_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('ql0', Gr, Pa)
        # Stats.write_reference_profile('ql0', ql_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('qv0', Gr, Pa)
        # Stats.write_reference_profile('qv0', qv_half[Gr.dims.gw:-Gr.dims.gw], Pa)
        # Stats.add_reference_profile('qi0', Gr, Pa)
        # Stats.write_reference_profile('qi0', qi_half[Gr.dims.gw:-Gr.dims.gw], Pa)


        return

