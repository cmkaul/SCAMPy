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
        self.alpha0 = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0 = np.zeros(Gr.nzg, dtype=np.double, order='c')

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
        z = np.array(Gr.z[Gr.gw-1:-Gr.gw+1])

        #z = np.append([0.0], np.array(Gr.z[Gr.gw:-Gr.gw]))

        # We are integrating the log pressure so need to take the log of the
        # surface pressure
        p0 = np.log(self.Pg)

        p = np.zeros(Gr.nzg, dtype=np.double, order='c')
        print Gr.gw
        print np.shape(np.multiply(Gr.z,1.0))
        print np.shape(np.multiply(z,1.0))
        print np.shape(np.multiply(p,1.0))
        # Perform the integration
        p[Gr.gw - 1:-Gr.gw +1] = odeint(rhs, p0, z, hmax=1.0)[:, 0]

        # Set boundary conditions
        p[:Gr.gw] = p[2 * Gr.gw - 1:Gr.gw - 1:-1]
        p[-Gr.gw:] = p[-Gr.gw - 1:-2 * Gr.gw - 1:-1]

        p = np.exp(p)


        cdef double[:] p_ = p
        cdef double[:] temperature = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv = np.zeros(Gr.nzg, dtype=np.double, order='c')

        # Compute reference state thermodynamic profiles
        #_____COMMENTED TO TEST COMPILATION_____________________
        for k in xrange(Gr.nzg):
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_[k], self.qtg, self.sg)
            temperature[k] = ret.T
            ql[k] = ret.ql
            qv[k] = self.qtg - (ql[k] + qi[k])
            alpha[k] = alpha_c(p_[k], temperature[k], self.qtg, qv[k])
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_[k], self.qtg, self.sg)
            temperature[k] = ret.T
            ql[k] = ret.ql
            qv[k] = self.qtg - (ql[k] + qi[k])
            alpha[k] = alpha_c(p_[k], temperature[k], self.qtg, qv[k])



        # Now do a sanity check to make sure that the Reference State entropy profile is uniform following
        # saturation adjustment
        cdef double s
        for k in xrange(Gr.nzg):
            s = t_to_entropy_c(p[k],temperature[k],self.qtg,ql[k],qi[k])
            if np.abs(s - self.sg)/self.sg > 0.01:
                print('Error in reference profiles entropy not constant !')
                print('Likely error in saturation adjustment')

        self.alpha0 = alpha
        self.p0 = p_
        self.rho0 = 1.0 / np.array(self.alpha0)

        Stats.add_reference_profile('alpha0')
        Stats.write_reference_profile('alpha0', alpha[Gr.gw:-Gr.gw])


        Stats.add_reference_profile('p0')
        Stats.write_reference_profile('p0', p_[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('rho0')
        Stats.write_reference_profile('rho0', 1.0 / np.array(alpha[Gr.gw:-Gr.gw]))

        # Stats.add_reference_profile('temperature0', Gr, Pa)
        # Stats.add_reference_profile('ql0', Gr, Pa)
        # Stats.add_reference_profile('qv0', Gr, Pa)
        # Stats.add_reference_profile('qi0', Gr, Pa)

        return

