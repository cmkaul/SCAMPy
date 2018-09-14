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

from scipy.integrate import odeint
from thermodynamic_functions cimport t_to_entropy_c, eos_first_guess_entropy, eos, alpha_c
include 'parameters.pxi'


cdef class ReferenceState:
    def __init__(self, Grid Gr ):

        self.p0_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.p0_c = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.alpha0_c = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        self.rho0_c = np.zeros(Gr.nzg, dtype=np.double, order='c')

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
        z_f = np.array(Gr.z_f[Gr.gw - 1:-Gr.gw + 1])
        z_c = np.append([0.0], np.array(Gr.z_c[Gr.gw:-Gr.gw]))

        # We are integrating the log pressure so need to take the log of the
        # surface pressure
        p0 = np.log(self.Pg)

        p_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        p_c = np.zeros(Gr.nzg, dtype=np.double, order='c')

        # Perform the integration
        p_f[Gr.gw - 1:-Gr.gw +1] = odeint(rhs, p0, z_f, hmax=1.0)[:, 0]
        p_c[Gr.gw:-Gr.gw] = odeint(rhs, p0, z_c, hmax=1.0)[1:, 0]

        # Set boundary conditions
        p_f[:Gr.gw - 1] = p_f[2 * Gr.gw - 2:Gr.gw - 1:-1]
        p_f[-Gr.gw + 1:] = p_f[-Gr.gw - 1:-2 * Gr.gw:-1]

        p_c[:Gr.gw] = p_c[2 * Gr.gw - 1:Gr.gw - 1:-1]
        p_c[-Gr.gw:] = p_c[-Gr.gw - 1:-2 * Gr.gw - 1:-1]

        p_f = np.exp(p_f)
        p_c = np.exp(p_c)


        cdef double[:] p_f_ = p_f
        cdef double[:] p_c_ = p_c
        cdef double[:] temperature_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] temperature_c = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha0_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] alpha0_c = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi_f = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv_f = np.zeros(Gr.nzg, dtype=np.double, order='c')

        cdef double[:] ql_c = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qi_c = np.zeros(Gr.nzg, dtype=np.double, order='c')
        cdef double[:] qv_c = np.zeros(Gr.nzg, dtype=np.double, order='c')

        # Compute reference state thermodynamic profiles
        #_____COMMENTED TO TEST COMPILATION_____________________
        for k in xrange(Gr.nzg):
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_f_[k], self.qtg, self.sg)
            temperature_f[k] = ret.T
            ql_f[k] = ret.ql
            qv_f[k] = self.qtg - (ql_f[k] + qi_f[k])
            alpha0_f[k] = alpha_c(p_f_[k], temperature_f[k], self.qtg, qv_f[k])
            ret = eos(t_to_entropy_c, eos_first_guess_entropy, p_c_[k], self.qtg, self.sg)
            temperature_c[k] = ret.T
            ql_c[k] = ret.ql
            qv_c[k] = self.qtg - (ql_c[k] + qi_c[k])
            alpha0_c[k] = alpha_c(p_c_[k], temperature_c[k], self.qtg, qv_c[k])



        # Now do a sanity check to make sure that the Reference State entropy profile is uniform following
        # saturation adjustment
        cdef double s
        for k in xrange(Gr.nzg):
            s = t_to_entropy_c(p_c[k],temperature_c[k],self.qtg,ql_c[k],qi_c[k])
            if np.abs(s - self.sg)/self.sg > 0.01:
                print('Error in reference profiles entropy not constant !')
                print('Likely error in saturation adjustment')

        self.alpha0_c = alpha0_c
        self.alpha0_f = alpha0_f
        self.p0_f = p_f_
        self.p0_c = p_c_
        self.rho0_c = 1.0 / np.array(self.alpha0_c)
        self.rho0_f = 1.0 / np.array(self.alpha0_f)

        Stats.add_reference_profile('alpha0_f')
        Stats.write_reference_profile('alpha0_f', alpha0_f[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('alpha0_c')
        Stats.write_reference_profile('alpha0_c', alpha0_c[Gr.gw:-Gr.gw])


        Stats.add_reference_profile('p0_f')
        Stats.write_reference_profile('p0_f', p_f[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('p0_c')
        Stats.write_reference_profile('p0_c', p_c[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('rho0_f')
        Stats.write_reference_profile('rho0_f', 1.0 / np.array(alpha0_f[Gr.gw:-Gr.gw]))
        Stats.add_reference_profile('rho0_c')
        Stats.write_reference_profile('rho0_c', 1.0 / np.array(alpha0_c[Gr.gw:-Gr.gw]))

        # Stats.add_reference_profile('temperature0', Gr, Pa)
        # Stats.add_reference_profile('ql0', Gr, Pa)
        # Stats.add_reference_profile('qv0', Gr, Pa)
        # Stats.add_reference_profile('qi0', Gr, Pa)

        return

