#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
include "parameters.pxi"
import cython

cdef double buoyancy_flux(double shf, double lhf, double T_b, double qt_b, double alpha0_0)

cdef double psi_m_unstable(double zeta, double zeta0)

cdef double psi_m_unstable(double zeta, double zeta0)

cdef double psi_h_unstable(double zeta, double zeta0)

cdef double psi_m_stable(double zeta, double zeta0)

cdef double psi_h_stable(double zeta, double zeta0)

cpdef double entropy_flux(tflux,qtflux, p0_1, T_1, qt_1)


cpdef double compute_ustar(double windspeed, double buoyancy_flux, double z0, double z1)

cdef void exchange_coefficients_byun(double Ri, double zb, double z0, double* cm, double* ch, double* lmo)
