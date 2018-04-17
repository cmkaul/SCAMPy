from Variables cimport GridMeanVariables

cdef class NudgingBase:
    cdef:
        double [:] relax_coeff
        double [:] t_ref
        double [:] qt_ref
        double [:] u_ref
        double [:] v_ref
    cpdef update(self)


# Nudge MEAN temperature, qt, u,v profiles toward a reference profile
cdef class NudgingStandard(NudgingBase):
    cpdef initialize(self)
    cpdef update(self, GridMeanVariables GMV)