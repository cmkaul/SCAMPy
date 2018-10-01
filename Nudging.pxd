from Variables cimport GridMeanVariables
from Grid cimport Grid
from ReferenceState cimport ReferenceState

cdef class NudgingBase:
    cdef:
        Grid Gr
        ReferenceState Ref
        double [:] relax_coeff
        double [:] t_ref
        double [:] qt_ref
        double [:] u_ref
        double [:] v_ref
        double [:] qt_tendency
        double [:] t_tendency
        double [:] h_tendency
        double [:] u_tendency
        double [:] v_tendency
        bint nudge_uv
        double (*convert_forcing_prog_fp)(double p0, double qt, double qv, double T,
                                          double qt_tendency, double T_tendency) nogil
    cpdef update(self, GridMeanVariables GMV)


# Nudge MEAN temperature, qt, u,v profiles toward a reference profile
cdef class NudgingStandard(NudgingBase):
    cpdef update(self, GridMeanVariables GMV)