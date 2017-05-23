from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables, VariablePrognostic

cdef class ForcingBase:
    cdef:
        double [:] subsidence
        double [:] dTdt # horizontal advection temperature tendency
        double [:] dqtdt # horizontal advection moisture tendency
        bint apply_coriolis
        double coriolis_param
        double [:] ug
        double [:] vg
        double (*convert_forcing_prog_fp)(double p0, double qt, double qv, double T,
                                          double qt_tendency, double T_tendency) nogil
        Grid Gr
        ReferenceState Ref

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)

cdef class ForcingNone(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)


cdef class ForcingStandard(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
