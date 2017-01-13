from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables

cdef class ForcingBase:
    cdef:
        double [:] subsidence
        double [:] dTdt # horizontal advection temperature tendency
        double [:] dqtdt # horizontal advection moisture tendency
        double (*convert_forcing_prog_fp)(double p0, double qt, double qv, double T, double qt_tendency, double T_tendency) nogil
        Grid Gr
        ReferenceState Ref
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)

cdef class ForcingNone(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)


cdef class ForcingStandard(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
