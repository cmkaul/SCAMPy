from Grid cimport Grid
from ReferenceState cimport ReferenceState
from Variables cimport GridMeanVariables, VariablePrognostic
from NetCDFIO cimport  NetCDFIO_Stats

cdef class ForcingBase:
    cdef:
        double [:] subsidence
        double [:] dTdt # horizontal advection temperature tendency
        double [:] dqtdt # horizontal advection moisture tendency
        bint apply_coriolis
        bint apply_subsidence
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
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingNone(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingStandard(ForcingBase):
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)

# cdef class ForcingRadiative(ForcingBase):
#     cpdef initialize(self, GridMeanVariables GMV)
#     cpdef update(self, GridMeanVariables GMV)
#     cpdef initialize_io(self, NetCDFIO_Stats Stats)
#     cpdef io(self, NetCDFIO_Stats Stats)

cdef class ForcingDYCOMS_RF01(ForcingBase):
    cdef:
        double alpha_z
        double kappa
        double F0
        double F1
        double divergence
        double [:] f_rad # radiative flux at cell edges

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef calculate_radiation(self, GridMeanVariables GMV)
    cpdef update(self, GridMeanVariables GMV)
    cpdef coriolis_force(self, VariablePrognostic U, VariablePrognostic V)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
