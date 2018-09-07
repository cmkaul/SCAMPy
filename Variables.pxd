from Grid cimport  Grid
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from ReferenceState cimport ReferenceState

cdef class VariablePrognostic:
    cdef:
        double [:] values
        double [:] new
        double [:] mf_update
        double [:] tendencies
        str loc
        str kind
        str bc
        str name
        str units
    cpdef set_bcs(self, Grid Gr)
    cpdef zero_tendencies(self, Grid Gr)


cdef class VariableDiagnostic:
    cdef:
        double [:] values
        str loc
        str kind
        str bc
        str name
        str units
    cpdef set_bcs(self, Grid Gr)



cdef class GridMeanVariables:
    cdef:
        Grid Gr
        ReferenceState Ref
        VariablePrognostic U
        VariablePrognostic V
        VariablePrognostic W
        VariablePrognostic QT
        VariablePrognostic QR
        VariablePrognostic H
        VariableDiagnostic QL
        VariableDiagnostic T
        VariableDiagnostic B
        VariableDiagnostic THL
        VariableDiagnostic TKE
        VariableDiagnostic QTvar
        VariableDiagnostic Hvar
        VariableDiagnostic HQTcov
        VariableDiagnostic THVvar
        double (*t_to_prog_fp)(double p0, double T,  double qt, double ql, double qi)   nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil
        bint calc_tke
        bint calc_scalar_var
        str EnvThermo_scheme

    cpdef zero_tendencies(self)
    cpdef update(self, TimeStepping TS)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef satadjust(self)
