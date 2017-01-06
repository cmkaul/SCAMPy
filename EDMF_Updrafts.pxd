cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats

cdef class UpdraftVariable:
    cdef:
        double [:,:] values
        double [:,:] tendencies
        double [:,:] flux
        double [:] bulkvalues
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self, Grid.Grid Gr)

cdef class UpdraftVariables:
    cdef:
        Grid.Grid Gr
        UpdraftVariable W
        UpdraftVariable Area
        UpdraftVariable QT
        UpdraftVariable QL
        UpdraftVariable H
        UpdraftVariable T
        UpdraftVariable B
        Py_ssize_t n_updrafts
    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef set_means(self, GridMeanVariables GMV)

cdef class UpdraftThermodynamics:
    cdef:
        double (*t_to_prog_fp)(double p0, double T,  double qt, double ql, double qi)   nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil
        Grid.Grid Gr
        ReferenceState.ReferenceState Ref
        Py_ssize_t n_updraft

    cpdef satadjust(self, UpdraftVariables UpdVar)
    cpdef buoyancy(self, UpdraftVariables UpdVar, GridMeanVariables GMV)

cdef class UpdraftMicrophysics:
    cdef:
        Grid.Grid Gr
        ReferenceState.ReferenceState Ref
        Py_ssize_t n_updraft
        double f_prec
        double [:,:] prec_source_h
        double [:,:] prec_source_qt
        double [:]  prec_source_h_tot
        double [:] prec_source_qt_tot
    cpdef compute_sources(self, UpdraftVariables UpdVar)
    cpdef update_updraftvars(self, UpdraftVariables UpdVar)
