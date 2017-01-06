from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport  Grid

cdef class EnvironmentVariable:
    cdef:
        double [:] values
        double [:] tendencies
        double [:] flux
        str loc
        str kind
        str name
        str units

cdef class EnvironmentVariables:
    cdef:
        EnvironmentVariable W
        EnvironmentVariable QT
        EnvironmentVariable QL
        EnvironmentVariable H
        EnvironmentVariable T
        EnvironmentVariable B
        Grid Gr

    cpdef initialize_io(self, NetCDFIO_Stats Stats )
    cpdef io(self, NetCDFIO_Stats Stats)

