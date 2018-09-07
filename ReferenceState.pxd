from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats

cdef class ReferenceState:
    cdef:
        double [:] p0
        double [:] alpha0
        double [:] rho0


        double sg
        double Tg  #Temperature at ground level
        double Pg  #Pressure at ground level
        double qtg #Surface total water mixing ratio
        double u0 #u velocity removed in Galilean transformation
        double v0 #v velocity removed in Galilean transformation


