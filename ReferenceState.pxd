from Grid cimport Grid
from NetCDFIO cimport NetCDFIO_Stats

cdef class ReferenceState:
    cdef:
        double [:] p0
        double [:] p0_half
        double [:] alpha0
        double [:] alpha0_half
        double [:] rho0
        double [:] rho0_half


        double sg
        double Tg  #Temperature at ground level
        double Pg  #Pressure at ground level
        double qtg #Surface total water mixing ratio
        double u0 #u velocity removed in Galilean transformation
        double v0 #v velocity removed in Galilean transformation


