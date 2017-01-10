from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
from Surface cimport SurfaceBase
from Forcing cimport ForcingBase
from NetCDFIO cimport  NetCDFIO_Stats


cdef class CasesBase:
    cdef:
        str casename
        SurfaceBase Sur
        ForcingBase Fo
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV)

cdef class Soares(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV)

cdef class Bomex(CasesBase):
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats)
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref )
    cpdef initialize_surface(self, Grid Gr,  ReferenceState Ref )
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV )
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update_surface(self, GridMeanVariables GMV)