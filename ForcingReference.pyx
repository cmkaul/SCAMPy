cimport numpy as np
from Grid cimport Grid

cdef class ForcingReferenceBase:
    def __init__(self):
        return
    cpdef initialize(self):
        return
    cpdef update(self):
        return
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)

cdef class ForcingReferenceNone(ForcingReferenceBase):
    def __init__(self):
        return
    cpdef initialize(self):
        return
    cpdef update(self):
        return
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)

#
# cdef class AdjustedMoistAdiabat(ForcingReferenceBase):
#     cdef:
#         double Tg
#         double Pg
#         double RH_ref
#     cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
#     cpdef eos(self, double p0, double s, double qt)
#     cpdef initialize(self)
#     cpdef update(self)


cdef class ReferenceRCE(ForcingReferenceBase):
    def __init__(self, co2_factor):
        self.filename = str(co2_factor)+'.nc'
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self):
        return
    cpdef update(self):
        return