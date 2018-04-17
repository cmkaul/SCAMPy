cdef class ForcingReferenceBase:
    cdef:
        double sst
        Py_ssize_t npressure
        double [:] pressure
        double [:] s
        double [:] qt
        double [:] temperature
        double [:] rv
        double [:] u
        double [:] v
        bint is_init
    cpdef initialize(self)
    cpdef update(self)
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)

cdef class ForcingReferenceNone(ForcingReferenceBase):
    cpdef initialize(self)
    cpdef update(self)
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
    cdef:
        str filename
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self)
    cpdef update(self)