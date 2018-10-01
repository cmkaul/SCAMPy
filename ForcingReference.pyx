cimport numpy as np
from Grid cimport Grid
import netCDF4 as nc

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
        self.filename ='./CGILSdata/RCE_'+ str(int(co2_factor))+'xCO2.nc'
    # cpdef entropy(self,double p0, double T,double qt, double ql, double qi)
    # cpdef eos(self, double p0, double s, double qt)
    cpdef initialize(self):

        data = nc.Dataset(self.filename, 'r')
        self.pressure = data.variables['p_full'][:]
        self.npressure = len(self.pressure)

        self.temperature = data.variables['temp_rc'][:]
        self.qt = data.variables['yv_rc'][:]
        self.u = data.variables['u'][:]
        self.v = data.variables['v'][:]
        return
    cpdef update(self):
        return