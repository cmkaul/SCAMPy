#!python
#cython: boundscheck=True
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
from Variables cimport GridMeanVariables
from forcing_functions cimport  convert_forcing_entropy, convert_forcing_thetal

cdef class ForcingBase:
    def __init__(self):
        return
    cpdef initialize(self, GridMeanVariables GMV):
        self.subsidence = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dTdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        self.dqtdt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
            self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    cpdef update(self, GridMeanVariables GMV):
        return

cdef class ForcingNone(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, GridMeanVariables GMV):
        ForcingBase.initialize(self, GMV)
        return
    cpdef update(self, GridMeanVariables GMV):
        return


cdef class ForcingStandard(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    cpdef initialize(self, GridMeanVariables GMV):
        ForcingBase.initialize(self, GMV)
        return
    cpdef update(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(self.Ref.p0_half[k],GMV.QT.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            GMV.H.tendencies[k] -= (GMV.H.values[k+1]-GMV.H.values[k]) * self.Gr.dzi * self.subsidence[k]
            GMV.QT.tendencies[k] -= (GMV.QT.values[k+1]-GMV.QT.values[k]) * self.Gr.dzi * self.subsidence[k]

        return
