#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cdef class TimeStepping:

    def __init__(self,namelist):
        try:
            self.dt = namelist['time_stepping']['dt']
        except:
            self.dt = 1.0

        self.dti = 1.0/self.dt

        try:
            self.t_max = namelist['time_stepping']['t_max']
        except:
            self.t_max = 7200.0


        # set time
        self.t = 0.0
        self.nstep = 0


        return

    cpdef update(self):
        self.t += self.dt
        self.nstep += 1
        return