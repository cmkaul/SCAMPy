cdef class TimeStepping:
    cdef:
        public double dt
        public double t_max
        public double t
        public double dti
        public Py_ssize_t nstep

    cpdef update(self)